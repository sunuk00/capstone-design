"""
SegFormer-UNet v2 — 경계 복원 강화 하이브리드 모델

v1 대비 주요 개선 사항:
    1. Deep Supervision  — 각 디코더 레벨에서 보조 손실을 계산해
                           경계·형태 정보가 하위 레이어까지 역전파되도록 강제.
    2. skip_proj 강화    — 1×1 Conv + BN + ReLU 로 skip 특징을 정규화.
    3. ASPP Bridge       — Stage4 출력에 다중 팽창 합성곱(Atrous Spatial Pyramid Pooling)
                           적용. 수용 영역을 넓혀 글로벌-로컬 맥락을 동시에 포착.
    4. FusionNeck 분리   — use_fusion_neck=True 시 h4를 neck 전용으로만 사용하고
                           up1 skip은 별도 저레벨 특징 브랜치에서 공급.
                           저레벨 경계 정보 희석 문제를 해결.
    5. 단계별 업샘플링   — H/4 → H/2 → H 로 2단계 upsample 해
                           마지막 단계의 해상도 점프 문제 완화.

── 아키텍처 개요 ─────────────────────────────────────────────────────────────

[인코더] Mix Transformer (MiT-B{variant})
    Stage 1: H/4  × W/4  (C1)  ← 질감·윤곽
    Stage 2: H/8  × W/8  (C2)
    Stage 3: H/16 × W/16 (C3)
    Stage 4: H/32 × W/32 (C4)  ← 글로벌 맥락

[ASPP Bridge]
    Stage4 출력에 rate=[1,6,12,18] 팽창 합성곱 적용 후 concat → projection
    → (B, bridge_ch, H/32, W/32)

[FusionNeck (선택적)]
    [h4,h8,h16,h32] → 각각 embed_dim projection 후 H/4 합산
    → bridge로 사용 (단, up1 skip은 h4 원본 사용)

[디코더] U-Net CNN Decoder
    up3: bridge  + skip(h16) → (B, D3, H/16)
    up2: d3      + skip(h8)  → (B, D2, H/8)
    up1: d2      + skip(h4)  → (B, D1, H/4)

[출력 헤드]
    H/4 → H/2 → H : 2단계 bilinear + 최종 1×1 Conv

[Deep Supervision (학습 시)]
    aux3: d3 → 1×1 Conv → logit (H/16 해상도)
    aux2: d2 → 1×1 Conv → logit (H/8  해상도)
    학습 손실 = main_loss + λ3·aux3_loss + λ2·aux2_loss

── 출력 규약 ──────────────────────────────────────────────────────────────────
    추론 시: (B, out_channels, H, W) 로짓 텐서
    학습 시: {"main": Tensor, "aux3": Tensor, "aux2": Tensor} dict 반환
             (training=True 일 때만)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_VARIANTS = ("b0", "b2", "b4")

_ENCODER_CHANNELS = {
    "b0": [32,  64,  160, 256],
    "b2": [64,  128, 320, 512],
    "b4": [64,  128, 320, 512],
}

_DECODER_CHANNELS = {
    "b0": [128, 64,  32],
    "b2": [256, 128, 64],
    "b4": [256, 128, 64],
}


# ── 기본 블록 ──────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    """Conv2d + BN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 p: int = 1, d: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvBlock(nn.Sequential):
    """2× (Conv2d + BN + ReLU)"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )


# ── ASPP Bridge ────────────────────────────────────────────────────────────────

class ASPPBridge(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Bridge.

    Stage4 출력(H/32)에 다중 수용 영역을 적용해
    글로벌 맥락과 로컬 세부 정보를 동시에 포착한다.

    처리 흐름:
        x (B, in_ch, H/32, W/32)
            ↓  rate=[1,6,12,18] 팽창 합성곱 + 글로벌 avg pool → 5 branch
            ↓  concat (in_ch×5) → 1×1 Conv projection → out_ch
        출력 (B, out_ch, H/32, W/32)
    """

    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, k=3, p=r, d=r) for r in rates
        ])
        # 글로벌 평균 풀링 브랜치 (이미지 수준 컨텍스트)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 5 branch 합산 후 projection
        self.proj = ConvBNReLU(out_ch * (len(rates) + 1), out_ch, k=1, p=0)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        # 글로벌 풀링 결과를 원본 크기로 복원
        gp = F.interpolate(self.global_pool(x), size=(h, w),
                           mode="bilinear", align_corners=False)
        feats.append(gp)
        out = self.proj(torch.cat(feats, dim=1))
        return self.dropout(out)


# ── FusionNeck ─────────────────────────────────────────────────────────────────

class FusionNeck(nn.Module):
    """
    멀티스케일 특징 융합 모듈.

    4개 MiT 스테이지를 동일 채널(embed_dim)로 projection 후 H/4 해상도로 합산.
    v2 변경: skip_proj에 BN+ReLU 추가로 feature 분포 정규화.

    ※ use_fusion_neck=True 시 h4는 neck 전용.
       up1 의 skip connection 은 원본 h4 를 별도로 사용한다.
    """

    def __init__(self, encoder_channels: list, embed_dim: int = 256) -> None:
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in encoder_channels
        ])

    def forward(self, features: list) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        out = None
        for proj, feat in zip(self.projs, features):
            feat = F.interpolate(feat, size=target_size,
                                 mode="bilinear", align_corners=False)
            feat = proj(feat)
            out = feat if out is None else out + feat
        return out


# ── _UpBlock v2 ────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    """
    U-Net 스타일 업샘플링 블록 (v2).

    변경점:
        - skip_proj: 1×1 Conv → Conv + BN + ReLU (feature 정규화 추가)
        - align_channels 기본값 = out_channels (명시적으로 동일하게)

    처리 흐름:
        x  ──→ bilinear upsample (skip 해상도 맞춤)
        skip → skip_proj (BN+ReLU 포함 1×1 Conv)
        cat([x, projected_skip]) → ConvBlock
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        align_channels: int | None = None,
    ) -> None:
        super().__init__()
        ac = align_channels if align_channels is not None else out_channels
        # BN + ReLU 추가로 skip feature 분포 안정화
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, ac, kernel_size=1, bias=False),
            nn.BatchNorm2d(ac),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(in_channels + ac, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, size=skip.shape[-2:],
                             mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        return self.conv(torch.cat([x, skip], dim=1))


# ── SegFormerUNet v2 ───────────────────────────────────────────────────────────

class SegFormerUNet(nn.Module):
    """
    SegFormer-UNet v2 하이브리드 이진 분할 모델.

    Args:
        out_channels (int)      : 출력 채널. 이진 분할 기본값 1.
        variant (str)           : MiT 크기. "b0" | "b2" | "b4".
        pretrained (bool)       : ImageNet 사전학습 인코더 사용 여부.
        use_fusion_neck (bool)  : FusionNeck으로 멀티스케일 특징 융합 여부.
        use_aspp (bool)         : ASPP Bridge 사용 여부.
                                  use_fusion_neck=True 시 자동으로 비활성화.
        deep_supervision (bool) : 학습 시 보조 출력 헤드 활성화 여부.
                                  추론 시는 항상 main logit만 반환.
    """

    def __init__(
        self,
        out_channels: int = 1,
        variant: str = "b2",
        pretrained: bool = True,
        use_fusion_neck: bool = False,
        use_aspp: bool = True,
        deep_supervision: bool = True,
        v1_compat: bool = False,
    ) -> None:
        super().__init__()

        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"variant는 {_VALID_VARIANTS} 중 하나여야 합니다. 입력값: {variant!r}"
            )

        model_id = f"nvidia/mit-{variant}"

        try:
            from transformers import SegformerConfig, SegformerModel as _HFEncoder
        except ImportError as exc:
            raise ImportError(
                "SegFormerUNet을 사용하려면 transformers 패키지가 필요합니다.\n"
                "pip install transformers"
            ) from exc

        # ── MiT 인코더 ────────────────────────────────────────────────────
        if pretrained:
            try:
                self.encoder = _HFEncoder.from_pretrained(model_id)
                print(f"[SegFormerUNet v2] '{model_id}' 사전학습 가중치 로드 완료.")
            except Exception as exc:
                print(f"[SegFormerUNet v2] 경고: 사전학습 로드 실패 → 랜덤 초기화.\n  원인: {exc}")
                config = SegformerConfig.from_pretrained(model_id)
                self.encoder = _HFEncoder(config)
        else:
            config = SegformerConfig.from_pretrained(model_id)
            self.encoder = _HFEncoder(config)
            print(f"[SegFormerUNet v2] '{model_id}' 랜덤 초기화.")

        # ── 채널 설정 ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = _ENCODER_CHANNELS[variant]
        d3, d2, d1     = _DECODER_CHANNELS[variant]

        # ── FusionNeck / ASPP Bridge ──────────────────────────────────────
        self.use_fusion_neck  = use_fusion_neck
        self.use_aspp         = use_aspp and not use_fusion_neck
        self.deep_supervision = deep_supervision

        if use_fusion_neck:
            # FusionNeck: 4 스테이지 → embed_dim(=d3) 합산
            self.neck = FusionNeck([c1, c2, c3, c4], embed_dim=d3)
            bridge_ch = d3
            # up1 skip 전용: h4 원본을 별도 projection (희석 방지)
            self.h4_skip_proj = nn.Sequential(
                nn.Conv2d(c1, c1, 1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
        elif self.use_aspp:
            self.aspp = ASPPBridge(c4, d3)
            bridge_ch = d3
        else:
            bridge_ch = c4

        # ── 디코더 ────────────────────────────────────────────────────────
        # v1_compat: v1 체크포인트는 skip_proj가 skip_channels 크기로 projection했으므로
        # align_channels=skip_channels 로 맞춰야 conv 입력 채널 수가 일치한다.
        if v1_compat:
            self.up3 = _UpBlock(bridge_ch, c3, d3, align_channels=c3)
            self.up2 = _UpBlock(d3,        c2, d2, align_channels=c2)
            self.up1 = _UpBlock(d2,        c1, d1, align_channels=c1)
        else:
            self.up3 = _UpBlock(bridge_ch, c3, d3)
            self.up2 = _UpBlock(d3,        c2, d2)
            self.up1 = _UpBlock(d2,        c1, d1)

        # ── 출력 헤드 (2단계 업샘플 + 1×1 conv) ──────────────────────────
        # H/4 → H/2 중간 정제 conv
        self.pre_head = ConvBNReLU(d1, d1)
        self.head     = nn.Conv2d(d1, out_channels, kernel_size=1)

        # ── Deep Supervision 보조 헤드 ────────────────────────────────────
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(d3, out_channels, kernel_size=1)  # H/16
            self.aux_head2 = nn.Conv2d(d2, out_channels, kernel_size=1)  # H/8

        self.variant = variant

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """
        Args:
            x: (B, 3, H, W) ImageNet 정규화 텐서

        Returns:
            추론 시 (training=False): (B, out_channels, H, W) 로짓
            학습 시 (training=True, deep_supervision=True):
                {
                  "main": (B, C, H,   W  ),   ← 메인 출력
                  "aux3": (B, C, H/16, W/16),  ← 보조 출력 (깊은 레벨)
                  "aux2": (B, C, H/8,  W/8 ),  ← 보조 출력 (중간 레벨)
                }
        """
        h, w = x.shape[-2:]

        # ── 인코더 ────────────────────────────────────────────────────────
        enc = self.encoder(pixel_values=x, output_hidden_states=True)
        h4, h8, h16, h32 = enc.hidden_states

        # ── Bridge 결정 ───────────────────────────────────────────────────
        if self.use_fusion_neck:
            bridge = self.neck([h4, h8, h16, h32])   # (B, d3, H/4, W/4)
            # up1 skip은 h4 원본(경계 정보 보존)을 별도 projection해 사용
            h4_for_skip = self.h4_skip_proj(h4)
        elif self.use_aspp:
            bridge = self.aspp(h32)                   # (B, d3, H/32, W/32)
            h4_for_skip = h4
        else:
            bridge = h32
            h4_for_skip = h4

        # ── 디코더 ────────────────────────────────────────────────────────
        d3 = self.up3(bridge, h16)    # (B, D3, H/16, W/16)
        d2 = self.up2(d3,    h8)      # (B, D2, H/8,  W/8 )
        d1 = self.up1(d2,    h4_for_skip)  # (B, D1, H/4,  W/4 )

        # ── 메인 출력 헤드: H/4 → H/2 → H ────────────────────────────────
        out = self.pre_head(d1)
        out = F.interpolate(out, size=(h // 2, w // 2),
                            mode="bilinear", align_corners=False)
        out = F.interpolate(out, size=(h, w),
                            mode="bilinear", align_corners=False)
        main_logit = self.head(out)

        # ── Deep Supervision (학습 시만) ──────────────────────────────────
        if self.training and self.deep_supervision:
            return {
                "main": main_logit,
                "aux3": self.aux_head3(d3),   # H/16 해상도 보조 출력
                "aux2": self.aux_head2(d2),   # H/8  해상도 보조 출력
            }

        return main_logit

    def __repr__(self) -> str:
        return (
            f"SegFormerUNet_v2(variant='{self.variant}', "
            f"aspp={self.use_aspp}, "
            f"fusion_neck={self.use_fusion_neck}, "
            f"deep_sup={self.deep_supervision})"
        )


# ── Deep Supervision Loss Wrapper ─────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision을 위한 손실 함수 래퍼.

    메인 출력 + 보조 출력들의 손실을 가중 합산한다.

    사용 예:
        criterion = DeepSupervisionLoss(base_loss_fn, aux_weights=(0.4, 0.2))
        loss = criterion(model_output, target)
        # model_output: dict (training) 또는 Tensor (inference)
        # target: (B, 1, H, W) 마스크

    Args:
        base_loss (nn.Module): 메인 손실 함수 (BCEWithLogitsLoss 계열).
        aux_weights (tuple)  : (aux3 가중치, aux2 가중치). 합이 0.6 이하 권장.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        aux_weights: tuple = (0.4, 0.2),
    ) -> None:
        super().__init__()
        self.base_loss   = base_loss
        self.aux_weights = aux_weights

    @property
    def needs_skeleton(self) -> bool:
        """base_loss의 needs_skeleton 속성을 위임 — engine.py의 배치 언팩 분기용."""
        return getattr(self.base_loss, "needs_skeleton", False)

    def forward(
        self,
        output: dict | torch.Tensor,
        target: torch.Tensor,
        skels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 추론 시 또는 deep_supervision=False 시: 메인 손실만 계산
        if isinstance(output, torch.Tensor):
            if skels is not None:
                return self.base_loss(output, target, skels)
            return self.base_loss(output, target)

        main = output["main"]
        # 메인 손실: skels는 원본 해상도에만 적용 (보조 출력과 해상도 불일치)
        if skels is not None:
            loss = self.base_loss(main, target, skels)
        else:
            loss = self.base_loss(main, target)

        # 보조 출력: target을 해당 해상도로 다운샘플 후 손실 계산 (skels 미적용)
        for key, w in zip(("aux3", "aux2"), self.aux_weights):
            if key in output:
                aux_logit = output[key]
                aux_target = F.interpolate(
                    target.float(), size=aux_logit.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                loss = loss + w * self.base_loss(aux_logit, aux_target)

        return loss


# ── Optimizer 헬퍼 ────────────────────────────────────────────────────────────

def build_optimizer(model: SegFormerUNet,
                    encoder_lr: float = 1e-5,
                    decoder_lr: float = 1e-4,
                    weight_decay: float = 1e-4) -> torch.optim.AdamW:
    """
    인코더/디코더 차등 학습률 AdamW 옵티마이저 생성.

    - 인코더(MiT 사전학습): encoder_lr  (낮게 → 가중치 보호)
    - 디코더 및 부가 모듈 : decoder_lr  (높게 → 빠른 수렴)
    """
    encoder_params = list(model.encoder.parameters())
    encoder_ids    = set(id(p) for p in encoder_params)
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    return torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr,  "weight_decay": weight_decay * 0.1},
        {"params": decoder_params, "lr": decoder_lr,  "weight_decay": weight_decay},
    ])
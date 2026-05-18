"""
SegFormer-UNet v3 — 세밀한 경계 복원 특화 하이브리드 모델

v2 대비 주요 개선 사항:
    1. RefinementBlock    — Inverted Bottleneck (×4 expand) + Depthwise Conv
                           + SE Block + Residual. 디코더 피처 품질 향상.
    2. CBAM               — skip connection에 채널+공간 어텐션 적용.
                           경계 영역에 집중하도록 skip 피처를 선택적으로 강조.
    3. PixelShuffle Head  — 서브픽셀 컨볼루션 기반 업샘플링.
                           Bilinear 대비 고주파 경계 복원 강화.
    4. Dice Loss + DS     — Deep Supervision 보조 손실에 Dice Loss 혼합.
                           클래스 불균형·경계 감도 향상.
    5. 3-group Optimizer  — MiT 하위/상위 스테이지 + 디코더 차등 학습률.

── 아키텍처 개요 ─────────────────────────────────────────────────────────────

[인코더] Mix Transformer (MiT-B{variant})
    Stage 1: H/4  × W/4  (C1)  ← 질감·윤곽
    Stage 2: H/8  × W/8  (C2)
    Stage 3: H/16 × W/16 (C3)
    Stage 4: H/32 × W/32 (C4)  ← 글로벌 맥락

[ASPP Bridge]
    Stage4 출력에 rate=[1,6,12,18] 팽창 합성곱 적용 후 concat → projection

[FusionNeck (선택적)]
    [h4,h8,h16,h32] → 각각 embed_dim projection 후 H/4 합산

[디코더] U-Net CNN Decoder (RefinementBlock + CBAM skip)
    up3: bridge  + CBAM(skip(h16)) → (B, D3, H/16)
    up2: d3      + CBAM(skip(h8))  → (B, D2, H/8)
    up1: d2      + CBAM(skip(h4))  → (B, D1, H/4)

[출력 헤드] PixelShuffle
    H/4 → H/2 → H : 2단계 PixelShuffle + 최종 1×1 Conv

[Deep Supervision (학습 시)]
    aux3: d3 → 1×1 Conv → logit (H/16 해상도)
    aux2: d2 → 1×1 Conv → logit (H/8  해상도)
    보조 손실 = base_loss + 0.5 × Dice

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


class SEBlock(nn.Module):
    """Squeeze-Excitation 채널 재보정."""
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class RefinementBlock(nn.Module):
    """Inverted Bottleneck (×4 expand) + Depthwise 3×3 + SE + Residual."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch * 4
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.se(self.body(x)) + self.shortcut(x))


# ── CBAM ──────────────────────────────────────────────────────────────────────

class _ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (채널 → 공간 순차 어텐션)."""
    def __init__(self, ch: int, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = _ChannelAttention(ch)
        self.sa = _SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── PixelShuffle 출력 헤드 ────────────────────────────────────────────────────

class PixelShuffleHead(nn.Module):
    """2× PixelShuffle 기반 업샘플링 헤드 (H/4 → H/2 → H)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            ConvBNReLU(in_ch, in_ch),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            ConvBNReLU(in_ch, in_ch),
        )
        self.head = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.up2(self.up1(x)))


# ── ASPP Bridge ────────────────────────────────────────────────────────────────

class ASPPBridge(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Bridge.

    Stage4 출력(H/32)에 다중 수용 영역을 적용해
    글로벌 맥락과 로컬 세부 정보를 동시에 포착한다.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, k=3, p=r, d=r) for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.proj = ConvBNReLU(out_ch * (len(rates) + 1), out_ch, k=1, p=0)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.global_pool(x), size=(h, w),
                           mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.dropout(self.proj(torch.cat(feats, dim=1)))


# ── FusionNeck ─────────────────────────────────────────────────────────────────

class FusionNeck(nn.Module):
    """멀티스케일 특징 융합 모듈 (4 스테이지 → embed_dim → H/4 합산)."""

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


# ── _UpBlock v3 (RefinementBlock + CBAM) ──────────────────────────────────────

class _UpBlock(nn.Module):
    """
    U-Net 스타일 업샘플링 블록 (v3).

    skip → skip_proj → CBAM → cat([upsample(x), skip]) → RefinementBlock
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        align_channels: int | None = None,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        ac = align_channels if align_channels is not None else out_channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, ac, kernel_size=1, bias=False),
            nn.BatchNorm2d(ac),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(ac, spatial_kernel_size)
        self.conv = RefinementBlock(in_channels + ac, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, size=skip.shape[-2:],
                             mode="bilinear", align_corners=False)
        skip = self.cbam(self.skip_proj(skip))
        return self.conv(torch.cat([x, skip], dim=1))


# ── SegFormerUNet v3 ───────────────────────────────────────────────────────────

class SegFormerUNet(nn.Module):
    """
    SegFormer-UNet v3 하이브리드 이진 분할 모델.

    Args:
        out_channels (int)      : 출력 채널. 이진 분할 기본값 1.
        variant (str)           : MiT 크기. "b0" | "b2" | "b4".
        pretrained (bool)       : ImageNet 사전학습 인코더 사용 여부.
        use_fusion_neck (bool)  : FusionNeck으로 멀티스케일 특징 융합 여부.
        use_aspp (bool)         : ASPP Bridge 사용 여부.
                                  use_fusion_neck=True 시 자동으로 비활성화.
        deep_supervision (bool) : 학습 시 보조 출력 헤드 활성화 여부.
        v1_compat (bool)        : v1 체크포인트 align_channels 호환 모드.
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
                print(f"[SegFormerUNet v3] '{model_id}' 사전학습 가중치 로드 완료.")
            except Exception as exc:
                print(f"[SegFormerUNet v3] 경고: 사전학습 로드 실패 → 랜덤 초기화.\n  원인: {exc}")
                config = SegformerConfig.from_pretrained(model_id)
                self.encoder = _HFEncoder(config)
        else:
            config = SegformerConfig.from_pretrained(model_id)
            self.encoder = _HFEncoder(config)
            print(f"[SegFormerUNet v3] '{model_id}' 랜덤 초기화.")

        # ── 채널 설정 ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = _ENCODER_CHANNELS[variant]
        d3, d2, d1     = _DECODER_CHANNELS[variant]

        # ── FusionNeck / ASPP Bridge ──────────────────────────────────────
        self.use_fusion_neck  = use_fusion_neck
        self.use_aspp         = use_aspp and not use_fusion_neck
        self.deep_supervision = deep_supervision

        if use_fusion_neck:
            self.neck = FusionNeck([c1, c2, c3, c4], embed_dim=d3)
            bridge_ch = d3
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
        if v1_compat:
            self.up3 = _UpBlock(bridge_ch, c3, d3, align_channels=c3, spatial_kernel_size=3)
            self.up2 = _UpBlock(d3,        c2, d2, align_channels=c2, spatial_kernel_size=3)
            self.up1 = _UpBlock(d2,        c1, d1, align_channels=c1, spatial_kernel_size=7)
        else:
            self.up3 = _UpBlock(bridge_ch, c3, d3, spatial_kernel_size=3)
            self.up2 = _UpBlock(d3,        c2, d2, spatial_kernel_size=3)
            self.up1 = _UpBlock(d2,        c1, d1, spatial_kernel_size=7)

        # ── PixelShuffle 출력 헤드 (H/4 → H) ─────────────────────────────
        self.pixel_shuffle_head = PixelShuffleHead(d1, out_channels)

        # ── Deep Supervision 보조 헤드 ────────────────────────────────────
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head2 = nn.Conv2d(d2, out_channels, kernel_size=1)

        self.variant = variant

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """
        Args:
            x: (B, 3, H, W) ImageNet 정규화 텐서

        Returns:
            추론 시 (training=False): (B, out_channels, H, W) 로짓
            학습 시 (training=True, deep_supervision=True):
                {"main": Tensor, "aux3": Tensor, "aux2": Tensor}
        """
        h, w = x.shape[-2:]

        # ── 인코더 ────────────────────────────────────────────────────────
        enc = self.encoder(pixel_values=x, output_hidden_states=True)
        h4, h8, h16, h32 = enc.hidden_states

        # ── Bridge 결정 ───────────────────────────────────────────────────
        if self.use_fusion_neck:
            bridge = self.neck([h4, h8, h16, h32])
            h4_for_skip = self.h4_skip_proj(h4)
        elif self.use_aspp:
            bridge = self.aspp(h32)
            h4_for_skip = h4
        else:
            bridge = h32
            h4_for_skip = h4

        # ── 디코더 ────────────────────────────────────────────────────────
        d3 = self.up3(bridge, h16)
        d2 = self.up2(d3,    h8)
        d1 = self.up1(d2,    h4_for_skip)

        # ── 메인 출력: H/4 → H (PixelShuffle 2단계) ──────────────────────
        main_logit = self.pixel_shuffle_head(d1)
        # H가 4의 배수가 아닌 경우 입력 해상도 보정
        if main_logit.shape[-2:] != (h, w):
            main_logit = F.interpolate(main_logit, size=(h, w),
                                       mode="bilinear", align_corners=False)

        # ── Deep Supervision (학습 시만) ──────────────────────────────────
        if self.training and self.deep_supervision:
            return {
                "main": main_logit,
                "aux3": self.aux_head3(d3),
                "aux2": self.aux_head2(d2),
            }

        return main_logit

    def __repr__(self) -> str:
        return (
            f"SegFormerUNet_v3(variant='{self.variant}', "
            f"aspp={self.use_aspp}, "
            f"fusion_neck={self.use_fusion_neck}, "
            f"deep_sup={self.deep_supervision}, "
            f"pixel_shuffle=True, cbam=True)"
        )


# ── Deep Supervision Loss Wrapper ─────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision을 위한 손실 함수 래퍼.

    메인 출력 + 보조 출력들의 손실을 가중 합산한다.
    보조 손실 = base_loss + 0.5 × Dice

    Args:
        base_loss (nn.Module): 메인 손실 함수 (BCEWithLogitsLoss 계열).
        aux_weights (tuple)  : (aux3 가중치, aux2 가중치).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        aux_weights: tuple = (0.5, 0.3),
    ) -> None:
        super().__init__()
        self.base_loss   = base_loss
        self.aux_weights = aux_weights

    @property
    def needs_skeleton(self) -> bool:
        """base_loss의 needs_skeleton 속성을 위임 — engine.py의 배치 언팩 분기용."""
        return getattr(self.base_loss, "needs_skeleton", False)

    @staticmethod
    def dice_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(logit)
        inter = (prob * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + 1) / (prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
        return dice.mean()

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
        # 메인 손실: skels는 원본 해상도에만 적용
        if skels is not None:
            loss = self.base_loss(main, target, skels)
        else:
            loss = self.base_loss(main, target)

        # 보조 출력: base_loss + 0.5 × Dice
        for key, w in zip(("aux3", "aux2"), self.aux_weights):
            if key in output:
                aux_logit  = output[key]
                aux_target = F.interpolate(
                    target.float(), size=aux_logit.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                aux_loss = (
                    self.base_loss(aux_logit, aux_target)
                    + 0.5 * self.dice_loss(aux_logit, aux_target)
                )
                loss = loss + w * aux_loss

        return loss


# ── Optimizer 헬퍼 ────────────────────────────────────────────────────────────

def build_optimizer(model: SegFormerUNet,
                    encoder_lr: float = 1e-5,
                    decoder_lr: float = 1e-4,
                    weight_decay: float = 1e-4) -> torch.optim.AdamW:
    """
    3-group AdamW 옵티마이저.

    - 그룹 1 (encoder 하위 스테이지 0,1): encoder_lr × 0.1
    - 그룹 2 (encoder 상위 스테이지 2,3): encoder_lr
    - 그룹 3 (디코더 전체):               decoder_lr
    """
    lower_ids: set = set()
    upper_ids: set = set()

    enc = model.encoder.encoder
    for attr in ("patch_embeddings", "block", "layer_norm"):
        module_list = getattr(enc, attr, None)
        if module_list is None:
            continue
        for idx, stage in enumerate(module_list):
            target = lower_ids if idx < 2 else upper_ids
            for p in stage.parameters():
                target.add(id(p))

    all_enc_ids    = lower_ids | upper_ids
    lower_params   = [p for p in model.encoder.parameters() if id(p) in lower_ids]
    upper_params   = [p for p in model.encoder.parameters() if id(p) in upper_ids]
    decoder_params = [p for p in model.parameters() if id(p) not in all_enc_ids]

    return torch.optim.AdamW([
        {"params": lower_params,   "lr": encoder_lr * 0.1, "weight_decay": weight_decay * 0.1},
        {"params": upper_params,   "lr": encoder_lr,       "weight_decay": weight_decay * 0.1},
        {"params": decoder_params, "lr": decoder_lr,       "weight_decay": weight_decay},
    ])

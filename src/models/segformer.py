"""
SegFormer 모델 래퍼

NVIDIA가 제안한 SegFormer 아키텍처를 마라톤 경로 이진 분할 태스크에 맞게 래핑한 모듈.
HuggingFace transformers 라이브러리의 사전학습 MiT 인코더를 활용한다.

지원 변형(variant):
    b0 - 경량 모델 (파라미터 약 3.7M). 빠른 실험 및 리소스 제한 환경에 적합.
    b2 - 중형 모델 (파라미터 약 27.4M). 정확도와 속도의 균형이 잘 잡힌 추천 선택지.
    b4 - 대형 모델 (파라미터 약 64.1M). 성능 우선 환경에 적합하지만 학습이 느림.

사용 시 주의사항:
    - 이 모델은 입력에 ImageNet 정규화(평균·표준편차 보정)가 적용된 텐서를 기대한다.
      dataset.py 의 apply_model_preprocess() 에서 자동으로 처리된다.
    - 첫 실행 시 HuggingFace Hub에서 사전학습 가중치를 다운로드한다 (~23–244 MB).
      이후에는 로컬 캐시를 사용하므로 인터넷 연결이 불필요하다.
    - 출력은 sigmoid 를 거치지 않은 로짓(logit) 값이다.
      기존 모델(UNet 등)과 동일한 규약이므로 손실 함수를 그대로 사용할 수 있다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 지원하는 MiT 백본 크기 목록
_VALID_VARIANTS = ("b0", "b2", "b4")


class SegFormerSeg(nn.Module):
    """
    SegFormer 이진 분할 모델

    ── 아키텍처 개요 ──────────────────────────────────────────────────────────

    [인코더] Mix Transformer (MiT-B{variant})
        - 이미지를 4개 스테이지로 나눠 계층적으로 특징을 추출한다.
        - 스테이지별 출력 해상도: H/4, H/8, H/16, H/32 (채널은 반대로 점점 증가)
        - 각 스테이지 내부 구성:
            1) Overlapping Patch Embedding
               · 겹치는 패치로 지역적 연속성을 보존하면서 토큰 시퀀스 생성
            2) Efficient Self-Attention (Sequence Reduction Attention)
               · Key/Value 시퀀스를 R² 비율로 줄여(Reduction) 연산량을 크게 감소시킴
               · 예: 해상도 64×64인 특징 맵에서 R=8이면 K/V 길이가 64 → 1로 압축
            3) Mix-FFN
               · 표준 FFN에 3×3 depth-wise conv 를 삽입해 위치 정보를 암묵적으로 인코딩
               · 기존 트랜스포머의 Position Encoding 을 대체

    [디코더] All-MLP Decoder Head
        - 4개 스테이지의 특징 맵을 모두 1/4 해상도로 업샘플링한 뒤 채널 수를 통일한다.
        - 통일된 특징 맵을 채널 방향으로 이어붙인(concat) 후 MLP 2단계로 최종 예측 생성.
        - 단순한 구조임에도 다중 스케일 정보를 효과적으로 융합한다.

    [출력 업샘플링]
        - 디코더 출력 해상도는 H/4 × W/4 이다.
        - Bilinear Interpolation 으로 원본 입력 해상도(H × W)까지 되돌린다.

    ── 출력 규약 ──────────────────────────────────────────────────────────────
        - 출력 형태: (B, out_channels, H, W)
        - 값 범위: 로짓(sigmoid 미적용). BCEWithLogitsLoss 계열 손실 함수와 함께 사용.

    Args:
        out_channels (int): 출력 채널 수. 이진 분할이므로 기본값 1.
        variant (str): MiT 백본 크기. "b0", "b2", "b4" 중 선택.
        pretrained (bool): True 이면 ImageNet 사전학습 인코더 가중치를 불러온다.
                           False 이면 인코더도 랜덤 초기화한다.
    """

    def __init__(
        self,
        out_channels: int = 1,
        variant: str = "b0",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"variant는 {_VALID_VARIANTS} 중 하나여야 합니다. 입력값: {variant!r}"
            )

        # HuggingFace Hub 모델 식별자 (e.g. "nvidia/mit-b0")
        model_id = f"nvidia/mit-{variant}"

        # transformers를 여기서 import 하여, 이 패키지가 없을 때
        # 다른 모델(UNet 등)의 import 에는 영향을 주지 않도록 격리한다.
        try:
            from transformers import (
                SegformerConfig,
                SegformerForSemanticSegmentation,
                SegformerModel as _HFEncoder,
            )
        except ImportError as exc:
            raise ImportError(
                "SegFormer를 사용하려면 transformers 패키지가 필요합니다.\n"
                "설치 명령어: pip install transformers"
            ) from exc

        # ── 설정(Config) 준비 ─────────────────────────────────────────────
        # 사전학습 모델과 동일한 인코더 구조 파라미터(채널 수, 레이어 수 등)를 유지하면서
        # 출력 클래스 수만 우리 태스크(이진 분할)에 맞게 덮어쓴다.
        config = SegformerConfig.from_pretrained(model_id)
        config.num_labels = out_channels
        config.id2label = {i: str(i) for i in range(out_channels)}
        config.label2id = {str(i): i for i in range(out_channels)}

        # ── 모델 초기화 ───────────────────────────────────────────────────
        # SegformerForSemanticSegmentation = MiT 인코더 + All-MLP 디코더 헤드
        # 이 시점에서 디코더 헤드의 최종 출력 채널이 out_channels=1 로 세팅된다.
        self.model = SegformerForSemanticSegmentation(config)

        # ── 사전학습 인코더 가중치 로드 ───────────────────────────────────
        if pretrained:
            # nvidia/mit-{variant} 는 인코더(MiT)만 포함한 모델이다.
            # 이 가중치를 self.model 의 인코더 부분(self.model.segformer)에만 복사한다.
            # 디코더 헤드는 out_channels=1 로 새로 초기화된 상태를 그대로 유지한다.
            try:
                pretrained_encoder = _HFEncoder.from_pretrained(model_id)
                self.model.segformer.load_state_dict(pretrained_encoder.state_dict())
                print(f"[SegFormer] '{model_id}' 사전학습 인코더 가중치 로드 완료.")
            except Exception as exc:
                # 네트워크 불가 등 예외 발생 시 경고만 출력하고 랜덤 초기화로 계속 진행
                print(
                    f"[SegFormer] 경고: 사전학습 가중치 로드 실패 → 랜덤 초기화로 진행합니다.\n"
                    f"  원인: {exc}"
                )
        else:
            print(f"[SegFormer] '{model_id}' 설정으로 랜덤 초기화합니다.")

        # 나중에 __repr__ 등에서 참조할 수 있도록 저장
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass)

        처리 흐름:
            입력 (B, 3, H, W)
                ↓  MiT 인코더 — 4스테이지 계층적 특징 추출
            특징 맵 4개 (H/4, H/8, H/16, H/32)
                ↓  All-MLP 디코더 — 멀티스케일 융합
            로짓 (B, out_channels, H/4, W/4)
                ↓  Bilinear Interpolation — 원본 해상도 복원
            출력 (B, out_channels, H, W)

        Args:
            x: ImageNet 정규화가 적용된 입력 이미지 텐서 (B, 3, H, W)

        Returns:
            분할 로짓 텐서 (B, out_channels, H, W). Sigmoid 미적용.
        """
        # 최종 업샘플링 목표 해상도를 입력으로부터 저장
        h, w = x.shape[-2:]

        # HuggingFace 모델은 키워드 인수 pixel_values 로 입력을 받는다.
        # out.logits 의 형태: (B, out_channels, H/4, W/4)
        out = self.model(pixel_values=x)

        # H/4 해상도의 로짓을 원본 해상도(H, W)로 업샘플링한다.
        # · mode="bilinear" : 주변 4픽셀 가중 평균 → 경계가 부드러움
        # · align_corners=False : 픽셀을 면적의 중심으로 정렬 (torchvision 권장)
        return F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)

    def __repr__(self) -> str:
        return f"SegFormerSeg(variant='b{self.variant}', model={self.model.__class__.__name__})"

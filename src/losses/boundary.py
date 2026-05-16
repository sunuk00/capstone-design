"""
Boundary Loss

GT 마스크의 경계(dilation - erosion ring)에 가중치를 집중시켜
경계 픽셀 예측 정밀도를 높이는 손실 함수.

── 동작 원리 ──────────────────────────────────────────────────────────────────
    1. GT 마스크를 dilation → 팽창된 마스크
    2. GT 마스크를 erosion  → 침식된 마스크
    3. boundary = dilated - eroded   → 경계 밴드(ring) 추출
    4. weight_map = 1 + (boundary_ratio - 1) × boundary
       경계 픽셀엔 boundary_ratio 배 가중치, 내부 픽셀엔 1배 유지
    5. 픽셀별 BCE × weight_map 의 평균을 손실로 사용

── 클래스 ─────────────────────────────────────────────────────────────────────
    BoundaryLoss          — 경계 가중 BCE 단독 손실
    BCEDiceBoundaryLoss   — BCE + Dice + Boundary 3-component 조합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_boundary(mask: torch.Tensor, dilation_radius: int = 3) -> torch.Tensor:
    """
    GT 마스크에서 경계 밴드를 추출한다.

    max_pool2d 로 dilation을, -max_pool2d(-·) 로 erosion을 근사하여
    두 연산의 차를 경계 밴드로 반환한다.

    Args:
        mask            : (B, 1, H, W) 이진 마스크, 값 범위 [0, 1]
        dilation_radius : 경계 밴드 두께 (픽셀 단위)

    Returns:
        (B, 1, H, W) 경계 밴드, 값 범위 [0, 1]
    """
    k   = 2 * dilation_radius + 1
    pad = dilation_radius
    dilated = F.max_pool2d(mask,  kernel_size=k, stride=1, padding=pad)
    eroded  = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    """
    경계 픽셀 집중 BCE 손실 함수.

    GT 경계 밴드를 추출하고 해당 영역의 BCE 손실에 boundary_ratio 배 가중치를 부여.
    내부 픽셀 손실은 그대로 유지하여 전체 정확도를 보존한다.

    Args:
        boundary_ratio  : 경계 픽셀에 부여할 가중치 배율.
                          5.0 → 경계 픽셀 손실이 내부 픽셀의 5배.
        dilation_radius : 경계 밴드 두께(픽셀). 클수록 더 넓은 경계 영역.
        pos_weight      : 전경(경로) 픽셀 가중치 (희소 foreground 보정).
    """

    def __init__(
        self,
        boundary_ratio: float = 5.0,
        dilation_radius: int  = 3,
        pos_weight: float     = 10.0,
    ) -> None:
        super().__init__()
        self.boundary_ratio  = boundary_ratio
        self.dilation_radius = dilation_radius
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W) raw logits
            target : (B, 1, H, W) 이진 마스크, 값 0 또는 1

        Returns:
            경계 가중 BCE 스칼라
        """
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)

        # 픽셀별 BCE (reduction='none' → 가중치 맵과 element-wise 곱 위해)
        bce_map = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=pos_weight,
            reduction="none",
        )

        # 경계 밴드 가중치 맵: 경계=boundary_ratio, 내부=1.0
        boundary   = _extract_boundary(target.float(), self.dilation_radius)
        weight_map = 1.0 + (self.boundary_ratio - 1.0) * boundary

        return (bce_map * weight_map).mean()


class BCEDiceBoundaryLoss(nn.Module):
    """
    BCE + Soft Dice + Boundary 3-component 조합 손실.

        Total = a·BCE + b·Dice + c·Boundary

    - BCE      : 전체 픽셀 분류 정확도
    - Dice     : 전체 영역 겹침(F1)
    - Boundary : 경계 픽셀 예측 정밀도  ← 세밀함 개선의 핵심

    Args:
        bce_weight      : BCE 비중 a (기본 0.4)
        dice_weight     : Dice 비중 b (기본 0.4)
        boundary_weight : Boundary 비중 c (기본 0.2)
        pos_weight      : 전경 픽셀 가중치
        boundary_ratio  : 경계 픽셀 내부 가중치 배율 (BoundaryLoss 전달)
        dilation_radius : 경계 밴드 두께 (BoundaryLoss 전달)
    """

    def __init__(
        self,
        bce_weight:      float = 0.4,
        dice_weight:     float = 0.4,
        boundary_weight: float = 0.2,
        pos_weight:      float = 10.0,
        boundary_ratio:  float = 5.0,
        dilation_radius: int   = 3,
    ) -> None:
        super().__init__()
        self.bce_weight      = bce_weight
        self.dice_weight     = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_loss   = BoundaryLoss(
            boundary_ratio=boundary_ratio,
            dilation_radius=dilation_radius,
            pos_weight=pos_weight,
        )
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W) raw logits
            target : (B, 1, H, W) 이진 마스크

        Returns:
            3-component 가중 합 손실 스칼라
        """
        from .dice import soft_dice_loss_from_logits

        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce  = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = soft_dice_loss_from_logits(logits, target)
        bnd  = self.boundary_loss(logits, target)

        return self.bce_weight * bce + self.dice_weight * dice + self.boundary_weight * bnd

"""
Skeleton Recall Loss

GT 마스크의 골격선(centerline)을 기준으로 예측이 그 선을 얼마나 커버하는지 측정.
경로처럼 가느다란 구조의 연결성(topology)을 보존하는 데 유효.

Reference: clDice - a Novel Topology-Preserving Loss Function
           for Tubular Structure Segmentation (Shit et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Soft morphological operations (differentiable)
# ──────────────────────────────────────────────

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """분리 가능한 min-pooling으로 형태학적 침식(erosion)을 근사."""
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """max-pooling으로 형태학적 팽창(dilation)을 근사."""
    return F.max_pool2d(img, kernel_size=(3, 3), stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """침식 후 팽창 = 형태학적 열림(opening)."""
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    마스크의 미분 가능한 soft skeleton(골격선)을 계산합니다.

    반복적으로 마스크를 침식하면서 각 단계에서 경계 픽셀을 추출,
    누적하여 골격선을 구성합니다.

    Args:
        img:       확률 맵 또는 이진 마스크 (B, 1, H, W), 값 범위 [0, 1]
        num_iters: 침식 반복 횟수. 클수록 더 얇은 골격선.

    Returns:
        img와 동일한 shape의 soft skeleton 텐서
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ──────────────────────────────────────────────
# Loss classes
# ──────────────────────────────────────────────

class SkeletonRecallLoss(nn.Module):
    """
    Skeleton Recall Loss

    GT 골격선 픽셀 중 예측이 커버한 비율을 최대화합니다.

        skeleton_recall = sum(pred * skel_gt) / (sum(skel_gt) + eps)
        loss            = 1 - skeleton_recall

    경로 골격선을 놓치는 것(False Negative on skeleton)에 강하게 페널티를 부여하므로
    경로의 연결이 끊기는 현상을 억제합니다.
    """

    def __init__(self, num_iters: int = 5, eps: float = 1e-6) -> None:
        """
        Args:
            num_iters: soft skeleton 반복 횟수. 클수록 더 얇은 골격선.
            eps:       분모 0 방지 상수.
        """
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 raw logits (B, 1, H, W)
            target: 정답 마스크 (B, 1, H, W), 값은 0 또는 1

        Returns:
            Skeleton Recall Loss 스칼라
        """
        probs = torch.sigmoid(logits)
        skel_gt = soft_skeleton(target.float(), num_iters=self.num_iters)

        intersection = (probs * skel_gt).sum(dim=(1, 2, 3))
        denom = skel_gt.sum(dim=(1, 2, 3))
        recall = (intersection + self.eps) / (denom + self.eps)
        return 1.0 - recall.mean()


class SkeletonRecallDiceLoss(nn.Module):
    """
    Skeleton Recall Loss + Soft Dice Loss 조합

        Total Loss = a * Skeleton Recall Loss + (1 - a) * Dice Loss

    - Skeleton Recall: 골격선 연결성 보존
    - Dice: 전체 영역 겹침(F1) 보존
    """

    def __init__(
        self,
        skel_weight: float = 0.5,
        num_iters: int = 5,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            skel_weight: Skeleton Recall Loss 비중 a (0~1). Dice 비중은 (1-a).
            num_iters:   soft skeleton 반복 횟수.
            eps:         분모 0 방지 상수.
        """
        super().__init__()
        self.skel_weight = skel_weight
        self.skel_loss = SkeletonRecallLoss(num_iters=num_iters, eps=eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from .dice import soft_dice_loss_from_logits
        skel_loss = self.skel_loss(logits, target)
        dice_loss = soft_dice_loss_from_logits(logits, target)
        return self.skel_weight * skel_loss + (1.0 - self.skel_weight) * dice_loss


class BCEDiceSkelRecallLoss(nn.Module):
    """
    BCE + Soft Dice + Skeleton Recall 3-component 조합 손실 함수

        Total Loss = alpha * BCE + beta * Dice + gamma * SkelRecall

    - skimage로 사전 계산된 GT skeleton을 입력으로 받아 정확도와 속도를 개선.
    - Dataset이 (image, mask, skel) 3-tuple을 반환해야 하며,
      run_epoch은 needs_skeleton 플래그를 보고 자동으로 처리함.

    Reference: 예시 코드의 BCEDiceSkelRecallLoss
    """

    needs_skeleton: bool = True  # Dataset/run_epoch에 skeleton이 필요함을 알려주는 플래그

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        pos_weight: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            alpha:      BCE 비중
            beta:       Dice 비중
            gamma:      Skeleton Recall 비중
            pos_weight: BCE의 양성(경로) 픽셀 가중치 (희소 foreground 보정)
            eps:        분모 0 방지 상수
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        skel_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  모델 출력 raw logits (B, 1, H, W)
            target:  정답 마스크 (B, 1, H, W), 값 0 또는 1
            skel_gt: 사전 계산된 GT skeleton 맵 (B, 1, H, W), 값 0 또는 1
        """
        from .dice import soft_dice_loss_from_logits

        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        dice_loss = soft_dice_loss_from_logits(logits, target, eps=self.eps)

        probs = torch.sigmoid(logits)
        skel_gt = skel_gt.to(device=logits.device, dtype=logits.dtype)
        numerator = (probs * skel_gt).sum(dim=(1, 2, 3))
        denominator = skel_gt.sum(dim=(1, 2, 3)) + self.eps
        skel_recall = (numerator + self.eps) / denominator
        skel_loss = 1.0 - skel_recall.mean()

        return self.alpha * bce_loss + self.beta * dice_loss + self.gamma * skel_loss

"""
Focal Loss 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation

    희소한 경로 픽셀처럼 클래스 불균형이 심할 때, 쉽게 분류되는 배경 픽셀의
    기여를 줄이고 어려운 경로 픽셀에 학습을 집중시킨다.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        """
        Args:
            alpha: 양성(경로) 픽셀에 부여할 가중치 (0~1). 경로가 희소할수록 높게 설정.
            gamma: focusing parameter. 클수록 쉬운 샘플의 가중치를 더 강하게 억제함.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델이 출력한 raw logits (batch_size, 1, H, W)
            target: 정답 마스크 (batch_size, 1, H, W)

        Returns:
            Focal Loss 값
        """
        # pixel-wise BCE (reduction=none으로 각 픽셀 loss 유지)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # p_t: 정답 클래스에 대한 예측 확률
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)

        # alpha_t: 정답 클래스에 따른 alpha 가중치
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)

        # focal weight: 잘 맞힌 픽셀일수록 가중치가 작아짐
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        return (focal_weight * bce).mean()

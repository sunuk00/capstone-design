"""
U-Net++ (UNet2+) 모델 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import ConvBlock


def _resize(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape[2:] != target.shape[2:]:
        x = F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)
    return x


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (UNet2+): 중첩 스킵 연결(nested skip connections)로 인코더-디코더 간
    의미적 격차를 줄이는 모델.

    표기: X^{i,j} = i번째 깊이(해상도), j번째 밀도 노드
    - j=0: 인코더 노드
    - j>0: 디코더 노드 (같은 해상도의 모든 이전 노드를 누적해서 입력받음)

    입력 채널 계산 (X^{i,j}):
        in_ch = ch[i+1]  (하위 레벨 X^{i+1, j-1}에서 업샘플)
              + ch[i] * j (같은 레벨 X^{i, 0..j-1} 누적)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        ch = [base_channels * (2 ** i) for i in range(5)]  # [F, 2F, 4F, 8F, 16F]

        self.pool = nn.MaxPool2d(2)

        # Encoder: X^{i,0}
        self.x_0_0 = ConvBlock(in_channels, ch[0])
        self.x_1_0 = ConvBlock(ch[0],       ch[1])
        self.x_2_0 = ConvBlock(ch[1],       ch[2])
        self.x_3_0 = ConvBlock(ch[2],       ch[3])
        self.x_4_0 = ConvBlock(ch[3],       ch[4])  # bridge

        # Dense decoder nodes X^{i,j}
        self.x_3_1 = ConvBlock(ch[4] + ch[3],     ch[3])

        self.x_2_1 = ConvBlock(ch[3] + ch[2],     ch[2])
        self.x_2_2 = ConvBlock(ch[3] + ch[2] * 2, ch[2])

        self.x_1_1 = ConvBlock(ch[2] + ch[1],     ch[1])
        self.x_1_2 = ConvBlock(ch[2] + ch[1] * 2, ch[1])
        self.x_1_3 = ConvBlock(ch[2] + ch[1] * 3, ch[1])

        self.x_0_1 = ConvBlock(ch[1] + ch[0],     ch[0])
        self.x_0_2 = ConvBlock(ch[1] + ch[0] * 2, ch[0])
        self.x_0_3 = ConvBlock(ch[1] + ch[0] * 3, ch[0])
        self.x_0_4 = ConvBlock(ch[1] + ch[0] * 4, ch[0])

        self.head = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x00 = self.x_0_0(x)
        x10 = self.x_1_0(self.pool(x00))
        x20 = self.x_2_0(self.pool(x10))
        x30 = self.x_3_0(self.pool(x20))
        x40 = self.x_4_0(self.pool(x30))

        # Level 3
        x31 = self.x_3_1(torch.cat([_resize(x40, x30), x30], dim=1))

        # Level 2
        x21 = self.x_2_1(torch.cat([_resize(x30, x20), x20], dim=1))
        x22 = self.x_2_2(torch.cat([_resize(x31, x20), x20, x21], dim=1))

        # Level 1
        x11 = self.x_1_1(torch.cat([_resize(x20, x10), x10], dim=1))
        x12 = self.x_1_2(torch.cat([_resize(x21, x10), x10, x11], dim=1))
        x13 = self.x_1_3(torch.cat([_resize(x22, x10), x10, x11, x12], dim=1))

        # Level 0
        x01 = self.x_0_1(torch.cat([_resize(x10, x00), x00], dim=1))
        x02 = self.x_0_2(torch.cat([_resize(x11, x00), x00, x01], dim=1))
        x03 = self.x_0_3(torch.cat([_resize(x12, x00), x00, x01, x02], dim=1))
        x04 = self.x_0_4(torch.cat([_resize(x13, x00), x00, x01, x02, x03], dim=1))

        return self.head(x04)

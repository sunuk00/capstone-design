"""
U-Net3+ (UNet3+) 모델 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import ConvBlock


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _to(x: torch.Tensor, size) -> torch.Tensor:
    if tuple(x.shape[2:]) != tuple(size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x


class UNet3Plus(nn.Module):
    """
    U-Net3+: 전 스케일 스킵 연결(full-scale skip connections).
    각 디코더 노드가 모든 인코더 스케일과 이전 디코더 노드를 동시에 참조한다.

    디코더 노드당 항상 5개의 입력을 받아 up_channels = 5 × base_channels으로 집계.
    해상도가 세밀해질수록 거친 인코더 연결이 디코더 연결로 대체된다:

        D4 (H/8) : E1↓  E2↓  E3↓  E4     E5↑
        D3 (H/4) : E1↓  E2↓  E3   E4↑    D4↑
        D2 (H/2) : E1↓  E2   E3↑  D3↑    D4↑
        D1 (H)   : E1   E2↑  D2↑  D3↑    D4↑
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        ec = [base_channels * (2 ** i) for i in range(5)]  # [F, 2F, 4F, 8F, 16F]
        cat = base_channels   # 입력별 projection 채널수
        up  = 5 * cat         # 디코더 출력 채널수 = 5F

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, ec[0])  # H,     F
        self.enc2 = ConvBlock(ec[0],       ec[1])  # H/2,  2F
        self.enc3 = ConvBlock(ec[1],       ec[2])  # H/4,  4F
        self.enc4 = ConvBlock(ec[2],       ec[3])  # H/8,  8F
        self.enc5 = ConvBlock(ec[3],       ec[4])  # H/16, 16F (bridge)

        # D4 (H/8): E1↓, E2↓, E3↓, E4(same), E5↑
        self.d4_e1 = _proj(ec[0], cat)
        self.d4_e2 = _proj(ec[1], cat)
        self.d4_e3 = _proj(ec[2], cat)
        self.d4_e4 = _proj(ec[3], cat)
        self.d4_e5 = _proj(ec[4], cat)
        self.dec4  = ConvBlock(up, up)

        # D3 (H/4): E1↓, E2↓, E3(same), E4↑, D4↑
        self.d3_e1 = _proj(ec[0], cat)
        self.d3_e2 = _proj(ec[1], cat)
        self.d3_e3 = _proj(ec[2], cat)
        self.d3_e4 = _proj(ec[3], cat)
        self.d3_d4 = _proj(up,    cat)
        self.dec3  = ConvBlock(up, up)

        # D2 (H/2): E1↓, E2(same), E3↑, D3↑, D4↑
        self.d2_e1 = _proj(ec[0], cat)
        self.d2_e2 = _proj(ec[1], cat)
        self.d2_e3 = _proj(ec[2], cat)
        self.d2_d3 = _proj(up,    cat)
        self.d2_d4 = _proj(up,    cat)
        self.dec2  = ConvBlock(up, up)

        # D1 (H): E1(same), E2↑, D2↑, D3↑, D4↑
        self.d1_e1 = _proj(ec[0], cat)
        self.d1_e2 = _proj(ec[1], cat)
        self.d1_d2 = _proj(up,    cat)
        self.d1_d3 = _proj(up,    cat)
        self.d1_d4 = _proj(up,    cat)
        self.dec1  = ConvBlock(up, up)

        self.head = nn.Conv2d(up, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        s4 = e4.shape[2:]
        s3 = e3.shape[2:]
        s2 = e2.shape[2:]
        s1 = e1.shape[2:]

        # D4 (H/8)
        d4 = self.dec4(torch.cat([
            _to(self.d4_e1(e1), s4),
            _to(self.d4_e2(e2), s4),
            _to(self.d4_e3(e3), s4),
                self.d4_e4(e4),
            _to(self.d4_e5(e5), s4),
        ], dim=1))

        # D3 (H/4)
        d3 = self.dec3(torch.cat([
            _to(self.d3_e1(e1), s3),
            _to(self.d3_e2(e2), s3),
                self.d3_e3(e3),
            _to(self.d3_e4(e4), s3),
            _to(self.d3_d4(d4), s3),
        ], dim=1))

        # D2 (H/2)
        d2 = self.dec2(torch.cat([
            _to(self.d2_e1(e1), s2),
                self.d2_e2(e2),
            _to(self.d2_e3(e3), s2),
            _to(self.d2_d3(d3), s2),
            _to(self.d2_d4(d4), s2),
        ], dim=1))

        # D1 (H)
        d1 = self.dec1(torch.cat([
                self.d1_e1(e1),
            _to(self.d1_e2(e2), s1),
            _to(self.d1_d2(d2), s1),
            _to(self.d1_d3(d3), s1),
            _to(self.d1_d4(d4), s1),
        ], dim=1))

        out = self.head(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

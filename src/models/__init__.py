from torch import nn

from .deeplabv3 import DeepLabV3Model
from .resunet import ResUNet
from ._blocks import ConvBlock
from .unet import UNet
from .unet_plus_plus import UNetPlusPlus
from .unet3plus import UNet3Plus
from .segformer import SegFormerSeg
from .segformer_unet import SegFormerUNet


def get_model(model_name: str, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> nn.Module:
    name = model_name.lower()

    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    if name == "deeplabv3":
        return DeepLabV3Model(in_channels=in_channels, out_channels=out_channels)

    if name == "resunet":
        return ResUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    if name == "unet++":
        return UNetPlusPlus(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    if name == "unet3+":
        return UNet3Plus(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    # segformer_unet 을 먼저 체크해야 한다 — "segformer_unet"도 "segformer"로 시작하기 때문
    if name.startswith("segformer_unet"):
        # "segformer_unet-b0", "segformer_unet-b2", "segformer_unet-b4" 형태를 파싱
        # "-" 구분자가 없으면 기본값 b2 사용
        variant = name.split("-")[1] if "-" in name else "b2"
        return SegFormerUNet(out_channels=out_channels, variant=variant)

    if name.startswith("segformer"):
        # "segformer-b0", "segformer-b2", "segformer-b4" 형태를 파싱
        # "-" 구분자가 없으면 기본값 b0 사용
        variant = name.split("-")[1] if "-" in name else "b0"
        return SegFormerSeg(out_channels=out_channels, variant=variant)

    raise ValueError(f"Unknown model name: {model_name}")


__all__ = [
    "ConvBlock", "UNet", "ResUNet", "DeepLabV3Model",
    "UNetPlusPlus", "UNet3Plus", "SegFormerSeg", "SegFormerUNet",
    "get_model",
]

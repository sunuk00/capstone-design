from torch import nn

from .deeplabv3 import DeepLabV3Model
from .resunet import ResUNet
from ._blocks import ConvBlock
from .unet import UNet
from .unet_plus_plus import UNetPlusPlus
from .unet3plus import UNet3Plus


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

    raise ValueError(f"Unknown model name: {model_name}")


__all__ = ["ConvBlock", "UNet", "ResUNet", "DeepLabV3Model", "UNetPlusPlus", "UNet3Plus", "get_model"]

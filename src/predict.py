"""
통합 예측 (추론) 스크립트
학습된 모델을 사용하여 마라톤 경로 예측
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Avoid conflicts with third-party packages named "src" when running as a script.
    sys.modules.pop("src", None)

from src.core import VALID_EXTENSIONS, parse_args_with_config
from src.data import apply_model_preprocess
from src.models import get_model
from src.models.segformer_unet import SegFormerUNet


def _remap_v1_keys(state_dict: dict) -> dict:
    """
    SegFormerUNet v1 체크포인트의 키를 v2 키 이름으로 변환한다.

    v1: ConvBlock.block.{0,1,3,4}.* 구조 (bias=True Conv2d + BN)
    v2: ConvBlock[0][0/1].* 구조 (ConvBNReLU×2, bias=False)
    """
    new_sd = {}
    for k, v in state_dict.items():
        k2 = k
        # upN.conv.block.0.weight → upN.conv.0.0.weight  (1번째 Conv2d)
        k2 = re.sub(r"(up\d\.conv)\.block\.0\.weight$", r"\1.0.0.weight", k2)
        # upN.conv.block.1.* → upN.conv.0.1.*  (1번째 BN)
        k2 = re.sub(r"(up\d\.conv)\.block\.1\.", r"\1.0.1.", k2)
        # upN.conv.block.3.weight → upN.conv.1.0.weight  (2번째 Conv2d)
        k2 = re.sub(r"(up\d\.conv)\.block\.3\.weight$", r"\1.1.0.weight", k2)
        # upN.conv.block.4.* → upN.conv.1.1.*  (2번째 BN)
        k2 = re.sub(r"(up\d\.conv)\.block\.4\.", r"\1.1.1.", k2)
        # upN.skip_proj.weight → upN.skip_proj.0.weight  (v2는 Sequential)
        k2 = re.sub(r"(up\d\.skip_proj)\.weight$", r"\1.0.weight", k2)
        # v1 Conv2d bias는 v2가 bias=False + BN 이므로 제거
        if re.search(r"conv\.block\.[03]\.bias$", k):
            continue
        new_sd[k2] = v
    return new_sd


def _init_as_identity(module: nn.Sequential) -> None:
    """
    ConvBNReLU(= Sequential[Conv2d, BN, ReLU]) 를 항등 변환으로 초기화한다.

    v1에 없던 pre_head가 v2에 추가됐을 때, 로드 후 feature를 왜곡하지 않도록
    center-pixel identity conv + BN identity 로 설정한다.
    """
    conv: nn.Conv2d         = module[0]
    bn:   nn.BatchNorm2d    = module[1]
    out_ch, in_ch = conv.weight.shape[:2]
    with torch.no_grad():
        conv.weight.zero_()
        for i in range(min(out_ch, in_ch)):
            conv.weight[i, i, 1, 1] = 1.0  # 중심 픽셀만 1 → 채널별 항등 근사
        bn.weight.fill_(1.0)
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)




def build_arg_parser() -> argparse.ArgumentParser:
    """
    커맨드 라인 인자 설정
    """
    parser = argparse.ArgumentParser(description="U-Net prediction for marathon path segmentation")

    # config 파일 경로. 이 파일의 값이 기본값으로 먼저 적용되고,
    # 동일 옵션을 CLI로 다시 주면 CLI 값이 최종적으로 우선 적용됨.
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    parser.add_argument("--model-path", type=str, default="outputs/unet_trained/model_final.pt", help="Path to trained model checkpoint")
    parser.add_argument("--image-dir", type=str, default="data/test/images", help="Directory containing images to predict")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory for predictions")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size")

    # 이진 마스크 생성 시 사용할 확률 threshold : 모델이 출력하는 확률값이 이 threshold보다 크면 해당 픽셀을 1(경로)로, 그렇지 않으면 0(비경로)으로 분류
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")  
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels in U-Net")
    
    return parser



def load_model(model_path: str, device: torch.device, base_channels: int = 32) -> Tuple[torch.nn.Module, dict]:
    """
    저장된 모델을 로드하는 함수.

    v1 SegFormerUNet 체크포인트(키 이름이 다름)도 자동으로 감지해
    v2 모델로 호환 로드한다.

    Args:
        model_path: 모델 체크포인트 경로
        device: 모델을 로드할 디바이스
        base_channels: U-Net의 기본 채널 수

    Returns:
        (모델, 설정값) 튜플
    """
    checkpoint = torch.load(model_path, map_location=device)

    ckpt_args   = checkpoint.get("args", {})
    model_name  = ckpt_args.get("model_name", "unet")
    state_dict  = checkpoint["model_state_dict"]

    # ── v1 SegFormerUNet 체크포인트 감지 ─────────────────────────────────────
    # v1은 ConvBlock.block.N 경로를 사용; v2는 ConvBNReLU 중첩 구조
    is_segformer_unet = model_name.lower().startswith("segformer_unet")
    is_v1_ckpt = is_segformer_unet and any("conv.block." in k for k in state_dict)

    if is_v1_ckpt:
        print("[경고] v1 SegFormerUNet 체크포인트 감지 → v2 호환 모드로 로드합니다.")
        variant = model_name.split("-")[-1] if "-" in model_name else "b0"
        model = SegFormerUNet(
            out_channels=1,
            variant=variant,
            use_aspp=False,         # v1에 없던 ASPP 비활성화
            deep_supervision=False, # v1에 없던 보조 헤드 비활성화
            v1_compat=True,         # v1은 skip_proj가 skip_ch→skip_ch 크기로 projection
        )
        remapped = _remap_v1_keys(state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # pre_head는 v2 신규 레이어 → feature 왜곡 방지를 위해 항등 변환으로 초기화
        _init_as_identity(model.pre_head)
        if missing:
            print(f"  미로드 키 {len(missing)}개 (v2 전용 레이어, 기본값 사용)")
        if unexpected:
            print(f"  불필요 키 {len(unexpected)}개 (무시)")
    else:
        model = get_model(
            model_name=model_name,
            in_channels=3,
            out_channels=1,
            base_channels=base_channels,
        )
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, ckpt_args


def preprocess_image(image_path: Path, image_size: int, model_name: str) -> torch.Tensor:
    """
    이미지를 전처리하는 함수
    
    Args:
        image_path: 이미지 파일 경로
        image_size: 목표 이미지 크기
    
    Returns:
        전처리된 이미지 텐서 (1, 3, H, W)
    """
    # Pilow(PIL) 라이브러리로 Image를 열고, 이미지를 무조건 RGB 컬러 이미지로 통일
    # 그리고 이미지를 지정된 크기로 조절, BILINEAR(쌍선형 보간법)을 사용하여 픽셀 사이를 부드럽게 채움
    image = Image.open(image_path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    
    # PIL 이미지를 numpy 배열로 변환
    # 255.0으로 나눠줌으로써 0 ~ 255 사이 픽셀값을 0.0 ~ 1.0 사이로 정규화
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    
    # numpy 배열을 Tensor로 변환하고, (H, W, C) -> (C, H, W) 순서로 변경
    image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
    image_tensor = apply_model_preprocess(image_tensor, model_name)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict_single(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    단일 이미지에 대해 예측을 수행하는 함수.

    Returns:
        (H, W) uint8 배열, 값은 0 또는 255
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        if isinstance(logits, dict):
            logits = logits["main"]
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float().squeeze().cpu().numpy()
    return (mask * 255).astype(np.uint8)


def main() -> None:
    args = parse_args_with_config(build_arg_parser)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, device, base_channels=args.base_channels)
    model_name = model_args.get("model_name", "unet")
    print(f"Model: {model_name}")

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 디렉토리에서 이미지 파일들을 찾기
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    
    if not image_paths:
        raise RuntimeError(f"No valid images found in {image_dir}")

    print(f"Found {len(image_paths)} images to predict")
    print("-" * 80)

    # 예측 수행
    predictions = []
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"Predicting {idx}/{len(image_paths)}: {image_path.name}")
        with Image.open(image_path) as original_image:
            original_size = original_image.size
        
        # 이미지 전처리
        image_tensor = preprocess_image(image_path, args.image_size, model_name=model_name)
        
        # 예측
        mask = predict_single(model, image_tensor, device, threshold=args.threshold)

        # 마스크를 원본 이미지 해상도로 복원한 뒤 저장
        output_name = image_path.stem + "_mask.png"
        output_path = output_dir / output_name

        mask_image = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
        mask_image.save(output_path)

        predictions.append({
            "image": image_path.name,
            "mask": output_name,
            "output_path": str(output_path),
            "original_size": [original_size[0], original_size[1]],
        })

        print(f"  Mask saved to {output_path}")

    print("-" * 80)
    print(f"Prediction completed! Generated {len(predictions)} masks")

    # 예측 결과 기록 저장
    result_log_path = output_dir / "prediction_log.json"
    with open(result_log_path, "w") as f:
        json.dump({
            "total_predictions": len(predictions),
            "model_path": str(args.model_path),
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "image_size": args.image_size,
            "threshold": args.threshold,
            "predictions": predictions,
        }, f, indent=4)
    print(f"Result log saved to {result_log_path}")


if __name__ == "__main__":
    main()

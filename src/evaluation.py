"""
evaluation.py — UNet / SegFormer / SegFormerUNet 성능 평가 스크립트

실행 예시:
    python src/evaluation.py \
        --unet_weights      weights/unet_best.pth \
        --segformer_weights weights/segformer_best.pth \
        --sfunet_weights    weights/segformer_unet_b2_best.pth
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: PyTorch+OpenCV OpenMP 충돌 방지

import argparse
import csv
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("[ERROR] albumentations 패키지 필요: pip install albumentations")
    sys.exit(1)

try:
    from skimage.morphology import skeletonize as sk_skeletonize
except ImportError:
    print("[ERROR] scikit-image 패키지 필요: pip install scikit-image")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_sys = platform.system()
if _sys == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif _sys == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# src/ 디렉터리를 경로에 추가하여 models 패키지를 import 가능하게 함
sys.path.insert(0, str(Path(__file__).parent))
from models.unet import UNet
from models.segformer import SegFormerSeg
from models.segformer_unet import SegFormerUNet


# ══════════════════════════════════════════════════════════════════════════════
# 설정 — 여기서 모든 파라미터를 수정하세요
# ══════════════════════════════════════════════════════════════════════════════

# 데이터 경로
IMAGE_DIR  = r"data\test\images"
MASK_DIR   = r"data\test\masks"
RESULT_DIR = "eval_results"
VIZ_DIR    = "eval_results/viz"

# 가중치 경로 (사용하지 않을 모델은 None으로 설정)
UNET_WEIGHTS      = r"outputs\unet\exp021__focal_dice_fw0.4_fa0.8_fg2.0__image768__aug_gray\model_best.pt"
SEGFORMER_WEIGHTS = r"outputs\segformer-b2\exp001__bce0.5_dice0.5__pos15__data_augmentation__data_syn\model_best.pt"
SFUNET_WEIGHTS    = r"outputs\segformer-unet-b2\exp008__v3_focal0.5_dice0.5__aug_gray__image768__batch4\model_best.pt"

# 모델별 입력 해상도
UNET_IMAGE_SIZE      = 768
SEGFORMER_IMAGE_SIZE = 512
SFUNET_IMAGE_SIZE    = 768

# 기타
TOLERANCE = 10
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════


# ── 모델 로딩 함수 ────────────────────────────────────────────────────────────

def _load_ckpt(weights_path: str) -> dict:
    return torch.load(weights_path, map_location=DEVICE, weights_only=False)


def load_unet(weights_path: str) -> Optional[nn.Module]:
    try:
        ckpt  = _load_ckpt(weights_path)
        state = ckpt.get("model_state_dict", ckpt)
        # head.weight shape: (out_ch, base_ch, 1, 1) → base_channels 자동 감지
        base_channels = int(state["head.weight"].shape[1]) if "head.weight" in state else 32
        model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        print(f"[UNet] 가중치 로드 완료: {weights_path}")
        return model
    except Exception as e:
        print(f"[UNet] 가중치 로드 실패: {e}")
        return None


def load_segformer(weights_path: str) -> Optional[nn.Module]:
    try:
        ckpt       = _load_ckpt(weights_path)
        state      = ckpt.get("model_state_dict", ckpt)
        variant    = "b2"
        model_name = ckpt.get("args", {}).get("model_name", "")
        for v in ("b0", "b2", "b4"):
            if v in model_name:
                variant = v
                break
        model = SegFormerSeg(out_channels=1, variant=variant)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        print(f"[SegFormer] 가중치 로드 완료: {weights_path}")
        return model
    except Exception as e:
        print(f"[SegFormer] 가중치 로드 실패: {e}")
        return None


def load_sfunet(weights_path: str) -> Optional[nn.Module]:
    try:
        ckpt  = _load_ckpt(weights_path)
        state = ckpt.get("model_state_dict", ckpt)
        saved = ckpt.get("args", {})
        use_fusion_neck = saved.get("use_fusion_neck", False)
        has_aux = any("aux_head" in k for k in state.keys())
        model = SegFormerUNet(
            out_channels=1, variant="b2", pretrained=False,
            use_fusion_neck=use_fusion_neck,
            use_aspp=not use_fusion_neck,
            deep_supervision=has_aux,
        )
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        print(f"[SegFormerUNet-b2] 가중치 로드 완료: {weights_path}")
        return model
    except Exception as e:
        print(f"[SegFormerUNet-b2] 가중치 로드 실패: {e}")
        return None


# ── 주경로 추출 ───────────────────────────────────────────────────────────────

def extract_main_path(logit: torch.Tensor) -> np.ndarray:
    """logit (B,1,H,W) → (H,W) uint8 binary mask (최대 연결 성분만)"""
    prob   = torch.sigmoid(logit[0, 0]).cpu().numpy()
    binary = (prob > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def compute_metrics(logit: torch.Tensor, gt_mask: np.ndarray) -> dict:
    """
    logit   : (B,1,H,W) raw logit tensor
    gt_mask : (H,W) float32 binary [0,1]
    반환    : path_precision, path_recall, path_f1, dice, iou, cldice

    평가 방식 설명:
        Path Precision/Recall/F1:
            예측 주경로와 GT 마스크를 각각 skeleton(1픽셀 선)으로 변환 후 비교.
            두께가 다른 예측이라도 skeleton은 동일한 1픽셀 선이 되므로
            뚱뚱하게 예측할수록 유리해지는 편향이 제거됨.
            GT skeleton은 TOLERANCE(10px)만큼 팽창해 허용 오차 적용.

        Dice / IoU / clDice:
            전체 예측 마스크(주경로 추출 전) 기준 참고 지표.
    """
    prob      = torch.sigmoid(logit[0, 0]).cpu().numpy()
    pred_full = (prob > 0.5).astype(np.uint8)
    gt        = gt_mask.astype(np.uint8)

    if pred_full.sum() == 0 or gt.sum() == 0:
        return dict(path_precision=0.0, path_recall=0.0, path_f1=0.0,
                    dice=0.0, iou=0.0, cldice=0.0)

    # 주경로 기반 지표 (skeleton 기반 — 두께 영향 제거)
    pred_main  = extract_main_path(logit)

    # 예측 주경로와 GT 각각 skeleton화 (1픽셀 선으로 통일)
    pred_skel = sk_skeletonize(pred_main.astype(bool)).astype(np.uint8)
    gt_skel   = sk_skeletonize(gt.astype(bool)).astype(np.uint8)

    # GT skeleton을 TOLERANCE만큼 팽창 (허용 오차 적용)
    kernel          = cv2.getStructuringElement(
                          cv2.MORPH_ELLIPSE, (TOLERANCE * 2 + 1, TOLERANCE * 2 + 1))
    gt_skel_dilated = cv2.dilate(gt_skel, kernel)

    # pred_skel이 0인 경우 대비
    if pred_skel.sum() == 0 or gt_skel.sum() == 0:
        path_p = path_r = path_f1 = 0.0
    else:
        inter   = int((pred_skel & gt_skel_dilated).sum())
        path_p  = inter / (int(pred_skel.sum())          + 1e-6)
        path_r  = inter / (int(gt_skel_dilated.sum())    + 1e-6)
        path_f1 = 2 * path_p * path_r / (path_p + path_r + 1e-6)

    # 전체 마스크 기반 지표
    dice = 2 * int((pred_full & gt).sum()) / (int(pred_full.sum()) + int(gt.sum()) + 1e-6)
    iou  = int((pred_full & gt).sum()) / (int((pred_full | gt).sum()) + 1e-6)

    # clDice
    skel_pred = sk_skeletonize(pred_full.astype(bool)).astype(np.uint8)
    skel_gt   = sk_skeletonize(gt.astype(bool)).astype(np.uint8)
    tprec  = float((skel_pred & gt).sum())       / (float(skel_pred.sum()) + 1e-6)
    tsens  = float((skel_gt & pred_full).sum())  / (float(skel_gt.sum())   + 1e-6)
    cldice = 2 * tprec * tsens / (tprec + tsens + 1e-6)

    return dict(
        path_precision=float(path_p),
        path_recall=float(path_r),
        path_f1=float(path_f1),
        dice=float(dice),
        iou=float(iou),
        cldice=float(cldice),
    )


# ── 시각화 저장 ───────────────────────────────────────────────────────────────

def save_visualization(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_main: np.ndarray,
    model_name: str,
    image_name: str,
) -> None:
    out_dir = Path(VIZ_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(image_name).stem}.png"

    # 오버레이를 위해 image_rgb를 마스크 크기에 맞춤 (원본이 더 클 수 있음)
    h, w = gt_mask.shape[:2]
    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("원본 이미지", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("GT 마스크", fontsize=14)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_main, cmap="gray")
    axes[1, 0].set_title("예측 주경로", fontsize=14)
    axes[1, 0].axis("off")

    # GT: 초록 / 예측: 빨강 / 겹침: 노랑
    base    = image_rgb.astype(np.float32) / 255.0
    overlay = base.copy()

    gt_bool   = gt_mask.astype(bool)
    pred_bool = pred_main.astype(bool)
    overlap   = gt_bool & pred_bool

    overlay[gt_bool]   = np.clip(overlay[gt_bool]   + np.array([0,   0.5, 0  ]), 0, 1)
    overlay[pred_bool] = np.clip(overlay[pred_bool] + np.array([0.5, 0,   0  ]), 0, 1)
    overlay[overlap]   = np.array([0.9, 0.85, 0.0])  # 노랑 (겹침 영역)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("오버레이", fontsize=14)
    axes[1, 1].axis("off")
    axes[1, 1].legend(
        handles=[
            mpatches.Patch(color="green",  label="GT",   alpha=0.6),
            mpatches.Patch(color="red",    label="Pred", alpha=0.6),
            mpatches.Patch(color="yellow", label="겹침", alpha=0.8),
        ],
        loc="lower right", fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── 단일 모델 평가 ────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, model_name: str, image_size: int) -> list:
    image_dir   = Path(IMAGE_DIR)
    mask_dir    = Path(MASK_DIR)
    image_paths = sorted(p for p in image_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    results = []

    _transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    for img_path in tqdm(image_paths, desc=f"[{model_name}]", unit="img"):
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = mask_dir / (img_path.stem + ext)
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            print(f"  경고: 마스크 없음 — {img_path.name} 스킵")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  경고: 이미지 로드 실패 — {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            print(f"  경고: 마스크 로드 실패 — {mask_path.name}")
            continue
        mask_r  = cv2.resize(mask_raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        gt_mask = (mask_r > 127).astype(np.float32)

        aug        = _transform(image=image_rgb)
        img_tensor = aug["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = model(img_tensor)

        if isinstance(logit, dict):
            logit = logit["main"]

        if logit.shape[-2:] != (image_size, image_size):
            logit = torch.nn.functional.interpolate(
                logit,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

        metrics   = compute_metrics(logit, gt_mask)
        pred_main = extract_main_path(logit)
        save_visualization(image_rgb, gt_mask, pred_main, model_name, img_path.name)

        results.append({"model": model_name, "image_name": img_path.name, **metrics})
        tqdm.write(
            f"  {img_path.name:<30} "
            f"F1={metrics['path_f1']:.3f}  "
            f"Dice={metrics['dice']:.3f}  "
            f"IoU={metrics['iou']:.3f}"
        )

    return results


# ── 결과 출력 / CSV 저장 ──────────────────────────────────────────────────────

_METRIC_KEYS = ("path_precision", "path_recall", "path_f1", "dice", "iou", "cldice")


def _model_avgs(all_results: list):
    order   = []
    metrics = defaultdict(lambda: defaultdict(list))
    for row in all_results:
        m = row["model"]
        if m not in order:
            order.append(m)
        for k in _METRIC_KEYS:
            metrics[m][k].append(row[k])
    return order, metrics


def print_summary(all_results: list) -> None:
    order, metrics = _model_avgs(all_results)
    sep = "=" * 71
    hdr = f"{'모델':<18} {'Path_P':>7} {'Path_R':>7} {'Path_F1':>8} {'Dice':>7} {'IoU':>7} {'clDice':>8}"
    print(f"\n{'=' * 30} 평가 결과 {'=' * 31}")
    print(hdr)
    print("-" * 71)
    for m in order:
        mm = metrics[m]
        print(
            f"{m:<18}"
            f" {np.mean(mm['path_precision']):>7.3f}"
            f" {np.mean(mm['path_recall']):>7.3f}"
            f" {np.mean(mm['path_f1']):>8.3f}"
            f" {np.mean(mm['dice']):>7.3f}"
            f" {np.mean(mm['iou']):>7.3f}"
            f" {np.mean(mm['cldice']):>8.3f}"
        )
    print(sep)


def save_csv(all_results: list) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path   = Path(RESULT_DIR) / "eval_results.csv"
    fieldnames = ["model", "image_name"] + list(_METRIC_KEYS)

    order, metrics = _model_avgs(all_results)
    avg_rows = [{
        "model": m, "image_name": "AVERAGE",
        **{k: float(np.mean(metrics[m][k])) for k in _METRIC_KEYS},
    } for m in order]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
        w.writerows(avg_rows)

    print(f"\n{csv_path} 저장 완료")


# ── 논문 표 저장 ──────────────────────────────────────────────────────────────

def save_paper_table(all_results: list) -> None:
    metric_keys   = ["path_precision", "path_recall", "path_f1", "dice", "iou"]
    metric_labels = ["Path P", "Path R", "Path F1", "Dice", "IoU"]
    col_xs        = [0.30, 0.42, 0.54, 0.68, 0.82]
    row_ys        = [0.85, 0.68, 0.48, 0.30, 0.12]

    order, metrics = _model_avgs(all_results)
    model_order    = ["UNet", "SegFormer", "SegFormerUNet-b2"]
    display_order  = [m for m in model_order if m in metrics] + \
                     [m for m in order if m not in model_order]
    data_rows = [
        {"model": m, **{k: float(np.mean(metrics[m][k])) for k in metric_keys}}
        for m in display_order
    ]

    if not data_rows:
        return

    best = {k: max(r[k] for r in data_rows) for k in metric_keys}

    # 폰트 선택: Times New Roman → DejaVu Serif fallback
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    FONT = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"

    fig, ax = plt.subplots(figsize=(8, 2.0))
    ax.axis("off")

    # ── 가로선 ────────────────────────────────────────────────────────────────
    ax.plot([0.0, 1.0], [0.97, 0.97], color="black", linewidth=1.8, transform=ax.transAxes)
    ax.plot([0.0, 1.0], [0.58, 0.58], color="black", linewidth=0.8, transform=ax.transAxes)
    ax.plot([0.0, 1.0], [0.01, 0.01], color="black", linewidth=1.8, transform=ax.transAxes)

    # ── 그룹 헤더행 ──────────────────────────────────────────────────────────
    main_cx = (col_xs[0] + col_xs[2]) / 2   # Path P ~ Path F1 중심
    ref_cx  = (col_xs[3] + col_xs[4]) / 2   # Dice ~ IoU 중심
    ax.text(main_cx, row_ys[0], "Main Metrics",
            ha="center", va="center", fontsize=9,
            fontweight="bold", fontstyle="italic",
            fontfamily=FONT, transform=ax.transAxes)
    ax.text(ref_cx, row_ys[0], "Reference Metrics",
            ha="center", va="center", fontsize=9,
            fontweight="bold", fontstyle="italic",
            fontfamily=FONT, transform=ax.transAxes)

    # ── 세부 헤더행 ──────────────────────────────────────────────────────────
    ax.text(0.02, row_ys[1], "Method",
            ha="left", va="center", fontsize=8.5,
            fontweight="bold", fontstyle="italic",
            fontfamily=FONT, transform=ax.transAxes)
    for x, label in zip(col_xs, metric_labels):
        ax.text(x, row_ys[1], label,
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", fontstyle="italic",
                fontfamily=FONT, transform=ax.transAxes)

    # ── 데이터 행 ────────────────────────────────────────────────────────────
    for i, row in enumerate(data_rows[:len(row_ys) - 2]):
        y = row_ys[2 + i]
        ax.text(0.02, y, row["model"],
                ha="left", va="center", fontsize=8.5,
                fontfamily=FONT, transform=ax.transAxes)
        for x, k in zip(col_xs, metric_keys):
            val = row[k]
            fw  = "bold" if abs(val - best[k]) < 1e-9 else "normal"
            ax.text(x, y, f"{val:.3f}",
                    ha="right", va="center", fontsize=8.5,
                    fontweight=fw, fontfamily=FONT, transform=ax.transAxes)

    plt.savefig(
        str(Path(RESULT_DIR) / "paper_table.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close(fig)
    print("paper_table.png 저장 완료")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Marathon path segmentation — evaluation")
    parser.add_argument("--unet_weights",      type=str, default=UNET_WEIGHTS)
    parser.add_argument("--segformer_weights", type=str, default=SEGFORMER_WEIGHTS)
    parser.add_argument("--sfunet_weights",    type=str, default=SFUNET_WEIGHTS)
    args = parser.parse_args()

    if not any([args.unet_weights, args.segformer_weights, args.sfunet_weights]):
        parser.error("최소 1개 모델의 가중치 경로를 지정하세요. (파일 상단 상수 또는 CLI 인수)")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    print(f"DEVICE: {DEVICE}")

    all_results = []

    if args.unet_weights:
        model = load_unet(args.unet_weights)
        if model:
            all_results.extend(evaluate_model(model, "UNet", image_size=UNET_IMAGE_SIZE))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.segformer_weights:
        model = load_segformer(args.segformer_weights)
        if model:
            all_results.extend(evaluate_model(model, "SegFormer", image_size=SEGFORMER_IMAGE_SIZE))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.sfunet_weights:
        model = load_sfunet(args.sfunet_weights)
        if model:
            all_results.extend(evaluate_model(model, "SegFormerUNet-b2", image_size=SFUNET_IMAGE_SIZE))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_results:
        print_summary(all_results)
        save_csv(all_results)
        save_paper_table(all_results)
    else:
        print("평가 결과 없음. 가중치 경로와 데이터 경로를 확인하세요.")


if __name__ == "__main__":
    main()

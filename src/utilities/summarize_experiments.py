"""
실험 결과 요약 스크립트

outputs/unet/ 아래의 모든 실험 폴더를 읽어
마크다운 표(기본) 또는 CSV로 출력합니다.

사용법:
    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --sort iou
    python scripts/summarize_experiments.py --format csv --output results.csv
    python scripts/summarize_experiments.py --dir outputs/resunet
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAML이 필요합니다: pip install pyyaml")
    sys.exit(1)


# ──────────────────────────────────────────────
# Config 로딩
# ──────────────────────────────────────────────

def _parse_folder_name(name: str) -> dict:
    """YAML이 없는 실험 폴더의 이름에서 설정값을 파싱합니다."""
    cfg: dict = {
        "model_name": "unet",
        "loss_type": None,
        "bce_weight": None,
        "dice_weight": None,
        "iou_weight": None,
        "focal_weight": None,
        "alpha": None,
        "gamma": None,
        "pos_weight": None,
        "use_augmentation": False,
        "image_size": 512,
        "data_root": "data/train",
    }

    parts = name.lower().split("__")
    cfg["use_augmentation"] = any(
        "agumentation" in p or "augmentation" in p for p in parts
    )
    cfg["data_root"] = (
        "data/Dataset" if any("new_data" in p for p in parts) else "data/train"
    )

    for p in parts[1:]:  # parts[0] == exp id
        m = re.search(r"image(\d+)", p)
        if m:
            cfg["image_size"] = int(m.group(1))
            continue

        if re.match(r"focal[\d.]*_dice[\d.]*", p):
            cfg["loss_type"] = "focal_dice"
            m = re.match(r"focal([\d.]+)_dice([\d.]+)", p)
            if m:
                cfg["focal_weight"] = float(m.group(1))
                cfg["dice_weight"] = float(m.group(2))
        elif re.match(r"bce[\d.]*_dice[\d.]*", p):
            cfg["loss_type"] = "bce_dice"
            m = re.match(r"bce([\d.]+)_dice([\d.]+)", p)
            if m:
                cfg["bce_weight"] = float(m.group(1))
                cfg["dice_weight"] = float(m.group(2))
        elif re.match(r"bce[\d.]*_iou[\d.]*", p):
            cfg["loss_type"] = "bce_iou"
            m = re.match(r"bce([\d.]+)_iou([\d.]+)", p)
            if m:
                cfg["bce_weight"] = float(m.group(1))
                cfg["iou_weight"] = float(m.group(2))
        elif p == "focal":
            cfg["loss_type"] = "focal"
        elif p == "bce":
            cfg["loss_type"] = "bce"
        elif p.startswith("pos"):
            m = re.match(r"pos([\d.]+)", p)
            if m:
                cfg["pos_weight"] = float(m.group(1))
        elif p.startswith("alpha"):
            m = re.match(r"alpha([\d.]+)", p)
            if m:
                cfg["alpha"] = float(m.group(1))
        elif p.startswith("gamma"):
            m = re.match(r"gamma([\d.]+)", p)
            if m:
                cfg["gamma"] = float(m.group(1))

    return cfg


def load_config(exp_dir: Path) -> dict:
    """폴더명 파싱 후 YAML로 덮어씌워 설정값을 반환합니다.

    YAML이 있으면 YAML의 non-None 값이 우선됩니다.
    단, 폴더명에서만 신뢰할 수 있는 new_data 플래그는 폴더명 기반을 유지합니다.
    """
    cfg = _parse_folder_name(exp_dir.name)
    folder_has_new_data = "new_data" in exp_dir.name.lower()

    yaml_files = list(exp_dir.glob("*.yaml"))
    if yaml_files:
        try:
            raw = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8")) or {}
            # focal_alpha / alpha 키 통일
            if raw.get("alpha") is None and raw.get("focal_alpha") is not None:
                raw["alpha"] = raw["focal_alpha"]
            # YAML의 non-None 값으로 cfg 덮어씌우기
            for k, v in raw.items():
                if v is not None:
                    cfg[k] = v
        except Exception:
            pass

    # 폴더명의 new_data 플래그가 더 신뢰할 수 있으면 우선 적용
    if folder_has_new_data:
        cfg["data_root"] = "data/Dataset"

    return cfg


# ──────────────────────────────────────────────
# 포매팅 헬퍼
# ──────────────────────────────────────────────

def _fmt(v, fmt=":g") -> str:
    return format(v, fmt.strip(":")) if v is not None else "?"


def fmt_params(cfg: dict) -> str:
    """손실 함수별 핵심 파라미터를 한 줄 문자열로 반환합니다."""
    loss = cfg.get("loss_type", "?")
    parts = []

    if loss == "bce":
        pw = cfg.get("pos_weight")
        if pw is not None and pw > 0:
            parts.append(f"pos={_fmt(pw)}")

    elif loss in ("bce_dice", "bce_iou"):
        bw = cfg.get("bce_weight")
        sw = cfg.get("dice_weight") if loss == "bce_dice" else cfg.get("iou_weight")
        suffix = "dice" if loss == "bce_dice" else "iou"
        if bw is not None:
            parts.append(f"bce={_fmt(bw)}")
        if sw is not None:
            parts.append(f"{suffix}={_fmt(sw)}")
        pw = cfg.get("pos_weight")
        if pw is not None and pw > 0:
            parts.append(f"pos={_fmt(pw)}")

    elif loss == "focal":
        parts.append(f"alpha={_fmt(cfg.get('alpha'))}")
        parts.append(f"gamma={_fmt(cfg.get('gamma'))}")

    elif loss == "focal_dice":
        parts.append(f"fw={_fmt(cfg.get('focal_weight'))}")
        parts.append(f"alpha={_fmt(cfg.get('alpha'))}")
        parts.append(f"gamma={_fmt(cfg.get('gamma'))}")

    return ", ".join(parts) if parts else "-"


def data_label(cfg: dict) -> str:
    return "new" if "Dataset" in str(cfg.get("data_root", "")) else "old"


# ──────────────────────────────────────────────
# 지표 로딩
# ──────────────────────────────────────────────

def load_metrics(exp_dir: Path) -> Optional[dict]:
    """training_log.json에서 val_loss 기준 best epoch의 지표를 반환합니다."""
    log_path = exp_dir / "training_log.json"
    if not log_path.exists():
        return None
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if not log:
        return None
    best = min(log, key=lambda x: x["val_loss"])
    return {
        "total_epochs": len(log),
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_dice": best["val_dice"],
        "val_iou": best["val_iou"],
    }


# ──────────────────────────────────────────────
# 표 출력
# ──────────────────────────────────────────────

HEADERS = ["Exp", "Model", "Loss", "Key Params", "Aug", "Data", "Size", "Epochs", "Best Ep", "Val Dice", "Val IoU"]
KEYS    = ["exp", "model", "loss", "params",     "aug", "data", "size", "epochs", "best_ep", "val_dice",  "val_iou"]


def print_markdown(rows: list[dict]) -> None:
    col_widths = [len(h) for h in HEADERS]
    for row in rows:
        for i, k in enumerate(KEYS):
            col_widths[i] = max(col_widths[i], len(str(row[k])))

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, col_widths)) + " |"

    print(fmt_row(HEADERS))
    print(sep)
    for row in rows:
        print(fmt_row([row[k] for k in KEYS]))


def write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=KEYS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장 완료: {output_path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def collect_rows(outputs_dir: Path) -> list[dict]:
    rows = []
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue

        exp_id = exp_dir.name.split("__")[0]
        cfg = load_config(exp_dir)
        metrics = load_metrics(exp_dir)

        rows.append({
            "exp":      exp_id,
            "model":    cfg.get("model_name", "unet"),
            "loss":     cfg.get("loss_type") or "?",
            "params":   fmt_params(cfg),
            "aug":      "Y" if cfg.get("use_augmentation") else "-",
            "data":     data_label(cfg),
            "size":     cfg.get("image_size", 512),
            "epochs":   metrics["total_epochs"] if metrics else "-",
            "best_ep":  metrics["best_epoch"]   if metrics else "-",
            "val_dice": f'{metrics["val_dice"]:.4f}' if metrics else "-",
            "val_iou":  f'{metrics["val_iou"]:.4f}'  if metrics else "-",
            # 정렬용 숫자값 (표에는 미출력)
            "_val_iou":  metrics["val_iou"]  if metrics else -1.0,
            "_val_dice": metrics["val_dice"] if metrics else -1.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="실험 결과 요약 표 생성")
    parser.add_argument(
        "--dir", default="./outputs/unet",
        help="실험 폴더 경로 (기본: outputs/unet)",
    )
    parser.add_argument(
        "--sort", choices=["exp", "iou", "dice"], default="exp",
        help="정렬 기준 — exp: 실험 번호순 | iou/dice: 내림차순 (기본: exp)",
    )
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="출력 형식 (기본: markdown)",
    )
    parser.add_argument(
        "--output", default=None,
        help="저장할 파일 경로 (csv 형식일 때 사용, 미지정 시 stdout)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = (project_root / args.dir).resolve()

    if not outputs_dir.exists():
        print(f"폴더를 찾을 수 없습니다: {outputs_dir}")
        sys.exit(1)

    rows = collect_rows(outputs_dir)

    if args.sort == "iou":
        rows.sort(key=lambda r: r["_val_iou"], reverse=True)
    elif args.sort == "dice":
        rows.sort(key=lambda r: r["_val_dice"], reverse=True)

    # 정렬용 내부 키 제거
    for row in rows:
        row.pop("_val_iou")
        row.pop("_val_dice")

    if args.format == "csv":
        out_path = Path(args.output) if args.output else project_root / "experiment_results.csv"
        write_csv(rows, out_path)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()

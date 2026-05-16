"""
마라톤 경로 후처리 파이프라인
------------------------------
Step 1. Connected component 탐색 + 주경로 선택 (픽셀 크기 기준)
Step 2. 형태 기반 노이즈 제거 (area × circularity × skeleton_length)
Step 3. 끊어진 경로 조각 연결 (iterative endpoint-based merging)
Step 4. 잔여 fragment 제거 (연결되지 못한 소형 조각 제거)

각 단계 결과 이미지를 분리 저장하고, 전체 흐름을 5-패널로 시각화한다.

실행 예:
  python pipeline.py /path/to/masks
  python pipeline.py /path/to/masks --area-thresh 150 --circ-thresh 0.65 --max-distance 80
  python pipeline.py /path/to/masks --final-min-size 50
  python pipeline.py /path/to/masks --no-vis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 스크립트 직접 실행 시 프로젝트 루트를 경로에 추가
if __package__ in (None, ""):
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from src.postprocess.components import (
    load_mask,
    find_connected_components,
    select_main_path,
)
from src.postprocess.filter_noise import filter_noise
from src.postprocess.connect_fragments import find_components, iterative_fragment_connection, remove_small_fragments


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    mask_uint8: np.ndarray,
    area_thresh: int,
    circ_thresh: float,
    skel_thresh: int,
    max_distance: float,
    min_fragment_size: int,
    line_thickness: int,
    morph_close_size: int,
    final_size_thresh: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, List, List]:
    """
    전체 후처리 파이프라인을 실행한다.

    Args:
        mask_uint8: (H, W) uint8, 경로=255 배경=0

    Returns:
        main_mask     : Step 1 주경로 bool mask
        noise_mask    : Step 2 제거된 노이즈 uint8 mask
        filtered_mask : Step 2 노이즈 제거 후 uint8 mask
        connected_mask: Step 3 연결 완료 uint8 mask
        final_mask    : Step 4 잔여 fragment 제거 uint8 mask
        features      : 컴포넌트별 feature dict
        noise_labels  : 제거된 레이블 리스트
        connect_log   : fragment 연결 로그
    """
    bool_mask = mask_uint8 > 127  # scipy CC는 bool 배열 사용

    # ── Step 1: 주경로 선택 ──────────────────────────────────────────────
    labeled_mask, components = find_connected_components(bool_mask)
    main_mask, main_comp = select_main_path(labeled_mask, components)

    if verbose:
        total = sum(c["size"] for c in components)
        main_px = main_comp.get("size", 0)
        ratio = main_px / total * 100 if total > 0 else 0
        print(f"  [Step 1] {len(components)}개 연결 요소 "
              f"→ 주경로 {main_px:,} px ({ratio:.1f}%)")

    # ── Step 2: 노이즈 제거 ──────────────────────────────────────────────
    filtered_mask, noise_mask, features, noise_labels = filter_noise(
        mask_uint8,
        area_thresh=area_thresh,
        circ_thresh=circ_thresh,
        skel_thresh=skel_thresh,
    )

    if verbose:
        kept = len(features) - len(noise_labels)
        print(f"  [Step 2] 노이즈 {len(noise_labels)}개 제거 / {kept}개 보존")
        _print_feature_table(features, noise_labels)

    # ── Step 3: Fragment 연결 ─────────────────────────────────────────────
    if verbose:
        print(f"  [Step 3] Fragment 연결 시작...")

    connected_mask, connect_log = iterative_fragment_connection(
        filtered_mask,
        max_distance=max_distance,
        min_fragment_size=min_fragment_size,
        line_thickness=line_thickness,
        morph_close_size=morph_close_size,
        verbose=verbose,
    )

    # ── Step 4: 잔여 fragment 제거 ──────────────────────────────────────────
    if verbose:
        print(f"  [Step 4] 잔여 fragment 제거 시작...")

    final_mask = remove_small_fragments(connected_mask, min_size=final_size_thresh)

    if verbose:
        num_conn,  _, _, _ = find_components(connected_mask)
        num_clean, _, _, _ = find_components(final_mask)
        removed = (num_conn - 1) - (num_clean - 1)
        print(f"  [Step 4] 잔여 fragment {removed}개 제거 → {num_clean - 1}개")

    return main_mask, noise_mask, filtered_mask, connected_mask, final_mask, features, noise_labels, connect_log


def _print_feature_table(features: Dict, noise_labels: List) -> None:
    """Step 2 컴포넌트 feature 테이블을 출력한다."""
    noise_set = set(noise_labels)
    print(f"  {'레이블':>6}  {'크기(px)':>9}  {'원형도':>8}  "
          f"{'스켈레톤':>8}  {'결과':>6}")
    for feat in sorted(features.values(), key=lambda x: x["area"], reverse=True):
        decision = "noise" if feat["label"] in noise_set else "keep "
        marker = " ← 제거" if feat["label"] in noise_set else ""
        role = "MAIN" if feat["is_main"] else "    "
        print(f"  {feat['label']:>6}  {feat['area']:>9,}  "
              f"{feat['circularity']:>8.4f}  {feat['skeleton_length']:>8}  "
              f"{role} {decision}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# 단계별 이미지 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_step_images(
    original_mask: np.ndarray,
    main_mask: np.ndarray,
    noise_mask: np.ndarray,
    filtered_mask: np.ndarray,
    connected_mask: np.ndarray,
    final_mask: np.ndarray,
    out_dir: Path,
    stem: str,
) -> None:
    """각 단계 결과를 분리된 PNG 파일로 저장한다."""
    cv2.imwrite(str(out_dir / f"{stem}_step0_original.png"),      original_mask)
    cv2.imwrite(str(out_dir / f"{stem}_step1_main_path.png"),     main_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / f"{stem}_step2_noise_removed.png"), filtered_mask)
    cv2.imwrite(str(out_dir / f"{stem}_step3_connected.png"),     connected_mask)
    cv2.imwrite(str(out_dir / f"{stem}_step4_cleaned.png"),       final_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(
    original_mask: np.ndarray,
    main_mask: np.ndarray,
    noise_mask: np.ndarray,
    filtered_mask: np.ndarray,
    connected_mask: np.ndarray,
    final_mask: np.ndarray,
    noise_labels: List,
    title: str = "Marathon Path Postprocess Pipeline",
    save_path: Optional[Path] = None,
) -> None:
    """
    5-패널 시각화.

    패널 (1) 원본 mask
    패널 (2) 주경로 선택 — 흰색=주경로, 회색=기타 경로 픽셀
    패널 (3) 노이즈 제거 — 빨간색=제거된 노이즈
    패널 (4) Fragment 연결 결과
    패널 (5) 잔여 fragment 제거 최종 결과 — 빨간색=제거된 조각
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    H, W = original_mask.shape

    # ── (1) 원본 ──────────────────────────────────────────────────────────
    num_orig, _, _, _ = find_components(original_mask)
    axes[0].imshow(original_mask, cmap="gray")
    axes[0].set_title(f"원본 마스크\n{num_orig - 1}개 연결 요소")
    axes[0].axis("off")

    # ── (2) 주경로 선택 ───────────────────────────────────────────────────
    overlay2 = np.zeros((H, W, 3), dtype=np.float32)
    other_path = (original_mask > 0) & ~main_mask
    overlay2[main_mask] = [1.0, 1.0, 1.0]
    overlay2[other_path] = [0.4, 0.4, 0.4]
    axes[1].imshow(overlay2)
    axes[1].set_title("주경로 선택 (Step 1)\n흰색=주경로 / 회색=기타")
    axes[1].axis("off")

    # ── (3) 노이즈 제거 ───────────────────────────────────────────────────
    overlay3 = np.stack([original_mask] * 3, axis=-1).astype(np.float32) / 255.0
    overlay3[noise_mask > 0] = [1.0, 0.18, 0.18]
    axes[2].imshow(np.clip(overlay3, 0.0, 1.0))
    axes[2].set_title(f"노이즈 제거 (Step 2)\n빨간색 = {len(noise_labels)}개 제거")
    axes[2].axis("off")

    # ── (4) Fragment 연결 ────────────────────────────────────────────────
    num_conn, _, _, _ = find_components(connected_mask)
    axes[3].imshow(connected_mask, cmap="gray")
    axes[3].set_title(f"Fragment 연결 (Step 3)\n{num_conn - 1}개 연결 요소")
    axes[3].axis("off")

    # ── (5) 잔여 fragment 제거 ───────────────────────────────────────────
    residual = (connected_mask > 0) & (final_mask == 0)
    overlay5 = np.stack([final_mask] * 3, axis=-1).astype(np.float32) / 255.0
    overlay5[residual] = [1.0, 0.18, 0.18]
    num_final, _, _, _ = find_components(final_mask)
    axes[4].imshow(np.clip(overlay5, 0.0, 1.0))
    axes[4].set_title(f"잔여 제거 (Step 4)\n빨간색=제거 / {num_final - 1}개")
    axes[4].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  시각화 저장: {save_path}")
    else:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="마라톤 경로 후처리 파이프라인: 주경로 선택 → 노이즈 제거 → Fragment 연결",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mask_dir", type=str,
        help="처리할 *_mask.png 파일이 있는 폴더 경로",
    )

    g2 = parser.add_argument_group("Step 2 — 노이즈 제거 threshold")
    g2.add_argument("--area-thresh",  type=int,   default=250,
                    help="이 픽셀 수 미만이면 노이즈 후보")
    g2.add_argument("--circ-thresh",  type=float, default=0.5,
                    help="circularity 초과 시 원형으로 간주 (0~1)")
    g2.add_argument("--skel-thresh",  type=int,   default=400,
                    help="skeleton 길이(px) 미만이면 노이즈 후보")

    g3 = parser.add_argument_group("Step 3 — Fragment 연결 설정")
    g3.add_argument("--max-distance",      type=float, default=150.0,
                    help="endpoint 간 최대 연결 거리 (px)")
    g3.add_argument("--min-fragment-size", type=int,   default=10,
                    help="연결 대상 fragment 최소 픽셀 수")
    g3.add_argument("--line-thickness",    type=int,   default=2,
                    help="연결선 두께 (px)")
    g3.add_argument("--morph-close",       type=int,   default=0,
                    help="morphology closing 커널 크기 (0=비활성화, 권장 3~7)")

    g4 = parser.add_argument_group("Step 4 — 잔여 fragment 제거")
    g4.add_argument("--final-min-size",    type=int,   default=0,
                    help="Step 3 후 남은 fragment 중 이 픽셀 수 미만을 제거. 0=주경로만 보존.")

    parser.add_argument("--no-vis", action="store_true",
                        help="시각화를 저장하지 않는다")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mask_dir = Path(args.mask_dir)
    if not mask_dir.exists():
        print(f"❌ 경로가 존재하지 않습니다: {mask_dir}")
        sys.exit(1)

    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    if not mask_paths:
        print(f"❌ *_mask.png 파일을 찾을 수 없습니다: {mask_dir}")
        sys.exit(1)

    out_dir = mask_dir / "postprocess_results"
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = mask_dir / "visualization" / "pipeline"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"처리 대상 : {len(mask_paths)}개 파일")
    print(f"[Step 2]  area < {args.area_thresh} px  |  "
          f"circ > {args.circ_thresh}  |  skel < {args.skel_thresh} px")
    print(f"[Step 3]  max_distance={args.max_distance} px  |  "
          f"min_fragment={args.min_fragment_size} px  |  "
          f"line={args.line_thickness} px")
    print(f"[Step 4]  final_min_size={args.final_min_size} px "
          f"({'주경로만 보존' if args.final_min_size == 0 else f'{args.final_min_size} px 미만 제거'})")

    for mask_path in mask_paths:
        print(f"\n{'=' * 60}")
        print(f"파일: {mask_path.name}")

        # uint8 로드 (파이프라인 전체 사용)
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask_uint8 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        (main_mask, noise_mask, filtered_mask, connected_mask, final_mask,
         features, noise_labels, connect_log) = run_pipeline(
            mask_uint8,
            area_thresh=args.area_thresh,
            circ_thresh=args.circ_thresh,
            skel_thresh=args.skel_thresh,
            max_distance=args.max_distance,
            min_fragment_size=args.min_fragment_size,
            line_thickness=args.line_thickness,
            morph_close_size=args.morph_close,
            final_size_thresh=args.final_min_size,
            verbose=True,
        )

        # 단계별 이미지 저장
        save_step_images(
            mask_uint8, main_mask, noise_mask, filtered_mask, connected_mask, final_mask,
            steps_dir, mask_path.stem,
        )

        # 최종 결과 저장
        final_path = out_dir / mask_path.name
        cv2.imwrite(str(final_path), final_mask)

        # 시각화
        if not args.no_vis:
            visualize_pipeline(
                mask_uint8, main_mask, noise_mask, filtered_mask, connected_mask, final_mask,
                noise_labels,
                title=f"Pipeline — {mask_path.name}",
                save_path=vis_dir / f"{mask_path.stem}.png",
            )

        # 요약
        num_orig,  _, _, _ = find_components(mask_uint8)
        num_conn,  _, _, _ = find_components(connected_mask)
        num_final, _, _, _ = find_components(final_mask)
        print(f"\n결과 저장 : {final_path}")
        print(f"요약      : {num_orig - 1}개 "
              f"→ 노이즈 {len(noise_labels)}개 제거 "
              f"→ {len(connect_log)}회 연결 "
              f"→ {num_conn - 1}개 "
              f"→ 잔여 {(num_conn - 1) - (num_final - 1)}개 제거 "
              f"→ {num_final - 1}개")


if __name__ == "__main__":
    main()

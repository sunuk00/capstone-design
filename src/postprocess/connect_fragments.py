"""
끊어진 경로 조각 연결 (Iterative Fragment Connection).

Main path ↔ Fragment를 반복적으로 연결하여 하나의 연속된 경로로 복원한다.

알고리즘:
  1. CC 계산 → 가장 큰 component = main path
    2. Main mask의 거리 변환 맵 계산
    3. fragment 외곽 픽셀 중 main에 가장 가까운 지점 탐색
    4. max_distance 이하면 midpoint polyline으로 연결 + 병합
  5. 반복 (main path가 점진적으로 확장됨)

Public API:
    find_components(mask)              → (num_labels, labels, stats, centroids)
    iterative_fragment_connection(...) → (result_mask, log)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def find_components(
    mask: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    8-연결성 기준 connected components 계산.

    Args:
        mask: (H, W) uint8, 경로=255 배경=0

    Returns:
        num_labels, labels, stats, centroids  (cv2.connectedComponentsWithStats 결과)
    """
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def iterative_fragment_connection(
    initial_mask: np.ndarray,
    max_distance: float = 50.0,
    min_fragment_size: int = 10,
    line_thickness: int = 2,
    morph_close_size: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Main path ↔ Fragment를 반복적으로 연결한다.
    매 반복마다 가장 가까운 fragment 하나를 main path에 병합한다.

    Args:
        initial_mask     : (H, W) uint8, 경로=255
        max_distance     : 최대 연결 거리 (px). 초과 시 중단.
        min_fragment_size: 이 픽셀 수 미만 fragment 무시
        line_thickness   : 연결선 두께 (px)
        morph_close_size : morphology closing 커널 크기 (0=비활성화, 권장 3~7)
        verbose          : 반복 진행 상황 출력 여부

    Returns:
        result_mask: 연결 완료된 mask (H, W) uint8
        log        : [{"iteration", "fragment_label", "fragment_area_px",
                       "distance_px", "main_endpoint", "frag_endpoint"}, ...]
    """
    mask = initial_mask.copy()

    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if verbose:
            print(f"  [Morphology Close] kernel={morph_close_size}px")

    log: List[dict] = []

    for iteration in range(1, 10_000):
        num_labels, labels, stats, _ = find_components(mask)
        main_label, frag_labels = _split_main_fragments(num_labels, stats)

        if main_label is None:
            break

        frag_labels = [
            lbl for lbl in frag_labels
            if stats[lbl, cv2.CC_STAT_AREA] >= min_fragment_size
        ]
        if not frag_labels:
            if verbose:
                area = int(stats[main_label, cv2.CC_STAT_AREA])
                print(f"  [Iter {iteration}] 연결 가능한 fragment 없음. "
                      f"Main: {area:,} px. 종료.")
            break

        main_mask = (labels == main_label).astype(np.uint8) * 255
        main_dt, main_dt_labels, main_dt_coords = _build_main_distance_transform(main_mask)
        if main_dt is None or len(main_dt_coords) == 0:
            break

        # 가장 가까운 fragment 탐색
        best: dict = {"dist": float("inf")}
        for frag_lbl in frag_labels:
            frag_mask = (labels == frag_lbl).astype(np.uint8) * 255
            frag_pt, mp, dist = _find_distance_transform_connection(
                main_dt,
                main_dt_labels,
                main_dt_coords,
                frag_mask,
            )
            if mp is not None and dist < best["dist"]:
                best = {
                    "dist": dist,
                    "label": frag_lbl,
                    "area": int(stats[frag_lbl, cv2.CC_STAT_AREA]),
                    "main_pt": mp,
                    "frag_pt": frag_pt,
                    "frag_mask": frag_mask,
                }

        if best["dist"] > max_distance:
            if verbose:
                print(f"  [Iter {iteration}] 최근접 fragment 거리 "
                      f"{best['dist']:.1f} px > {max_distance} px. 종료.")
            break

        # 연결 수행: fragment 병합 + midpoint polyline
        mask[best["frag_mask"] > 0] = 255
        _draw_polyline_bridge(
            mask,
            best["main_pt"],
            best["frag_pt"],
            line_thickness,
        )
        log.append({
            "iteration": iteration,
            "fragment_label": int(best["label"]),
            "fragment_area_px": best["area"],
            "distance_px": round(best["dist"], 2),
            "main_point": best["main_pt"].tolist(),
            "frag_point": best["frag_pt"].tolist(),
        })
        if verbose:
            print(f"  [Iter {iteration:>3}] Fragment #{best['label']:>4} 연결 — "
                  f"크기: {best['area']:>6,} px, 거리: {best['dist']:>7.1f} px")

    return mask, log


def remove_small_fragments(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    주경로(최대 component)를 제외한 fragment 중 min_size 미만인 것을 제거한다.

    Args:
        mask    : (H, W) uint8, 경로=255
        min_size: 이 픽셀 수 미만의 fragment를 제거. 0이면 주경로만 보존.

    Returns:
        cleaned mask (H, W) uint8
    """
    num_labels, labels, stats, _ = find_components(mask)
    if num_labels <= 1:
        return mask.copy()

    main_label = max(range(1, num_labels), key=lambda lbl: int(stats[lbl, cv2.CC_STAT_AREA]))

    cleaned = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        if lbl == main_label:
            cleaned[labels == lbl] = 255
        elif min_size > 0 and int(stats[lbl, cv2.CC_STAT_AREA]) >= min_size:
            cleaned[labels == lbl] = 255
    return cleaned


# ── Internal ──────────────────────────────────────────────────────────────────

def _split_main_fragments(
    num_labels: int,
    stats: np.ndarray,
) -> Tuple[Optional[int], List[int]]:
    if num_labels <= 1:
        return None, []
    areas = [
        (lbl, int(stats[lbl, cv2.CC_STAT_AREA]))
        for lbl in range(1, num_labels)
    ]
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[0][0], [lbl for lbl, _ in areas[1:]]


def _compute_skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 127)


def _extract_boundary_points(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.empty((0, 2), dtype=np.int64)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = binary & (cv2.erode(binary, kernel, iterations=1) == 0)
    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([rows, cols])


def _build_main_distance_transform(
    main_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    binary = (main_mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return None, None, np.empty((0, 2), dtype=np.int64)

    src = (binary == 0).astype(np.uint8)
    main_dt, main_dt_labels = cv2.distanceTransformWithLabels(
        src,
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    main_dt_coords = np.column_stack(np.where(binary > 0))
    return main_dt, main_dt_labels, main_dt_coords


def _find_distance_transform_connection(
    main_dt: np.ndarray,
    main_dt_labels: np.ndarray,
    main_dt_coords: np.ndarray,
    frag_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    frag_pts = _extract_boundary_points(frag_mask)
    if len(frag_pts) == 0:
        rows, cols = np.where(frag_mask > 0)
        if len(rows) == 0:
            return None, None, float("inf")
        frag_pts = np.column_stack([rows, cols])

    dists = main_dt[frag_pts[:, 0], frag_pts[:, 1]]
    best_idx = int(np.argmin(dists))
    frag_pt = frag_pts[best_idx]

    label = int(main_dt_labels[frag_pt[0], frag_pt[1]])
    main_idx = label - 1
    if main_idx < 0 or main_idx >= len(main_dt_coords):
        return None, None, float("inf")

    main_pt = main_dt_coords[main_idx]
    return frag_pt, main_pt, float(dists[best_idx])


def _draw_polyline_bridge(
    mask: np.ndarray,
    main_pt: np.ndarray,
    frag_pt: np.ndarray,
    line_thickness: int,
) -> None:
    mid_pt = np.rint((main_pt.astype(np.float32) + frag_pt.astype(np.float32)) / 2.0).astype(np.int32)
    pts = np.array([
        [int(main_pt[1]), int(main_pt[0])],
        [int(mid_pt[1]), int(mid_pt[0])],
        [int(frag_pt[1]), int(frag_pt[0])],
    ], dtype=np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)

"""
끊어진 경로 조각 연결 (Iterative Fragment Connection).

Main path ↔ Fragment를 반복적으로 연결하여 하나의 연속된 경로로 복원한다.

알고리즘:
  1. CC 계산 → 가장 큰 component = main path
  2. Skeletonize → 8-neighbor 기반 endpoint 탐색
  3. main endpoint ↔ fragment endpoint 최소 거리 쌍 탐색
  4. max_distance 이하면 cv2.line으로 연결 + 병합
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
        line_thickness   : endpoint 연결선 두께 (px)
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
        main_pts = _get_endpoints_or_boundary(main_mask)
        if len(main_pts) == 0:
            break

        # 가장 가까운 fragment 탐색
        best: dict = {"dist": float("inf")}
        for frag_lbl in frag_labels:
            frag_mask = (labels == frag_lbl).astype(np.uint8) * 255
            frag_pts = _get_endpoints_or_boundary(frag_mask)
            mp, fp, dist = _find_closest_pair(main_pts, frag_pts)
            if mp is not None and dist < best["dist"]:
                best = {
                    "dist": dist,
                    "label": frag_lbl,
                    "area": int(stats[frag_lbl, cv2.CC_STAT_AREA]),
                    "main_pt": mp,
                    "frag_pt": fp,
                    "frag_mask": frag_mask,
                }

        if best["dist"] > max_distance:
            if verbose:
                print(f"  [Iter {iteration}] 최근접 fragment 거리 "
                      f"{best['dist']:.1f} px > {max_distance} px. 종료.")
            break

        # 연결 수행: fragment 병합 + endpoint 간 직선
        mask[best["frag_mask"] > 0] = 255
        cv2.line(
            mask,
            (int(best["main_pt"][1]), int(best["main_pt"][0])),  # (col, row)
            (int(best["frag_pt"][1]),  int(best["frag_pt"][0])),
            255,
            line_thickness,
        )
        log.append({
            "iteration": iteration,
            "fragment_label": int(best["label"]),
            "fragment_area_px": best["area"],
            "distance_px": round(best["dist"], 2),
            "main_endpoint": best["main_pt"].tolist(),
            "frag_endpoint": best["frag_pt"].tolist(),
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


def _detect_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """8-neighbor 중 연결 수가 1인 픽셀 = endpoint."""
    skel_u8 = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel)
    endpoint_mask = skeleton & (neighbor_count == 1)
    rows, cols = np.where(endpoint_mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([rows, cols])


def _get_endpoints_or_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Skeleton endpoint를 구한다.
    없으면 skeleton 전체 픽셀로, 그마저도 없으면 mask 전체 픽셀로 대체한다.
    """
    skel = _compute_skeleton(mask)
    eps = _detect_endpoints(skel)
    if len(eps) > 0:
        return eps
    rows, cols = np.where(skel)
    if len(rows) > 0:
        return np.column_stack([rows, cols])
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([rows, cols])


def _find_closest_pair(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """두 점 집합 사이의 최소 거리 쌍을 브로드캐스트 연산으로 탐색한다."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None, None, float("inf")
    diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return pts_a[idx[0]], pts_b[idx[1]], float(dists[idx])

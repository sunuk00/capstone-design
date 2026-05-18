"""
형태 기반 노이즈 제거 (Shape-based Noise Filtering).

제거 조건 (점수 기반, score >= 2 이면 제거):
    area < area_thresh            → 2점
    circularity > circ_thresh     → 1점
    extent > extent_thresh        → 1점
    skeleton_length < skel_thresh → 1점  (단, 길쭉한 path-like component는 보호)

Public API:
    filter_noise(mask, area_thresh, circ_thresh, skel_thresh)
        → (filtered_mask, noise_mask, features, noise_labels)
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def filter_noise(
    mask: np.ndarray,
    area_thresh: int = 200,
    circ_thresh: float = 0.6,
    skel_thresh: int = 30,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, dict], List[int]]:
    """
    Binary mask에서 형태 기반으로 노이즈를 제거한다.
    주경로(가장 큰 component)는 어떤 경우에도 보존한다.

    Args:
        mask        : (H, W) uint8, 경로=255 배경=0
        area_thresh : 이 픽셀 수 미만이면 2점
        circ_thresh : circularity 가 이 값 초과면 1점 (0~1)
        skel_thresh : skeleton 길이(px)가 이 값 미만이면 1점

    Notes:
        권장 기본값 예시:
            - 512 기준  : area_thresh=250, circ_thresh=0.50, skel_thresh=400
            - 1024 기준 : area_thresh=500, circ_thresh=0.50, skel_thresh=800

    Returns:
        filtered_mask: 노이즈 제거된 mask (H, W) uint8
        noise_mask   : 제거된 픽셀만 255인 mask (H, W) uint8
        features     : {label: {"area", "circularity", "skeleton_length",
                                "bbox_width", "bbox_height", "bbox_area",
                                "extent", "bbox_aspect_ratio", "noise_score",
                                "is_main"}}
        noise_labels : 제거된 레이블 리스트
    """
    num_labels, labels, stats, _ = _find_components(mask)

    if num_labels <= 1:
        return mask.copy(), np.zeros_like(mask), {}, []

    main_label = max(
        range(1, num_labels),
        key=lambda lbl: stats[lbl, cv2.CC_STAT_AREA],
    )

    features = _extract_features(mask, labels, stats, num_labels, main_label)
    noise_labels = _classify_noise(features, area_thresh, circ_thresh, skel_thresh)

    filtered_mask = mask.copy()
    noise_mask = np.zeros_like(mask)
    for lbl in noise_labels:
        region = labels == lbl
        noise_mask[region] = 255
        filtered_mask[region] = 0

    return filtered_mask, noise_mask, features, noise_labels


# ── Internal ──────────────────────────────────────────────────────────────────

def _find_components(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _compute_circularity(area: int, comp_mask: np.ndarray) -> float:
    """circularity = 4πA / P²  (원형=1, 길쭉한 형태→0)"""
    contours, _ = cv2.findContours(
        comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    if perimeter == 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _extract_features(
    mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    num_labels: int,
    main_label: int,
) -> Dict[int, dict]:
    features: Dict[int, dict] = {}
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        width = int(stats[lbl, cv2.CC_STAT_WIDTH])
        height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        bbox_area = max(width * height, 1)
        extent = float(area / bbox_area)
        short_side = max(min(width, height), 1)
        bbox_aspect_ratio = float(max(width, height) / short_side)
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        skel_len = int(skeletonize(comp_mask > 0).sum())
        features[lbl] = {
            "label": lbl,
            "area": area,
            "circularity": round(_compute_circularity(area, comp_mask), 4),
            "skeleton_length": skel_len,
            "bbox_width": width,
            "bbox_height": height,
            "bbox_area": bbox_area,
            "extent": round(extent, 4),
            "bbox_aspect_ratio": round(bbox_aspect_ratio, 4),
            "noise_score": 0,
            "noise_reasons": [],
            "is_main": (lbl == main_label),
        }
    return features


def _classify_noise(
    features: Dict[int, dict],
    area_thresh: int,
    circ_thresh: float,
    skel_thresh: int,
) -> List[int]:
    extent_thresh = 0.45
    path_like_aspect_thresh = 2.5
    noise: List[int] = []
    for lbl, feat in features.items():
        if feat["is_main"]:
            feat["noise_score"] = 0
            feat["noise_reasons"] = []
            continue

        path_like = feat["bbox_aspect_ratio"] >= path_like_aspect_thresh
        score = 0
        reasons = []

        if feat["area"] < area_thresh:
            score += 2
            reasons.append("area")
        if feat["circularity"] > circ_thresh:
            score += 1
            reasons.append("circularity")
        if feat["extent"] > extent_thresh and not path_like:
            score += 1
            reasons.append("extent")
        if feat["skeleton_length"] < skel_thresh and feat["extent"] > extent_thresh and not path_like:
            score += 1
            reasons.append("skeleton")

        feat["noise_score"] = score
        feat["noise_reasons"] = reasons
        if score >= 2:
            noise.append(lbl)
    return noise

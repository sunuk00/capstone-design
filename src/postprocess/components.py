"""
Connected component 탐색 및 주경로 선택.

Public API:
    load_mask(path)                         → np.ndarray (bool)
    find_connected_components(binary_mask)  → (labeled_mask, components)
    select_main_path(labeled_mask, components) → (main_mask, main_comp)
"""

from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def load_mask(mask_path: str | Path) -> np.ndarray:
    """
    PNG 마스크를 bool 배열로 로드한다.
    픽셀 값 > 127 → True (경로), 그 외 → False (배경).

    Returns:
        (H, W) bool
    """
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return arr > 127


def find_connected_components(
    binary_mask: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    """
    이진 마스크에서 4-연결성 connected components를 탐색하고
    픽셀 수 기준 내림차순으로 정렬한다.

    scipy가 설치되어 있으면 C 레벨 구현을 사용하고,
    없으면 BFS로 폴백한다.

    Args:
        binary_mask: (H, W) bool

    Returns:
        labeled_mask: (H, W) int32, 레이블 1,2,3... / 배경=0
        components  : [{"label", "size", "rank"}, ...] 크기 내림차순
    """
    labeled_mask, num_labels = _label_components(binary_mask)

    counts = np.bincount(labeled_mask.ravel())
    components = [
        {"label": lbl, "size": int(counts[lbl])}
        for lbl in range(1, num_labels + 1)
    ]
    components.sort(key=lambda x: x["size"], reverse=True)
    for rank, comp in enumerate(components, start=1):
        comp["rank"] = rank

    return labeled_mask, components


def select_main_path(
    labeled_mask: np.ndarray,
    components: List[dict],
) -> Tuple[np.ndarray, dict]:
    """
    가장 큰 component(rank=1)를 주경로로 확정한다.

    Args:
        labeled_mask: find_connected_components 의 labeled_mask
        components  : find_connected_components 의 components

    Returns:
        main_mask: (H, W) bool, 주경로 픽셀=True
        main_comp: {"label", "size", "rank"}
    """
    if not components:
        return np.zeros(labeled_mask.shape, dtype=bool), {}

    main_comp = components[0]
    main_mask = labeled_mask == main_comp["label"]
    return main_mask, main_comp


# ── Internal ──────────────────────────────────────────────────────────────────

def _label_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label as scipy_label
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        labeled, n = scipy_label(binary_mask, structure=structure)
        return labeled.astype(np.int32), int(n)
    except ImportError:
        return _label_bfs(binary_mask)


def _label_bfs(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = binary_mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    path_ys, path_xs = np.where(binary_mask)

    for i in range(len(path_ys)):
        y, x = int(path_ys[i]), int(path_xs[i])
        if labeled[y, x] != 0:
            continue
        current_label += 1
        queue = deque([(y, x)])
        labeled[y, x] = current_label
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < H and 0 <= nx < W
                        and binary_mask[ny, nx]
                        and labeled[ny, nx] == 0):
                    labeled[ny, nx] = current_label
                    queue.append((ny, nx))

    return labeled, current_label

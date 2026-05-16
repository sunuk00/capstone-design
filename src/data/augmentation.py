"""
데이터 증강 모듈

마라톤 경로 분할 학습용으로 가벼운 기하학적 증강을 제공한다.
- 좌우 뒤집기
- 상하 뒤집기
- 작은 각도 회전
- 그레이스케일 변환 (색상 의존도를 줄이기 위한 선택적 증강)
"""

import random
from typing import Tuple

from PIL import Image, ImageOps


def apply_grayscale_augmentation(
    image: Image.Image,
    prob: float = 0.5,
) -> Image.Image:
    """
    이미지를 일정 확률로 그레이스케일로 변환한다.

    마라톤 경로 이미지는 지도마다 경로 색상이 다양하다(빨강, 파랑, 검정 등).
    이 증강을 적용하면 모델이 색상 단서에 과도하게 의존하지 않고
    경로의 형태(선, 연속성 등)를 학습하도록 유도할 수 있다.

    변환 방식:
        RGB → L(단일 채널 명도) → RGB 재변환
        채널 수는 3으로 유지되지만 R=G=B 인 무채색 이미지가 된다.
        마스크는 경로 위치 정보만 담고 있으므로 변환하지 않는다.

    Args:
        image: RGB PIL 이미지
        prob: 그레이스케일 변환을 적용할 확률

    Returns:
        변환된(또는 원본) RGB PIL 이미지
    """
    if random.random() < prob:
        # L 모드(명도 단일 채널)로 변환했다가 다시 RGB 3채널로 복원
        # → 색상 정보는 사라지고 밝기 정보만 남은 3채널 이미지
        image = image.convert("L").convert("RGB")
    return image


def apply_basic_augmentation(
    image: Image.Image,
    mask: Image.Image,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.2,
    rotate_prob: float = 0.3,
    max_rotate_deg: float = 10.0,
) -> Tuple[Image.Image, Image.Image]:
    """
    이미지-마스크 쌍에 동일한 랜덤 증강을 적용한다.

    Args:
        image: RGB PIL 이미지
        mask: L(단일 채널) PIL 마스크
        hflip_prob: 좌우 뒤집기 확률
        vflip_prob: 상하 뒤집기 확률
        rotate_prob: 회전 적용 확률
        max_rotate_deg: 회전 최대 각도 ([-max, +max])

    Returns:
        증강된 (image, mask)
    """
    if random.random() < hflip_prob:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    if random.random() < vflip_prob:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    if random.random() < rotate_prob:
        angle = random.uniform(-max_rotate_deg, max_rotate_deg)

        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )
        mask = mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )

    return image, mask

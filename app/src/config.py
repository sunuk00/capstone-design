"""Central configuration for marathon route extraction defaults."""

AREA_THRESH = 1000          # 이 픽셀 수 미만이면 노이즈 후보
CIRC_THRESH = 0.5           # circularity 초과 시 원형으로 간주 (0~1)
SKEL_THRESH = 400            # skeleton 길이(px) 미만이면 노이즈 후보
MAX_DISTANCE = 150.0        # fragment 연결 허용 최대 거리 (px)
MIN_FRAGMENT_SIZE = 10      # 연결 대상 fragment 최소 픽셀 수
LINE_THICKNESS = 2          # 연결선 두께 (px)
MORPH_CLOSE_SIZE = 0        # morphology closing 커널 크기. 미지정 시 입력 mask 단변의 1%%(최소 5)로 자동 계산
FINAL_SIZE_THRESH = 0       # Step 3 후 남은 fragment 중 이 픽셀 수 미만을 제거. 0=주경로만 보존.
SPUR_LENGTH = 20            # 이 픽셀 수 미만인 가지를 잔가지로 간주해 제거. 0=제거 안 함.
SKEL_MORPH_CLOSE = 0        # 스켈레톤화 전 morphology closing 커널 크기 (0=비활성화, 권장 3~7).


import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ── 세그멘테이션 모델 ────────────────────────────────────────────────────────
    # "segformer_unet_b2" 또는 "unet" — 환경변수 MODEL_TYPE으로 변경 가능
    MODEL_TYPE = os.environ.get("MODEL_TYPE", "segformer_unet_b2")
    MODEL_PATH = BASE_DIR / "weights" / (
        "segformer_unet_b2_best.pt" if MODEL_TYPE == "segformer_unet_b2" else "unet_best.pt"
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 768 if MODEL_TYPE == "segformer_unet_b2" else 512
    THRESHOLD = 0.5

    OPENING_ITERATIONS = 2
    CLOSING_ITERATIONS = 1
    MIN_COMPONENT_AREA = 1200

    # ── 웹 서버 ─────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB = 32
    HOST  = "0.0.0.0"
    PORT  = 8010
    DEBUG = False
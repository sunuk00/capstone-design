"""FastAPI server for marathon route extraction demo."""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import base64
import io
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel

from src.config import Config
if Config.MODEL_TYPE == "segformer_unet_b2":
    from src.marathon_route_extraction.segformer_unet_b2 import load_model, predict_mask
else:
    from src.marathon_route_extraction.unet import load_model, predict_mask
from src.config import (
    AREA_THRESH,
    CIRC_THRESH,
    SKEL_THRESH,
    MAX_DISTANCE,
    MIN_FRAGMENT_SIZE,
    LINE_THICKNESS,
    MORPH_CLOSE_SIZE,
    FINAL_SIZE_THRESH,
    SPUR_LENGTH,
    SKEL_MORPH_CLOSE,
)
from src.marathon_route_extraction.path_extractor import extract_ordered_path, auto_extract_ordered_path
from src.marathon_route_extraction.postprocess import postprocess_mask

# ── FastAPI App Setup ─────────────────────────────────────────────────────────

app = FastAPI(title="Marathon Route Extraction")

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Model singleton ───────────────────────────────────────────────────────────

_model: torch.nn.Module | None = None
_device: torch.device | None = None


def get_model() -> torch.nn.Module:
    global _model, _device
    if _model is None:
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found: {Config.MODEL_PATH}")
        _device = Config.DEVICE
        _model = load_model(str(Config.MODEL_PATH), _device)
        print(f"[info] Model loaded on device={_device}")
    return _model


def get_device() -> torch.device:
    if _device is None:
        get_model()
    return _device


# ── Utility functions ─────────────────────────────────────────────────────────

def _b64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode_mask(mask_b64: str) -> np.ndarray:
    """Decode base64 mask image to numpy array."""
    _, data = mask_b64.split(",", 1)
    buf = io.BytesIO(base64.b64decode(data))
    img = Image.open(buf).convert("L")
    return np.asarray(img, dtype=np.uint8)


# ── DEBUG ──────────────────────────────────────────────────────────────────────
def _debug_path_overlay(
    skeleton_arr: np.ndarray,
    path: list,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    bg_img: Image.Image | None = None,
) -> str:
    """
    Overlay the extracted pixel path on the marathon image (or skeleton as fallback).

    bg_img must be the same resolution as skeleton_arr (both 512×512 from the
    predict pipeline) so that path coordinates map 1-to-1 without any scaling.

    red squares (3×3) = extracted BFS path
    blue circle       = start point
    green circle      = end point
    """
    h, w = skeleton_arr.shape

    if bg_img is not None:
        # Convert to grayscale first to make the path more visible
        gray = bg_img.convert("L")
        base = gray.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        rgb = np.asarray(base, dtype=np.uint8).copy()
    else:
        gray = (skeleton_arr // 4).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw path as 3×3 red squares so individual pixels are visible on the map.
    for x, y in (path or []):
        ix, iy = int(x), int(y)
        if 0 <= iy < h and 0 <= ix < w:
            draw.rectangle([ix - 1, iy - 1, ix + 1, iy + 1], fill=(255, 50, 50))

    r = 7
    sx, sy = int(start_xy[0]), int(start_xy[1])
    ex, ey = int(end_xy[0]), int(end_xy[1])
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(60, 120, 255))   # blue = start
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(50, 220, 60))    # green = end
    return _b64(img)
# ── END DEBUG ──────────────────────────────────────────────────────────────────


# ── Request/Response Models ───────────────────────────────────────────────────

class PostprocessRequest(BaseModel):
    mask_b64: str
    # Use centralized defaults from src.config (module-level imports)
    area_thresh: int = AREA_THRESH
    circ_thresh: float = CIRC_THRESH
    skel_thresh: int = SKEL_THRESH
    max_distance: float = MAX_DISTANCE
    min_fragment_size: int = MIN_FRAGMENT_SIZE
    line_thickness: int = LINE_THICKNESS
    morph_close_size: int = MORPH_CLOSE_SIZE
    final_size_thresh: int = FINAL_SIZE_THRESH
    spur_length: int = SPUR_LENGTH
    skel_morph_close: int = SKEL_MORPH_CLOSE


class PointsRequest(BaseModel):
    skeleton_b64: str | None = None
    start: list[float]   # [x, y]
    end: list[float]     # [x, y]
    tau: float = 3.0
    angle_thresh: float = 20.0
    min_dist: float = 8.0
    input_img_b64: str | None = None  # ── DEBUG: 512×512 resized marathon image for overlay


class AutoExtractRequest(BaseModel):
    skeleton_b64: str
    input_img_b64: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(static_dir / "index.html", media_type="text/html")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and run mask prediction.
    
    Returns:
        - input_img_b64: resized input image
        - mask_b64: predicted mask
    """
    try:
        # Read uploaded file
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get model and device
        model = get_model()
        device = get_device()
        
        # Predict mask
        _, mask_pil = predict_mask(
            model=model,
            image_pil=image_pil,
            device=device,
            image_size=Config.IMAGE_SIZE,
            threshold=Config.THRESHOLD,
            min_component_area=20,
            opening_iterations=0,
            closing_iterations=0,
        )
        
        # input_img is now the original-resolution image; the browser already
        # holds it in state.uploadedImage, so we only return the mask.
        return JSONResponse({
            "status": "success",
            "mask_b64": _b64(mask_pil),
        })
    except Exception as e:
        print(f"[error] predict: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/postprocess")
async def postprocess(req: PostprocessRequest):
    """
    Perform post-processing using the 4-step marathon path pipeline, then skeletonize.

    Steps:
        1. Main path selection (largest connected component)
        2. Shape-based noise filtering (area × circularity × skeleton_length)
        3. Fragment connection (iterative endpoint-based merging)
        4. Residual fragment removal

    Returns:
        - skeleton_b64: skeletonized result image
    """
    try:
        mask_arr = _decode_mask(req.mask_b64)

        # postprocess_mask now returns a tuple of intermediate results
        # (main_mask, noise_mask, filtered_mask, connected_mask,
        #  final_mask, skeleton_mask, features, noise_labels, connect_log)
        res = postprocess_mask(
            mask_arr,
            area_thresh=req.area_thresh,
            circ_thresh=req.circ_thresh,
            skel_thresh=req.skel_thresh,
            max_distance=req.max_distance,
            min_fragment_size=req.min_fragment_size,
            line_thickness=req.line_thickness,
            morph_close_size=req.morph_close_size if req.morph_close_size > 0 else 0,
            final_size_thresh=req.final_size_thresh,
            spur_length=req.spur_length,
            skel_morph_close=req.skel_morph_close,
        )
        # Extract skeleton mask from returned tuple (6th element)
        if isinstance(res, tuple) or isinstance(res, list):
            skeleton_arr = res[5]
        else:
            skeleton_arr = res
        skeleton_img = Image.fromarray(skeleton_arr, mode="L")

        return JSONResponse({
            "status": "success",
            "skeleton_b64": _b64(skeleton_img),
        })
    except Exception as e:
        print(f"[error] postprocess: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract_path")
async def extract_path(req: PointsRequest):
    """
    Extract ordered path from start to end on skeleton.
    
    Input:
        - skeleton_b64: base64-encoded skeleton image
        - start: [x, y] start point
        - end: [x, y] end point
    
    Returns:
        - path: list of [x, y] tuples ordered start → end
    """
    try:
        if len(req.start) != 2 or len(req.end) != 2:
            raise ValueError("start/end must be [x, y]")
        if not req.skeleton_b64:
            raise ValueError("skeleton_b64 is required")

        skeleton_arr = _decode_mask(req.skeleton_b64)
        start_xy = (int(req.start[0]), int(req.start[1]))
        end_xy   = (int(req.end[0]),   int(req.end[1]))
        ordered = extract_ordered_path(
            skeleton_arr, start_xy, end_xy,
            tau=req.tau,
            angle_thresh=req.angle_thresh,
            min_dist=req.min_dist,
        )

        # ── DEBUG ──────────────────────────────────────────────────────────────
        # Decode the 512×512 marathon image so the overlay shares the same
        # coordinate space as the skeleton and no scaling is needed.
        bg_img: Image.Image | None = None
        if req.input_img_b64:
            _, _data = req.input_img_b64.split(",", 1)
            bg_img = Image.open(io.BytesIO(base64.b64decode(_data))).convert("RGB")
        # ── END DEBUG ──────────────────────────────────────────────────────────

        if ordered is None:
            # ── DEBUG ──────────────────────────────────────────────────────────
            debug_b64 = _debug_path_overlay(skeleton_arr, [], start_xy, end_xy, bg_img)
            # ── END DEBUG ──────────────────────────────────────────────────────
            return JSONResponse({
                "status": "failed",
                "path": [],
                "message": "No connected path found between start and end.",
                "debug_overlay_b64": debug_b64,  # ── DEBUG ──
            })

        # ── DEBUG ──────────────────────────────────────────────────────────────
        debug_b64 = _debug_path_overlay(skeleton_arr, ordered, start_xy, end_xy, bg_img)
        # ── END DEBUG ──────────────────────────────────────────────────────────
        return JSONResponse({
            "status": "success",
            "path": [[int(x), int(y)] for x, y in ordered],
            "debug_overlay_b64": debug_b64,  # ── DEBUG ──
        })
    except Exception as e:
        print(f"[error] extract_path: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto_extract_path")
async def auto_extract_path_endpoint(req: AutoExtractRequest):
    """
    Automatically determine start/end from skeleton topology and extract ordered path.

    Direction heuristic: vertical span ≥ horizontal span → bottom-to-top,
    otherwise left-to-right. Returns the same JSON format as /api/extract_path.
    """
    try:
        skeleton_arr = _decode_mask(req.skeleton_b64)
        result = auto_extract_ordered_path(skeleton_arr)

        if result is None:
            return JSONResponse({
                "status": "failed",
                "path": [],
                "message": "스켈레톤에서 경로를 찾을 수 없습니다.",
            })

        bg_img: Image.Image | None = None
        if req.input_img_b64:
            _, _data = req.input_img_b64.split(",", 1)
            bg_img = Image.open(io.BytesIO(base64.b64decode(_data))).convert("RGB")

        overlay_b64 = _debug_path_overlay(
            skeleton_arr,
            result["path"],
            tuple(result["start"]),
            tuple(result["end"]),
            bg_img,
        )

        return JSONResponse({
            "status": "success",
            "start": result["start"],
            "end":   result["end"],
            "path":  [[int(x), int(y)] for x, y in result["path"]],
            "overlay_b64": overlay_b64,
        })
    except Exception as e:
        print(f"[error] auto_extract_path: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
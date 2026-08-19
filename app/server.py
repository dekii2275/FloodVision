from __future__ import annotations

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Any, List, Optional
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import list_images, load_image, safe_stem
from src.flood_depth import (
    FloodVisionPipeline,
    PipelineConfig,
    VirtualGauge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FloodVisionServer")

app = FastAPI(
    title="FloodVision Production Service",
    description="Hệ thống giám sát và đo độ sâu ngập lụt kết hợp CLIPSeg + Virtual Gauges",
    version="2.0.0",
)

UPLOAD_DIR = ROOT / "outputs" / "web_runs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "outputs" / "web_runs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo Pipeline dùng chung (Singleton)
CONFIG_PATH = ROOT / "configs" / "camera_gauges_default.yaml"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
pipeline = FloodVisionPipeline.from_config(CONFIG_PATH, device=DEVICE)


class GaugeInput(BaseModel):
    id: str
    name: str
    base_x: float
    base_y: float
    top_x: float
    top_y: float
    max_height_cm: float = 80.0
    band_width_px: int = 15
    calibration_points: Optional[List[dict[str, float]]] = None


class AnalyzeRequest(BaseModel):
    image_path: str
    gauges: List[GaugeInput]
    threshold: float = 0.30
    roi_polygon: Optional[List[List[float]]] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


@app.get("/api/sample_images")
def get_sample_images():
    """Liệt kê danh sách các ảnh mẫu có sẵn trong thư mục data/."""
    sample_dirs = [
        ROOT / "data" / "ngapmuc1",
        ROOT / "data" / "ngapmuc2",
        ROOT / "data" / "ngapmuc3",
        ROOT / "data" / "ngapmuc4",
        ROOT / "data" / "khongngap",
        ROOT / "data" / "debug_cv_vlm" / "images",
    ]
    all_samples = []
    for s_dir in sample_dirs:
        if s_dir.exists():
            for p in list_images(s_dir):
                all_samples.append({
                    "name": p.name,
                    "rel_path": str(p.relative_to(ROOT)),
                    "category": s_dir.name,
                    "url": f"/api/image_raw?path={p.relative_to(ROOT)}",
                })
    return {"samples": all_samples}


@app.get("/api/image_raw")
def get_raw_image(path: str):
    """Phục vụ file ảnh tĩnh từ đường dẫn tương đối."""
    full_p = (ROOT / path).resolve()
    if not full_p.exists() or not str(full_p).startswith(str(ROOT)):
        raise HTTPException(status_code=404, detail="Không tìm thấy file ảnh")
    return FileResponse(str(full_p))


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload ảnh mới từ người dùng."""
    filename = f"upload_{int(time.time()*1000)}_{file.filename}"
    save_path = UPLOAD_DIR / filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = load_image(save_path)
    h, w = img.shape[:2]
    return {
        "image_path": str(save_path.relative_to(ROOT)),
        "url": f"/api/image_raw?path={save_path.relative_to(ROOT)}",
        "width": w,
        "height": h,
    }


@app.post("/api/analyze")
def analyze_flood_scene(req: AnalyzeRequest):
    """
    Thực hiện phân tích độ sâu ngập qua FloodVisionPipeline.
    """
    full_path = (ROOT / req.image_path).resolve()
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh: {req.image_path}")

    stem = safe_stem(full_path)

    # Chuyển đổi cấu hình Gauge đầu vào
    gauges_dicts = [
        {
            "id": g.id,
            "name": g.name,
            "base_normalized": [g.base_x, g.base_y],
            "top_normalized": [g.top_x, g.top_y],
            "band_width_px": g.band_width_px,
            "max_height_cm": g.max_height_cm,
            "calibration_points": g.calibration_points or [],
        }
        for g in req.gauges
    ]

    # Thực thi Pipeline
    output, img_bgr = pipeline.predict(
        image=full_path,
        gauges=gauges_dicts,
        roi_polygon=req.roi_polygon,
        threshold=req.threshold,
    )

    # Lưu ảnh Dashboard và Mask
    timestamp = int(time.time())
    out_dash_path = RESULTS_DIR / f"{stem}_{timestamp}_dashboard.png"
    out_mask_path = RESULTS_DIR / f"{stem}_{timestamp}_mask.png"

    output.save_dashboard(
        orig_bgr=img_bgr,
        output_path=out_dash_path,
        roi_polygon=req.roi_polygon or pipeline.config.roi_polygon_normalized,
    )
    cv2.imwrite(str(out_mask_path), output.flood_mask * 255)

    res_dict = output.to_dict()
    res_dict.update({
        "success": True,
        "image_path": req.image_path,
        "dashboard_url": f"/api/image_raw?path={out_dash_path.relative_to(ROOT)}",
        "mask_url": f"/api/image_raw?path={out_mask_path.relative_to(ROOT)}",
    })

    return res_dict


# Mount Static Files (HTML/CSS/JS)
STATIC_DIR = ROOT / "app" / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)

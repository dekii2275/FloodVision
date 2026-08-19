#!/usr/bin/env python3
"""
FloodVision - Baseline Giám Sát Độ Sâu Ngập Lụt (Water Gauge Depth Estimation Baseline)

Pipeline:
  IMAGE + G1/G2/G3 Virtual Gauges
        │
        ▼
  CLIPSeg (Parallel Water-Centric Multi-Prompt Ensemble)
        │
        ▼
  Water Probability Map
        │
        ▼
  CAMERA ROI MASK  ◄──── camera_gauges_default.yaml
        │
        ▼
  Threshold (>= 0.30)
        │
        ▼
  Morphology & Connected Component Filtering (ROI Ratio Based)
        │
        ▼
  Clean Flood Mask + Base-Connected Component Identification
        │
        ▼
  Gauge Band Sampling (15 Parallel Rays across band_width_px)
        │
        ▼
  Base-Connected Waterline Scan (Scan from base upwards, stop at stable transition)
        │
        ▼
  Multi-Point Piecewise Calibration (depth_cm per gauge)
        │
        ▼
  Robust Median Fusion & Outlier Rejection (|d - median| > 25cm -> OUTLIER)
        │
        ▼
  Final Fused Depth & L0-L4 Alert Level
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import list_images, load_image, safe_stem
from src.flood_depth import (
    VirtualGauge,
    GaugeMeasurement,
    create_roi_mask,
    filter_connected_components,
    measure_gauge,
    fuse_gauge_measurements,
    make_depth_dashboard_visualization,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("water_gauge_depth")

OPTIMAL_WATER_PROMPTS = [
    "flood water",
    "water",
    "muddy water",
    "standing flood water",
    "puddle of water",
    "water surface",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FloodVision - Water Surface Segmentation & Virtual Gauge Depth Estimation Baseline"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Đường dẫn tới 1 ảnh bất kỳ")
    source.add_argument("--input-dir", "--input", type=Path, help="Thư mục chứa danh sách ảnh cần chạy")
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("configs/camera_gauges_default.yaml"),
        help="File cấu hình ROI và Virtual Gauges",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/water_gauge_depth"),
        help="Thư mục lưu kết quả",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Ngưỡng xác suất mặt nước (mặc định: 0.30)",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="max",
        choices=["max", "mean", "top2_mean"],
        help="Chế độ Prompt Ensemble (mặc định: max)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán: cuda hoặc cpu",
    )
    parser.add_argument("--limit", type=int, help="Giới hạn số lượng ảnh chạy")
    return parser.parse_args()


class ClipSegWaterEngine:
    """Đóng gói mô hình CLIPSeg với Multi-Prompt Ensemble."""

    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined", device: str = "cuda"):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = device
        logger.info("Đang nạp mô hình CLIPSeg (%s) trên thiết bị %s...", model_name, device.upper())
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_water_probability(
        self,
        img_pil: Image.Image,
        prompts: list[str] | None = None,
        ensemble_mode: str = "max",
    ) -> np.ndarray:
        w, h = img_pil.size
        target_prompts = prompts or OPTIMAL_WATER_PROMPTS

        inputs = self.processor(
            text=target_prompts,
            images=[img_pil] * len(target_prompts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.sigmoid(outputs.logits)

        if ensemble_mode == "mean":
            ens_prob = torch.mean(probs, dim=0)
        elif ensemble_mode == "top2_mean" and probs.shape[0] >= 2:
            top2_vals = torch.topk(probs, k=2, dim=0).values
            ens_prob = torch.mean(top2_vals, dim=0)
        else:  # max
            ens_prob = torch.max(probs, dim=0).values

        prob_resized = torch.nn.functional.interpolate(
            ens_prob.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return prob_resized


def process_image(
    image_path: Path,
    engine: ClipSegWaterEngine,
    config: dict[str, Any],
    output_dir: Path,
    threshold: float = 0.30,
    ensemble_mode: str = "max",
) -> dict[str, Any]:
    started = time.perf_counter()
    stem = safe_stem(image_path)
    img_bgr = load_image(image_path)
    height, width = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 1. Trích xuất Water Probability Map
    prob_map = engine.predict_water_probability(img_pil, ensemble_mode=ensemble_mode)

    # 2. Tạo Camera ROI Mask
    roi_poly_norm = config.get("roi_polygon_normalized", [])
    roi_mask = create_roi_mask((height, width), roi_poly_norm)

    # 3. Cắt ngưỡng & Lọc Connected Components
    prob_map_roi = prob_map * roi_mask
    raw_binary = (prob_map_roi >= threshold).astype(np.uint8)

    post_cfg = config.get("postprocess", {})
    min_ratio = float(post_cfg.get("min_component_ratio", 0.001))
    min_px = int(post_cfg.get("min_component_px", 50))
    morph_k = int(post_cfg.get("close_kernel", 5))

    final_flood_mask, labels_map, stats = filter_connected_components(
        binary_mask=raw_binary,
        roi_mask=roi_mask,
        min_component_ratio=min_ratio,
        min_component_px=min_px,
        morph_kernel_size=morph_k,
    )

    # 4. Tính % diện tích ngập
    roi_area_px = float(np.sum(roi_mask == 1))
    water_area_px = float(np.sum(final_flood_mask == 1))
    flood_area_pct = (water_area_px / roi_area_px * 100.0) if roi_area_px > 0 else 0.0

    # 5. Đo đạc các Cột mốc ảo với Gauge Band & Base-Connected Waterline Scan
    gauges_cfg = config.get("gauges", [])
    meas_cfg = config.get("measurement", {})

    measurements: list[GaugeMeasurement] = []
    for g_dict in gauges_cfg:
        vg = VirtualGauge.from_dict(g_dict, width, height)
        meas = measure_gauge(
            gauge=vg,
            flood_mask=final_flood_mask,
            prob_map=prob_map,
            labels_map=labels_map,
            config=meas_cfg,
        )
        measurements.append(meas)

    # 6. Hợp nhất kết quả đo (Robust Median Fusion & Outlier Rejection)
    fusion_cfg = config.get("fusion", {})
    fusion_res = fuse_gauge_measurements(
        measurements=measurements,
        flood_area_pct=flood_area_pct,
        config=fusion_cfg,
    )

    latency_ms = (time.perf_counter() - started) * 1000.0

    # 7. Tạo Dashboard Trực quan hóa
    output_dir.mkdir(parents=True, exist_ok=True)
    out_vis_path = output_dir / f"{stem}_depth_dashboard.png"
    make_depth_dashboard_visualization(
        orig_bgr=img_bgr,
        prob_map=prob_map,
        final_water_mask=final_flood_mask,
        roi_mask=roi_mask,
        roi_polygon_normalized=roi_poly_norm,
        measurements=measurements,
        fusion_result=fusion_res,
        flood_area_pct=flood_area_pct,
        output_path=str(out_vis_path),
    )

    # Lưu mask nhị phân
    mask_path = output_dir / f"{stem}_flood_mask.png"
    cv2.imwrite(str(mask_path), final_flood_mask * 255)

    # 8. Đóng gói kết quả JSON
    gauge_details = {
        m.gauge_id: {
            "name": m.gauge_name,
            "status": m.status.value,
            "is_submerged": m.is_submerged,
            "depth_cm": m.local_depth_cm,
            "waterline_t": m.waterline_t,
            "valid_rays": m.valid_rays,
            "total_rays": m.total_rays,
            "calibration_mode": m.calibration_mode,
            "warning": m.warning,
            "intersection_pt": list(m.intersection_pt) if m.intersection_pt else None,
            "confidence_metrics": m.confidence_metrics,
        }
        for m in measurements
    }

    result = {
        "image_path": str(image_path),
        "image_id": image_path.name,
        "width": width,
        "height": height,
        "flood_area_pct": round(flood_area_pct, 2),
        "fused_depth_cm": fusion_res.fused_depth_cm,
        "max_depth_cm": fusion_res.max_depth_cm,
        "mean_depth_cm": fusion_res.mean_depth_cm,
        "fusion": fusion_res.to_dict(),
        "level": fusion_res.severity.to_dict(),
        "gauges": gauge_details,
        "latency_ms": round(latency_ms, 2),
        "dashboard_visualization": str(out_vis_path),
        "binary_mask": str(mask_path),
    }

    return result


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.camera_config.exists():
        logger.error("Không tìm thấy file cấu hình camera: %s", args.camera_config)
        return 1
    with open(args.camera_config, "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    engine = ClipSegWaterEngine(device=args.device)

    if args.image:
        if not args.image.exists():
            logger.error("Không tìm thấy ảnh: %s", args.image)
            return 1
        image_paths = [args.image]
    else:
        if not args.input_dir.exists():
            logger.error("Không tìm thấy thư mục: %s", args.input_dir)
            return 1
        image_paths = list_images(args.input_dir)
        if not image_paths:
            logger.error("Không tìm thấy ảnh nào trong: %s", args.input_dir)
            return 1

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    logger.info("Bắt đầu xử lý %d ảnh. Thư mục xuất: %s", len(image_paths), out_dir)

    all_results = []
    flat_rows = []

    for idx, img_p in enumerate(image_paths):
        logger.info("[%d/%d] Đang xử lý: %s", idx + 1, len(image_paths), img_p.name)
        res = process_image(
            image_path=img_p,
            engine=engine,
            config=camera_config,
            output_dir=out_dir,
            threshold=args.threshold,
            ensemble_mode=args.ensemble,
        )
        all_results.append(res)

        fused_d = res['fused_depth_cm'] if res['fused_depth_cm'] is not None else 0.0
        print("\n" + "=" * 60)
        print(f"📊 KẾT QUẢ ĐO ĐỘ SÂU NGẬP: {img_p.name}")
        print("=" * 60)
        print(f"🌊 Diện tích ngập trong lòng đường: {res['flood_area_pct']:.2f}%")
        print(f"📏 Độ sâu ước tính (Fused Depth)   : {fused_d:.1f} cm")
        print(f"🚨 Cấp độ cảnh báo (Alert Status) : [{res['level']['level_code']}] {res['level']['level_name']}")
        print(f"🔄 Hợp nhất: Dùng {res['fusion']['used_gauges']} | Loại bỏ: {res['fusion']['rejected_gauges']}")
        for g_id, g_val in res["gauges"].items():
            sub_txt = f"{g_val['depth_cm']:.1f} cm ({g_val['status']})" if g_val["status"] == "VALID" else f"0.0 cm ({g_val['status']})"
            print(f"   • {g_val['name']:<30}: {sub_txt}")
        print("=" * 60 + "\n")

        flat_rows.append({
            "image_id": res["image_id"],
            "flood_area_pct": res["flood_area_pct"],
            "fused_depth_cm": res["fused_depth_cm"],
            "max_depth_cm": res["max_depth_cm"],
            "flood_level": res["level"]["level_code"],
            "alert_status": res["level"]["level_name"],
            "used_gauges": ",".join(res["fusion"]["used_gauges"]),
            "rejected_gauges": ",".join(res["fusion"]["rejected_gauges"]),
            "latency_ms": res["latency_ms"],
        })

    with (out_dir / "summary_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if flat_rows:
        csv_fields = list(flat_rows[0].keys())
        with (out_dir / "summary_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(flat_rows)

    logger.info("Hoàn tất! Kết quả và Dashboard đã lưu tại: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

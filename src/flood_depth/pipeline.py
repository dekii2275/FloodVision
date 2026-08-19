from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from src.common.io import load_image
from src.flood_depth.calibration import CalibrationPoint
from src.flood_depth.gauge_geometry import VirtualGauge
from src.flood_depth.gauge_sampling import GaugeMeasurement, measure_gauge
from src.flood_depth.gauge_validation import GaugeStatus
from src.flood_depth.fusion import FusionResult, SeverityLevel, fuse_gauge_measurements
from src.flood_depth.roi_filter import create_roi_mask, filter_connected_components
from src.flood_depth.visualizer import make_depth_dashboard_visualization

logger = logging.getLogger("FloodVisionPipeline")

DEFAULT_WATER_PROMPTS = [
    "flood water",
    "water",
    "muddy water",
    "standing flood water",
    "puddle of water",
    "water surface",
]


@dataclass
class PipelineConfig:
    camera_id: str = "DEFAULT_CAMERA"
    roi_polygon_normalized: List[List[float]] = field(
        default_factory=lambda: [[0.02, 0.25], [0.98, 0.25], [1.00, 1.00], [0.00, 1.00]]
    )
    gauges: List[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.30
    ensemble_mode: str = "max"  # 'max', 'mean', 'top2_mean'
    min_component_ratio: float = 0.001
    min_component_px: int = 50
    close_kernel: int = 5
    open_kernel: int = 5
    gauge_band_width_px: int = 15
    base_search_radius_px: int = 8
    stable_transition_samples: int = 3
    min_valid_ray_ratio: float = 0.40
    num_rays: int = 15
    fusion_method: str = "median"
    outlier_abs_cm: float = 25.0

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PipelineConfig:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        seg_cfg = data.get("segmentation", {})
        post_cfg = data.get("postprocess", {})
        meas_cfg = data.get("measurement", {})
        fuse_cfg = data.get("fusion", {})

        return cls(
            camera_id=str(data.get("camera_id", "DEFAULT_CAMERA")),
            roi_polygon_normalized=data.get(
                "roi_polygon_normalized",
                [[0.02, 0.25], [0.98, 0.25], [1.00, 1.00], [0.00, 1.00]],
            ),
            gauges=data.get("gauges", []),
            threshold=float(seg_cfg.get("threshold", data.get("water_threshold", 0.30))),
            ensemble_mode=str(seg_cfg.get("ensemble", "max")),
            min_component_ratio=float(post_cfg.get("min_component_ratio", 0.001)),
            min_component_px=int(post_cfg.get("min_component_px", 50)),
            close_kernel=int(post_cfg.get("close_kernel", 5)),
            open_kernel=int(post_cfg.get("open_kernel", 5)),
            gauge_band_width_px=int(meas_cfg.get("gauge_band_width_px", 15)),
            base_search_radius_px=int(meas_cfg.get("base_search_radius_px", 8)),
            stable_transition_samples=int(meas_cfg.get("stable_transition_samples", 3)),
            min_valid_ray_ratio=float(meas_cfg.get("min_valid_ray_ratio", 0.40)),
            num_rays=int(meas_cfg.get("num_rays", 15)),
            fusion_method=str(fuse_cfg.get("method", "median")),
            outlier_abs_cm=float(fuse_cfg.get("outlier_abs_cm", 25.0)),
        )


@dataclass
class PipelineOutput:
    width: int
    height: int
    flood_area_pct: float
    fused_depth_cm: float | None
    max_depth_cm: float
    mean_depth_cm: float
    level: SeverityLevel
    fusion: FusionResult
    measurements: List[GaugeMeasurement]
    prob_map: np.ndarray
    flood_mask: np.ndarray
    roi_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "flood_area_pct": round(self.flood_area_pct, 2),
            "fused_depth_cm": round(self.fused_depth_cm, 1) if self.fused_depth_cm is not None else 0.0,
            "max_depth_cm": round(self.max_depth_cm, 1),
            "mean_depth_cm": round(self.mean_depth_cm, 1),
            "level": self.level.to_dict(),
            "fusion": self.fusion.to_dict(),
            "gauges": [
                {
                    "id": m.gauge_id,
                    "name": m.gauge_name,
                    "status": m.status.value,
                    "is_submerged": m.is_submerged,
                    "depth_cm": m.local_depth_cm,
                    "max_height_cm": m.max_height_cm,
                    "submerged_px": m.submerged_px,
                    "waterline_t": m.waterline_t,
                    "intersection_pt": list(m.intersection_pt) if m.intersection_pt else None,
                    "valid_rays": m.valid_rays,
                    "total_rays": m.total_rays,
                    "calibration_mode": m.calibration_mode,
                    "warning": m.warning,
                    "confidence_metrics": m.confidence_metrics,
                }
                for m in self.measurements
            ],
        }

    def save_dashboard(self, orig_bgr: np.ndarray, output_path: Union[str, Path], roi_polygon: Sequence[Sequence[float]] | None = None) -> None:
        make_depth_dashboard_visualization(
            orig_bgr=orig_bgr,
            prob_map=self.prob_map,
            final_water_mask=self.flood_mask,
            roi_mask=self.roi_mask,
            roi_polygon_normalized=roi_polygon,
            measurements=self.measurements,
            fusion_result=self.fusion,
            flood_area_pct=self.flood_area_pct,
            output_path=str(output_path),
        )


class FloodVisionPipeline:
    """
    Pipeline hợp nhất chuẩn hóa cho bài toán Phân đoạn & Ước lượng độ sâu ngập (CLIPSeg + Virtual Gauges).
    Dễ dàng tích hợp vào Server Web, CLI hoặc các luồng Camera RTSP.
    """

    def __init__(
        self,
        config: PipelineConfig | dict[str, Any] | str | Path | None = None,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        device: str | None = None,
        prompts: Sequence[str] | None = None,
    ):
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = PipelineConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = PipelineConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts = list(prompts or DEFAULT_WATER_PROMPTS)

        logger.info("Khởi tạo FloodVisionPipeline (Model: %s, Device: %s)", model_id, self.device.upper())
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

    @classmethod
    def from_config(cls, config_path: Union[str, Path], device: str | None = None) -> FloodVisionPipeline:
        return cls(config=config_path, device=device)

    @torch.inference_mode()
    def segment_water(
        self,
        image_pil: Image.Image,
        ensemble_mode: str | None = None,
    ) -> np.ndarray:
        w, h = image_pil.size
        mode = ensemble_mode or self.config.ensemble_mode

        inputs = self.processor(
            text=self.prompts,
            images=[image_pil] * len(self.prompts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.sigmoid(outputs.logits)

        if mode == "mean":
            ens_prob = torch.mean(probs, dim=0)
        elif mode == "top2_mean" and probs.shape[0] >= 2:
            top2 = torch.topk(probs, k=2, dim=0).values
            ens_prob = torch.mean(top2, dim=0)
        else:
            ens_prob = torch.max(probs, dim=0).values

        prob_resized = torch.nn.functional.interpolate(
            ens_prob.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return prob_resized

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        gauges: Sequence[VirtualGauge] | Sequence[dict[str, Any]] | None = None,
        roi_polygon: Sequence[Sequence[float]] | None = None,
        threshold: float | None = None,
    ) -> Tuple[PipelineOutput, np.ndarray]:
        """
        Dự đoán toàn diện độ sâu và cấp độ ngập trên 1 ảnh.
        
        Returns:
            (pipeline_output, original_bgr_image)
        """
        # 1. Tiền xử lý ảnh đầu vào
        if isinstance(image, (str, Path)):
            img_bgr = load_image(image)
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            img_pil = image.convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image.copy()
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Không hỗ trợ định dạng ảnh: {type(image)}")

        height, width = img_bgr.shape[:2]
        thresh = threshold if threshold is not None else self.config.threshold

        # 2. Phân đoạn mặt nước (Water Probability Map)
        prob_map = self.segment_water(img_pil)

        # 3. Tạo Camera ROI Mask
        target_roi = roi_polygon if roi_polygon is not None else self.config.roi_polygon_normalized
        roi_mask = create_roi_mask((height, width), target_roi)

        # 4. Cắt ngưỡng và lọc Connected Components
        prob_map_roi = prob_map * roi_mask
        raw_binary = (prob_map_roi >= thresh).astype(np.uint8)

        final_flood_mask, labels_map, stats = filter_connected_components(
            binary_mask=raw_binary,
            roi_mask=roi_mask,
            min_component_ratio=self.config.min_component_ratio,
            min_component_px=self.config.min_component_px,
            morph_kernel_size=self.config.close_kernel,
        )

        # 5. Tính % diện tích ngập
        roi_area_px = float(np.sum(roi_mask == 1))
        water_area_px = float(np.sum(final_flood_mask == 1))
        flood_area_pct = (water_area_px / roi_area_px * 100.0) if roi_area_px > 0 else 0.0

        # 6. Đo đạc các Cột mốc ảo (Virtual Gauges)
        if gauges is not None:
            active_gauges = []
            for g in gauges:
                if isinstance(g, VirtualGauge):
                    active_gauges.append(g)
                elif isinstance(g, dict):
                    active_gauges.append(VirtualGauge.from_dict(g, width, height))
        else:
            active_gauges = [
                VirtualGauge.from_dict(g_dict, width, height)
                for g_dict in self.config.gauges
            ]

        meas_cfg = {
            "num_rays": self.config.num_rays,
            "base_search_radius_px": self.config.base_search_radius_px,
            "stable_transition_samples": self.config.stable_transition_samples,
            "min_valid_ray_ratio": self.config.min_valid_ray_ratio,
        }

        measurements: List[GaugeMeasurement] = [
            measure_gauge(
                gauge=vg,
                flood_mask=final_flood_mask,
                prob_map=prob_map,
                labels_map=labels_map,
                config=meas_cfg,
            )
            for vg in active_gauges
        ]

        # 7. Hợp nhất kết quả qua Robust Median Fusion & Outlier Rejection
        fusion_cfg = {
            "method": self.config.fusion_method,
            "outlier_abs_cm": self.config.outlier_abs_cm,
        }
        fusion_res = fuse_gauge_measurements(
            measurements=measurements,
            flood_area_pct=flood_area_pct,
            config=fusion_cfg,
        )

        output = PipelineOutput(
            width=width,
            height=height,
            flood_area_pct=flood_area_pct,
            fused_depth_cm=fusion_res.fused_depth_cm,
            max_depth_cm=fusion_res.max_depth_cm,
            mean_depth_cm=fusion_res.mean_depth_cm,
            level=fusion_res.severity,
            fusion=fusion_res,
            measurements=measurements,
            prob_map=prob_map,
            flood_mask=final_flood_mask,
            roi_mask=roi_mask,
        )

        return output, img_bgr

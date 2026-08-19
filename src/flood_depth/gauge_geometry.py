from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple
import numpy as np

from src.flood_depth.calibration import CalibrationPoint, validate_calibration_points


@dataclass
class VirtualGauge:
    id: str
    name: str
    base_pt: Tuple[int, int]                  # (xb, yb) tại chân cột mốc (t = 0.0)
    top_pt: Tuple[int, int]                   # (xt, yt) tại đỉnh cột mốc (t = 1.0)
    band_width_px: int = 15                   # Bề rộng dải đo (pixel)
    max_height_cm: float = 100.0              # Chiều cao đo tối đa fallback (cm)
    calibration_points: List[CalibrationPoint] = field(default_factory=list)
    fusion_group: str = "default"             # Nhóm gộp phân đoạn đường

    @property
    def vector(self) -> Tuple[float, float]:
        return float(self.top_pt[0] - self.base_pt[0]), float(self.top_pt[1] - self.base_pt[1])

    @property
    def pixel_length(self) -> float:
        vx, vy = self.vector
        return max(1.0, math.hypot(vx, vy))

    @property
    def pixel_height(self) -> float:
        # Fallback chiều cao theo trục Y
        return float(max(1.0, abs(self.base_pt[1] - self.top_pt[1])))

    @property
    def unit_vector(self) -> Tuple[float, float]:
        l = self.pixel_length
        vx, vy = self.vector
        return vx / l, vy / l

    @property
    def normal_vector(self) -> Tuple[float, float]:
        # Vector pháp tuyến vuông góc với trục thân thước
        ux, uy = self.unit_vector
        return -uy, ux

    def point_at_t(self, t: float) -> Tuple[int, int]:
        """Tọa độ pixel tại tham số t dọc theo thân mốc (0.0 <= t <= 1.0)."""
        bx, by = self.base_pt
        vx, vy = self.vector
        px = int(round(bx + t * vx))
        py = int(round(by + t * vy))
        return px, py

    def get_band_rays(self, num_rays: int = 15) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        Sinh danh sách các tia song song trên dải đo (Gauge Band).
        Returns: list of (base_k, top_k, offset_px)
        """
        num_rays = max(1, int(num_rays))
        if num_rays == 1:
            return [(self.base_pt, self.top_pt, 0.0)]

        half_w = self.band_width_px / 2.0
        offsets = np.linspace(-half_w, half_w, num_rays)
        nx, ny = self.normal_vector

        rays = []
        bx, by = self.base_pt
        tx, ty = self.top_pt

        for off in offsets:
            ray_base = (int(round(bx + off * nx)), int(round(by + off * ny)))
            ray_top = (int(round(tx + off * nx)), int(round(ty + off * ny)))
            rays.append((ray_base, ray_top, float(off)))

        return rays

    def validate_geometry(self, image_shape: Tuple[int, int], min_length_px: float = 10.0) -> Tuple[bool, str | None]:
        h, w = image_shape[:2]
        bx, by = self.base_pt
        tx, ty = self.top_pt

        if not (0 <= bx < w and 0 <= by < h):
            return False, f"Gauge {self.id} base ({bx}, {by}) is outside image bounds ({w}x{h})"
        if not (0 <= tx < w and 0 <= ty < h):
            return False, f"Gauge {self.id} top ({tx}, {ty}) is outside image bounds ({w}x{h})"
        if self.pixel_length < min_length_px:
            return False, f"Gauge {self.id} length ({self.pixel_length:.1f}px) is shorter than minimum {min_length_px}px"

        if self.calibration_points and not validate_calibration_points(self.calibration_points):
            return False, f"Gauge {self.id} has invalid non-monotonic calibration points"

        return True, None

    @classmethod
    def from_dict(
        cls,
        gauge_dict: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> VirtualGauge:
        base_norm = gauge_dict.get("base_normalized")
        if base_norm is None:
            # Fallback format base_x, base_y
            base_norm = [gauge_dict.get("base_x", 0.5), gauge_dict.get("base_y", 0.90)]
        top_norm = gauge_dict.get("top_normalized")
        if top_norm is None:
            top_norm = [gauge_dict.get("top_x", 0.5), gauge_dict.get("top_y", 0.35)]

        base_pt = (
            int(np.clip(base_norm[0] * image_width, 0, image_width - 1)),
            int(np.clip(base_norm[1] * image_height, 0, image_height - 1)),
        )
        top_pt = (
            int(np.clip(top_norm[0] * image_width, 0, image_width - 1)),
            int(np.clip(top_norm[1] * image_height, 0, image_height - 1)),
        )

        calib_raw = gauge_dict.get("calibration_points", [])
        calib_pts = [CalibrationPoint.from_dict(cp) for cp in calib_raw] if calib_raw else []

        return cls(
            id=str(gauge_dict.get("id", "G1")),
            name=str(gauge_dict.get("name", "Gauge")),
            base_pt=base_pt,
            top_pt=top_pt,
            band_width_px=int(gauge_dict.get("band_width_px", 15)),
            max_height_cm=float(gauge_dict.get("max_height_cm", 100.0)),
            calibration_points=calib_pts,
            fusion_group=str(gauge_dict.get("fusion_group", "default")),
        )

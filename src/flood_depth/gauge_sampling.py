from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple
import numpy as np

from src.flood_depth.calibration import piecewise_linear_interpolate
from src.flood_depth.gauge_geometry import VirtualGauge
from src.flood_depth.gauge_validation import GaugeStatus
from src.flood_depth.roi_filter import find_component_near_gauge_base


@dataclass
class RayWaterlineResult:
    ray_index: int
    offset_px: float
    waterline_t: float
    is_valid: bool
    intersection_pt: Tuple[int, int]
    status: GaugeStatus


@dataclass
class GaugeMeasurement:
    gauge_id: str
    gauge_name: str
    status: GaugeStatus
    local_depth_cm: float
    fused_into_group: bool = True
    calibration_mode: str = "legacy_linear"
    warning: str | None = None
    waterline_t: float = 0.0
    intersection_pt: Tuple[int, int] | None = None
    valid_rays: int = 0
    total_rays: int = 15
    spread_t: float = 0.0
    confidence_metrics: dict[str, float] = field(default_factory=dict)
    ray_results: List[RayWaterlineResult] = field(default_factory=list)
    gauge_geometry: VirtualGauge | None = None

    # Backward compatibility properties
    @property
    def depth_cm(self) -> float:
        return self.local_depth_cm

    @property
    def is_submerged(self) -> bool:
        return self.status == GaugeStatus.VALID and self.local_depth_cm > 0.5

    @property
    def base_pt(self) -> Tuple[int, int]:
        return self.gauge_geometry.base_pt if self.gauge_geometry else (0, 0)

    @property
    def top_pt(self) -> Tuple[int, int]:
        return self.gauge_geometry.top_pt if self.gauge_geometry else (0, 0)

    @property
    def max_height_cm(self) -> float:
        return self.gauge_geometry.max_height_cm if self.gauge_geometry else 100.0

    @property
    def submerged_px(self) -> float:
        if self.gauge_geometry:
            return round(self.waterline_t * self.gauge_geometry.pixel_height, 1)
        return 0.0


def sample_ray_line(
    image_mat: np.ndarray,
    p_start: Tuple[int, int],
    p_end: Tuple[int, int],
    num_samples: int,
) -> np.ndarray:
    """Lấy mẫu các giá trị pixel dọc theo đoạn thẳng từ p_start đến p_end."""
    h, w = image_mat.shape[:2]
    xs = np.linspace(p_start[0], p_end[0], num_samples)
    ys = np.linspace(p_start[1], p_end[1], num_samples)

    xs_int = np.clip(np.round(xs).astype(int), 0, w - 1)
    ys_int = np.clip(np.round(ys).astype(int), 0, h - 1)

    return image_mat[ys_int, xs_int]


def find_base_connected_waterline(
    ray_binary: np.ndarray,
    ray_labels: np.ndarray,
    target_component_id: int,
    stable_transition_samples: int = 3,
) -> Tuple[float, GaugeStatus, int | None]:
    """
    Quét từ chân mốc (base, index 0) đi lên (top, index N-1).
    Chỉ công nhận vùng nước liên tục kết nối với chân mốc.
    Dừng lại khi gặp chuyển tiếp ổn định (stable transition: water -> non-water).
    
    Returns:
        (waterline_t, status, transition_index)
    """
    n_samples = len(ray_binary)
    if n_samples < 2:
        return 0.0, GaugeStatus.INVALID_GEOMETRY, None

    # 1. Kiểm tra chân mốc: nếu chân mốc (3 điểm đầu) không thuộc component ngập nước
    base_window_labels = ray_labels[: min(4, n_samples)]
    base_window_bin = ray_binary[: min(4, n_samples)]

    is_base_in_component = bool(
        np.any(base_window_labels == target_component_id) or (np.mean(base_window_bin) >= 0.5)
    )

    if not is_base_in_component:
        return 0.0, GaugeStatus.NO_WATER_AT_BASE, None

    # 2. Quét liên tục từ chân mốc đi lên
    # Tìm điểm chuyển tiếp ổn định (K mẫu liên tiếp không phải nước)
    non_water_run = 0
    last_water_idx = 0

    for i in range(n_samples):
        in_water = bool(ray_labels[i] == target_component_id or ray_binary[i] == 1)
        if in_water:
            last_water_idx = i
            non_water_run = 0
        else:
            non_water_run += 1
            if non_water_run >= stable_transition_samples:
                # Đã xác nhận chuyển tiếp ra khỏi vùng nước
                break

    waterline_t = float(last_water_idx / (n_samples - 1))
    return waterline_t, GaugeStatus.VALID, last_water_idx


def measure_gauge(
    gauge: VirtualGauge,
    flood_mask: np.ndarray,
    prob_map: np.ndarray,
    labels_map: np.ndarray,
    config: dict[str, Any] | None = None,
) -> GaugeMeasurement:
    """
    Đo đạc độ sâu ngập trên dải đo của một cột mốc (Gauge Band Measurement).
    """
    cfg = config or {}
    h, w = flood_mask.shape[:2]

    # 1. Kiểm tra hình học cột mốc
    valid_geom, geom_err = gauge.validate_geometry((h, w))
    if not valid_geom:
        return GaugeMeasurement(
            gauge_id=gauge.id,
            gauge_name=gauge.name,
            status=GaugeStatus.INVALID_GEOMETRY,
            local_depth_cm=0.0,
            fused_into_group=False,
            warning=geom_err,
            gauge_geometry=gauge,
        )

    # 2. Xác định Connected Component ngập nước tại chân cột mốc (Base)
    search_radius = int(cfg.get("base_search_radius_px", 8))
    target_comp = find_component_near_gauge_base(labels_map, gauge.base_pt, search_radius_px=search_radius)

    if target_comp is None:
        # Chân cột mốc hoàn toàn khô ráo
        return GaugeMeasurement(
            gauge_id=gauge.id,
            gauge_name=gauge.name,
            status=GaugeStatus.NO_WATER_AT_BASE,
            local_depth_cm=0.0,
            fused_into_group=True,
            waterline_t=0.0,
            intersection_pt=None,
            valid_rays=0,
            total_rays=int(cfg.get("num_rays", 15)),
            gauge_geometry=gauge,
        )

    # 3. Lấy mẫu đa tia song song trên dải đo (Gauge Band)
    num_rays = int(cfg.get("num_rays", 15))
    stable_samples = int(cfg.get("stable_transition_samples", 3))
    min_valid_ratio = float(cfg.get("min_valid_ray_ratio", 0.40))
    n_sample_pts = max(30, int(round(gauge.pixel_length)))

    band_rays = gauge.get_band_rays(num_rays=num_rays)
    ray_results: List[RayWaterlineResult] = []
    valid_t_list: List[float] = []

    for ray_idx, (r_base, r_top, off_px) in enumerate(band_rays):
        ray_bin = sample_ray_line(flood_mask, r_base, r_top, n_sample_pts)
        ray_lbl = sample_ray_line(labels_map, r_base, r_top, n_sample_pts)

        r_t, r_status, r_idx = find_base_connected_waterline(
            ray_binary=ray_bin,
            ray_labels=ray_lbl,
            target_component_id=target_comp,
            stable_transition_samples=stable_samples,
        )

        rx = int(round(r_base[0] + r_t * (r_top[0] - r_base[0])))
        ry = int(round(r_base[1] + r_t * (r_top[1] - r_base[1])))

        is_v = bool(r_status == GaugeStatus.VALID)
        if is_v:
            valid_t_list.append(r_t)

        ray_results.append(
            RayWaterlineResult(
                ray_index=ray_idx,
                offset_px=off_px,
                waterline_t=round(r_t, 4),
                is_valid=is_v,
                intersection_pt=(rx, ry),
                status=r_status,
            )
        )

    valid_count = len(valid_t_list)
    valid_ratio = valid_count / float(num_rays)

    # 4. Đánh giá tính hợp lệ của dải tia
    if valid_count == 0:
        return GaugeMeasurement(
            gauge_id=gauge.id,
            gauge_name=gauge.name,
            status=GaugeStatus.NO_WATER_AT_BASE,
            local_depth_cm=0.0,
            fused_into_group=True,
            waterline_t=0.0,
            valid_rays=0,
            total_rays=num_rays,
            ray_results=ray_results,
            gauge_geometry=gauge,
        )

    if valid_ratio < min_valid_ratio:
        median_t = float(np.median(valid_t_list))
        depth_cm, calib_mode, calib_warn = piecewise_linear_interpolate(
            median_t, gauge.calibration_points, gauge.max_height_cm
        )
        return GaugeMeasurement(
            gauge_id=gauge.id,
            gauge_name=gauge.name,
            status=GaugeStatus.INSUFFICIENT_VALID_RAYS,
            local_depth_cm=depth_cm,
            fused_into_group=False,
            calibration_mode=calib_mode,
            warning=f"Only {valid_count}/{num_rays} valid rays (min ratio: {min_valid_ratio})",
            waterline_t=round(median_t, 4),
            intersection_pt=gauge.point_at_t(median_t),
            valid_rays=valid_count,
            total_rays=num_rays,
            ray_results=ray_results,
            gauge_geometry=gauge,
        )

    # 5. Tính Median Waterline và Spread
    median_t = float(np.median(valid_t_list))
    spread_t = float(np.std(valid_t_list)) if len(valid_t_list) > 1 else 0.0
    waterline_std_px = spread_t * gauge.pixel_length

    # 6. Quy đổi độ sâu centimet qua Piecewise Linear Calibration
    depth_cm, calib_mode, calib_warn = piecewise_linear_interpolate(
        median_t, gauge.calibration_points, gauge.max_height_cm
    )

    final_status = GaugeStatus.VALID
    if calib_warn == "OUT_OF_CALIBRATION_RANGE":
        final_status = GaugeStatus.OUT_OF_CALIBRATION_RANGE

    # 7. Tính các chỉ số tự tin cậy (Confidence Metrics)
    base_probs = sample_ray_line(prob_map, gauge.base_pt, gauge.point_at_t(0.10), 10)
    trans_probs = sample_ray_line(
        prob_map,
        gauge.point_at_t(max(0.0, median_t - 0.05)),
        gauge.point_at_t(min(1.0, median_t + 0.05)),
        10,
    )

    conf_metrics = {
        "valid_ray_ratio": round(valid_ratio, 3),
        "waterline_std_px": round(waterline_std_px, 2),
        "mean_base_probability": round(float(np.mean(base_probs)), 3) if base_probs.size > 0 else 0.0,
        "mean_transition_probability": round(float(np.mean(trans_probs)), 3) if trans_probs.size > 0 else 0.0,
    }

    return GaugeMeasurement(
        gauge_id=gauge.id,
        gauge_name=gauge.name,
        status=final_status,
        local_depth_cm=depth_cm,
        fused_into_group=True,
        calibration_mode=calib_mode,
        warning=calib_warn,
        waterline_t=round(median_t, 4),
        intersection_pt=gauge.point_at_t(median_t),
        valid_rays=valid_count,
        total_rays=num_rays,
        spread_t=round(spread_t, 4),
        confidence_metrics=conf_metrics,
        ray_results=ray_results,
        gauge_geometry=gauge,
    )

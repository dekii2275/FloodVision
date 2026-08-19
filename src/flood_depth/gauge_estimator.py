from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from src.flood_depth.calibration import CalibrationPoint, piecewise_linear_interpolate
from src.flood_depth.gauge_geometry import VirtualGauge
from src.flood_depth.gauge_sampling import (
    GaugeMeasurement,
    RayWaterlineResult,
    find_base_connected_waterline,
    measure_gauge,
)
from src.flood_depth.gauge_validation import GaugeStatus
from src.flood_depth.fusion import (
    FusionResult,
    SEVERITY_LEVELS,
    SeverityLevel,
    classify_severity,
    fuse_gauge_measurements,
)


def measure_gauge_depth(
    gauge: VirtualGauge,
    water_mask: np.ndarray,
    search_radius_px: int = 8,
    prob_map: np.ndarray | None = None,
    labels_map: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> GaugeMeasurement:
    """
    Backward-compatible wrapper for measuring flood depth on a virtual gauge.
    """
    if prob_map is None:
        prob_map = water_mask.astype(np.float32)
    if labels_map is None:
        labels_map = water_mask.astype(np.int32)

    cfg = dict(config or {})
    cfg.setdefault("base_search_radius_px", search_radius_px)

    return measure_gauge(
        gauge=gauge,
        flood_mask=water_mask,
        prob_map=prob_map,
        labels_map=labels_map,
        config=cfg,
    )


def fuse_gauge_depths(
    measurements: Sequence[GaugeMeasurement],
    strategy: str = "median",
    flood_area_pct: float = 0.0,
    levels_cfg: dict[str, float] | None = None,
    outlier_abs_cm: float = 25.0,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper returning dict format for legacy callers.
    """
    cfg = {"method": strategy, "outlier_abs_cm": outlier_abs_cm}
    res = fuse_gauge_measurements(measurements, flood_area_pct=flood_area_pct, config=cfg)
    
    return {
        "fused_depth_cm": res.fused_depth_cm if res.fused_depth_cm is not None else 0.0,
        "max_depth_cm": res.max_depth_cm,
        "mean_depth_cm": res.mean_depth_cm,
        "flood_level": res.flood_level,
        "alert_status": res.alert_status,
        "alert_color_bgr": res.alert_color_bgr,
        "active_gauges_count": len(res.used_gauges),
        "used_gauges": res.used_gauges,
        "rejected_gauges": res.rejected_gauges,
        "fusion_quality": res.fusion_quality,
        "severity": res.severity.to_dict(),
    }

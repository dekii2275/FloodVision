from __future__ import annotations

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
from src.flood_depth.roi_filter import (
    create_roi_mask,
    extract_water_contours,
    filter_connected_components,
    find_component_near_gauge_base,
)
from src.flood_depth.visualizer import make_depth_dashboard_visualization
from src.flood_depth.pipeline import (
    FloodVisionPipeline,
    PipelineConfig,
    PipelineOutput,
    DEFAULT_WATER_PROMPTS,
)

__all__ = [
    "FloodVisionPipeline",
    "PipelineConfig",
    "PipelineOutput",
    "DEFAULT_WATER_PROMPTS",
    "VirtualGauge",
    "GaugeMeasurement",
    "GaugeStatus",
    "RayWaterlineResult",
    "CalibrationPoint",
    "piecewise_linear_interpolate",
    "find_base_connected_waterline",
    "measure_gauge",
    "FusionResult",
    "SeverityLevel",
    "SEVERITY_LEVELS",
    "classify_severity",
    "fuse_gauge_measurements",
    "create_roi_mask",
    "extract_water_contours",
    "filter_connected_components",
    "find_component_near_gauge_base",
    "make_depth_dashboard_visualization",
]

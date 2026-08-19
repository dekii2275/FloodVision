from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.flood_depth.calibration import (
    CalibrationPoint,
    piecewise_linear_interpolate,
    validate_calibration_points,
)
from src.flood_depth.gauge_geometry import VirtualGauge
from src.flood_depth.gauge_sampling import (
    GaugeMeasurement,
    find_base_connected_waterline,
    measure_gauge,
)
from src.flood_depth.gauge_validation import GaugeStatus
from src.flood_depth.fusion import fuse_gauge_measurements
from src.flood_depth.roi_filter import (
    create_roi_mask,
    filter_connected_components,
    find_component_near_gauge_base,
)


# ==============================================================================
# CASE A: OUTLIER REJECTION (G1 = 48.1, G2 = 44.6, G3 = 149.7)
# ==============================================================================
def test_case_a_outlier_rejection():
    """
    Case A: G1=48.1, G2=44.6, G3=149.7
    Expected: G3 marked as OUTLIER and rejected. Fused depth ≈ 44-48 cm (L2: Ngập vừa), KHÔNG nhảy lên L4.
    """
    vg1 = VirtualGauge(id="G1", name="Mốc 1", base_pt=(100, 500), top_pt=(100, 100), max_height_cm=100.0)
    vg2 = VirtualGauge(id="G2", name="Mốc 2", base_pt=(200, 500), top_pt=(200, 100), max_height_cm=100.0)
    vg3 = VirtualGauge(id="G3", name="Mốc 3", base_pt=(300, 500), top_pt=(300, 100), max_height_cm=200.0)

    m1 = GaugeMeasurement(
        gauge_id="G1",
        gauge_name="Mốc 1",
        status=GaugeStatus.VALID,
        local_depth_cm=48.1,
        waterline_t=0.481,
        gauge_geometry=vg1,
    )
    m2 = GaugeMeasurement(
        gauge_id="G2",
        gauge_name="Mốc 2",
        status=GaugeStatus.VALID,
        local_depth_cm=44.6,
        waterline_t=0.446,
        gauge_geometry=vg2,
    )
    m3 = GaugeMeasurement(
        gauge_id="G3",
        gauge_name="Mốc 3",
        status=GaugeStatus.VALID,
        local_depth_cm=149.7,
        waterline_t=0.748,
        gauge_geometry=vg3,
    )

    fusion_res = fuse_gauge_measurements(
        measurements=[m1, m2, m3],
        flood_area_pct=30.0,
        config={"method": "median", "outlier_abs_cm": 25.0},
    )

    # 1. G3 phải bị loại bỏ và đánh dấu OUTLIER
    assert "G3" in fusion_res.rejected_gauges
    assert m3.status == GaugeStatus.OUTLIER

    # 2. G1 và G2 được sử dụng
    assert "G1" in fusion_res.used_gauges
    assert "G2" in fusion_res.used_gauges

    # 3. Độ sâu hợp nhất phải nằm trong khoảng [44.6, 48.1] cm
    assert fusion_res.fused_depth_cm is not None
    assert 44.0 <= fusion_res.fused_depth_cm <= 49.0
    assert fusion_res.severity.level_code == "L2"  # L2: 20-50 cm, KHÔNG PHẢI L4!


# ==============================================================================
# CASE B: BASE-CONNECTED WATERLINE (Main Flood at 40cm, False Blob at 110cm)
# ==============================================================================
def test_case_b_base_connected_waterline_ignores_false_blob():
    """
    Case B: Gauge có vùng nước liên tục từ chân lên 40cm, phía trên cao tại 110cm có đốm nước ảo.
    Expected: find_base_connected_waterline chỉ lấy điểm ranh giới của vùng nước kết nối với chân (≈ 40cm),
              bỏ qua hoàn toàn false-positive blob ở 110cm.
    """
    # Tạo ray 100 điểm: Chân (0) -> Đỉnh (99)
    n_samples = 100
    ray_binary = np.zeros(n_samples, dtype=np.uint8)
    ray_labels = np.zeros(n_samples, dtype=np.int32)

    # Vùng nước chính kết nối từ chân mốc (0 đến 40)
    ray_binary[0:41] = 1
    ray_labels[0:41] = 1

    # Đốm false-positive ở phía trên cao (75 đến 85)
    ray_binary[75:86] = 1
    ray_labels[75:86] = 2  # Hoặc cùng label 1

    t_waterline, status, last_idx = find_base_connected_waterline(
        ray_binary=ray_binary,
        ray_labels=ray_labels,
        target_component_id=1,
        stable_transition_samples=3,
    )

    assert status == GaugeStatus.VALID
    assert last_idx == 40
    assert abs(t_waterline - 0.404) < 0.02


# ==============================================================================
# CASE C: GAUGE BASE DRY (Chân mốc không chạm nước)
# ==============================================================================
def test_case_c_dry_gauge_base():
    """
    Case C: Gauge base không chạm flood component.
    Expected: status = NO_WATER_AT_BASE, depth = 0.0 cm.
    """
    mask = np.zeros((400, 400), dtype=np.uint8)
    # Vùng nước nằm xa ở góc khác
    mask[50:100, 50:100] = 1
    labels = mask.astype(np.int32)
    probs = mask.astype(np.float32)

    # Gauge đặt ở vùng hoàn toàn khô ráo
    gauge = VirtualGauge(
        id="G_dry",
        name="Mốc Khô",
        base_pt=(300, 350),
        top_pt=(300, 150),
        max_height_cm=100.0,
    )

    meas = measure_gauge(
        gauge=gauge,
        flood_mask=mask,
        prob_map=probs,
        labels_map=labels,
        config={"base_search_radius_px": 8},
    )

    assert meas.status == GaugeStatus.NO_WATER_AT_BASE
    assert meas.local_depth_cm == 0.0
    assert meas.waterline_t == 0.0


# ==============================================================================
# CASE D: GAUGE BAND MULTI-RAY MEDIAN (10 rays ~45cm, 5 noisy rays at 80-120cm)
# ==============================================================================
def test_case_d_gauge_band_multi_ray_median():
    """
    Case D: 10/15 tia cho kết quả ~45cm, 5 tia bị nhiễu nhảy lên 80-120cm.
    Expected: Median của dải tia cho kết quả ổn định ≈ 45cm.
    """
    h, w = 500, 500
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Tạo vùng nước ngập từ y=500 lên y=320 (cao 180px / 400px = 45% = 45cm)
    mask[320:500, 200:300] = 1
    # Thêm vài đốm nhiễu ở rìa trên cao
    mask[150:200, 260:290] = 1

    labels = np.zeros((h, w), dtype=np.int32)
    labels[mask == 1] = 1
    probs = mask.astype(np.float32)

    gauge = VirtualGauge(
        id="G_band",
        name="Mốc Dải Tia",
        base_pt=(250, 499),
        top_pt=(250, 100),
        band_width_px=20,
        max_height_cm=100.0,
    )

    meas = measure_gauge(
        gauge=gauge,
        flood_mask=mask,
        prob_map=probs,
        labels_map=labels,
        config={"num_rays": 15, "base_search_radius_px": 8},
    )

    assert meas.status == GaugeStatus.VALID
    assert meas.valid_rays >= 10
    # Độ sâu đo được phải xấp xỉ 45cm (180px / 400px * 100cm = 45.0 cm)
    assert 43.0 <= meas.local_depth_cm <= 47.0


# ==============================================================================
# CASE E: MULTI-POINT PIECEWISE LINEAR INTERPOLATION
# ==============================================================================
def test_case_e_multi_point_calibration_interpolation():
    """
    Case E: Calibration phi tuyến tính với các mốc [0cm, 20cm, 40cm, 60cm, 80cm].
    Expected: Nội suy chính xác tại các điểm trung gian.
    """
    calib_pts = [
        CalibrationPoint(t=0.00, depth_cm=0.0),
        CalibrationPoint(t=0.25, depth_cm=20.0),
        CalibrationPoint(t=0.50, depth_cm=40.0),
        CalibrationPoint(t=0.75, depth_cm=60.0),
        CalibrationPoint(t=1.00, depth_cm=80.0),
    ]

    assert validate_calibration_points(calib_pts) is True

    # Tại t = 0.375 (nằm giữa t=0.25 và t=0.50) -> độ sâu = 30.0 cm
    d_mid, mode, warn = piecewise_linear_interpolate(0.375, calib_pts, max_height_cm=80.0)
    assert mode == "multi_point"
    assert warn is None
    assert abs(d_mid - 30.0) < 1e-3

    # Tại t = 0.0 -> 0.0 cm
    d_base, _, _ = piecewise_linear_interpolate(0.0, calib_pts)
    assert d_base == 0.0

    # Tại t = 1.0 -> 80.0 cm
    d_top, _, _ = piecewise_linear_interpolate(1.0, calib_pts)
    assert d_top == 80.0


# ==============================================================================
# CASE F: OUT OF CALIBRATION RANGE
# ==============================================================================
def test_case_f_out_of_calibration_range():
    """
    Case F: Waterline vượt quá đỉnh mốc hiệu chuẩn cao nhất.
    Expected: Trả về trạng thái OUT_OF_CALIBRATION_RANGE và không ngoại suy vô hạn.
    """
    calib_pts = [
        CalibrationPoint(t=0.00, depth_cm=0.0),
        CalibrationPoint(t=0.50, depth_cm=50.0),
        CalibrationPoint(t=0.80, depth_cm=80.0),  # Đỉnh hiệu chuẩn chỉ đến t=0.80
    ]

    d_out, mode, warn = piecewise_linear_interpolate(0.95, calib_pts, max_height_cm=100.0)
    assert mode == "multi_point"
    assert warn == "OUT_OF_CALIBRATION_RANGE"
    assert d_out == 80.0  # Chặn ở đỉnh mốc hiệu chuẩn, không ngoại suy vô hạn


# ==============================================================================
# CASE G: ROI RATIO CONNECTED COMPONENT FILTERING
# ==============================================================================
def test_case_g_roi_ratio_component_filtering():
    """
    Case G: Lọc thành phần liên thông theo tỷ lệ diện tích ROI.
    """
    h, w = 400, 400
    roi_mask = np.ones((h, w), dtype=np.uint8)
    binary = np.zeros((h, w), dtype=np.uint8)

    # Đốm rác nhỏ 20 pixel (< 50px lower bound và < 0.001 * 160,000 = 160px)
    binary[10:14, 10:15] = 1

    # Mảng ngập lớn 500 pixel
    binary[200:225, 200:220] = 1

    filtered, labels, stats = filter_connected_components(
        binary_mask=binary,
        roi_mask=roi_mask,
        min_component_ratio=0.001,
        min_component_px=50,
    )

    # Đốm nhỏ bị lọc bỏ, mảng lớn được giữ lại
    assert filtered[12, 12] == 0
    assert filtered[210, 210] == 1

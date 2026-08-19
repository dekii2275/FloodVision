from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class CalibrationPoint:
    t: float          # Vị trí chuẩn hóa dọc theo thân mốc (0.0 = Chân mốc, 1.0 = Đỉnh mốc)
    depth_cm: float   # Độ sâu ngập thực tế tương ứng (cm)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPoint:
        return cls(
            t=float(data.get("t", 0.0)),
            depth_cm=float(data.get("depth_cm", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {"t": round(self.t, 4), "depth_cm": round(self.depth_cm, 1)}


def validate_calibration_points(points: Sequence[CalibrationPoint]) -> bool:
    if not points or len(points) < 2:
        return False
    sorted_pts = sorted(points, key=lambda p: p.t)
    for i in range(len(sorted_pts) - 1):
        p1 = sorted_pts[i]
        p2 = sorted_pts[i + 1]
        # t phải tăng đơn điệu và nằm trong [0.0, 1.0]
        if p1.t < 0.0 or p2.t > 1.0 or p2.t <= p1.t:
            return False
        # depth_cm phải tăng đơn điệu
        if p2.depth_cm < p1.depth_cm:
            return False
    return True


def piecewise_linear_interpolate(
    t: float,
    calibration_points: Sequence[CalibrationPoint] | None = None,
    max_height_cm: float = 100.0,
) -> tuple[float, str, str | None]:
    """
    Nội suy tuyến tính từng đoạn (Piecewise Linear Interpolation) từ tham số t sang độ sâu cm.
    
    Returns:
        (depth_cm, calibration_mode, warning_or_error)
    """
    # 1. Fallback chế độ tuyến tính cũ nếu không có calibration_points
    if not calibration_points or len(calibration_points) < 2 or not validate_calibration_points(calibration_points):
        clamped_t = max(0.0, min(1.0, float(t)))
        depth_cm = clamped_t * float(max_height_cm)
        return (
            round(depth_cm, 1),
            "legacy_linear",
            "Gauge is using linear calibration (no multi-point calibration configured)",
        )

    sorted_pts = sorted(calibration_points, key=lambda p: p.t)

    # 2. Kiểm tra nếu t vượt ngoài dải hiệu chuẩn (không ngoại suy vô hạn)
    if t < sorted_pts[0].t:
        return (sorted_pts[0].depth_cm, "multi_point", None)
    
    if t > sorted_pts[-1].t:
        # Vượt quá đỉnh mốc hiệu chuẩn cao nhất
        return (
            sorted_pts[-1].depth_cm,
            "multi_point",
            "OUT_OF_CALIBRATION_RANGE",
        )

    # 3. Tìm đoạn [p_i, p_{i+1}] chứa t và nội suy
    for i in range(len(sorted_pts) - 1):
        p1 = sorted_pts[i]
        p2 = sorted_pts[i + 1]
        if p1.t <= t <= p2.t:
            segment_range = p2.t - p1.t
            if segment_range <= 1e-6:
                return (p1.depth_cm, "multi_point", None)
            ratio = (t - p1.t) / segment_range
            depth_cm = p1.depth_cm + ratio * (p2.depth_cm - p1.depth_cm)
            return (round(depth_cm, 1), "multi_point", None)

    return (sorted_pts[-1].depth_cm, "multi_point", None)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence
import numpy as np

from src.flood_depth.gauge_sampling import GaugeMeasurement
from src.flood_depth.gauge_validation import GaugeStatus


@dataclass
class SeverityLevel:
    level_code: str
    level_name: str
    depth_range: str
    severity: str
    badge_color: str
    advice: str

    def to_dict(self) -> dict[str, str]:
        return {
            "level_code": self.level_code,
            "level_name": self.level_name,
            "depth_range": self.depth_range,
            "severity": self.severity,
            "badge_color": self.badge_color,
            "advice": self.advice,
        }


SEVERITY_LEVELS = {
    "L0": SeverityLevel(
        level_code="L0",
        level_name="L0: Không ngập",
        depth_range="0 cm",
        severity="AN TOÀN",
        badge_color="#10b981",
        advice="Mặt đường khô ráo hoặc chỉ ướt nhẹ sau mưa. Mọi phương tiện lưu thông bình thường, an toàn tuyệt đối.",
    ),
    "L1": SeverityLevel(
        level_code="L1",
        level_name="L1: Ngập nông",
        depth_range="0 – 20 cm",
        severity="THẤP / AN TOÀN",
        badge_color="#84cc16",
        advice="Mực nước mấp mé mép lốp xe. Xe máy và ô tô con có thể di chuyển chậm qua vùng ngập.",
    ),
    "L2": SeverityLevel(
        level_code="L2",
        level_name="L2: Ngập vừa",
        depth_range="20 – 50 cm",
        severity="CẢNH BÁO",
        badge_color="#f59e0b",
        advice="Nước ngập nửa bánh xe, đến gần miệng ống xả xe máy. Khuyến cáo xe gầm thấp không nên đi qua.",
    ),
    "L3": SeverityLevel(
        level_code="L3",
        level_name="L3: Ngập sâu",
        depth_range="50 – 100 cm",
        severity="NGUY HIỂM CAO",
        badge_color="#ef4444",
        advice="Nước ngập qua yên xe máy / nắp capo ô tô con. Nguy cơ thủy kích phá hủy động cơ cực cao. Cấm xe máy và ô tô con lưu thông.",
    ),
    "L4": SeverityLevel(
        level_code="L4",
        level_name="L4: Ngập nghiêm trọng",
        depth_range="> 100 cm",
        severity="THẢM HỌA / CẤM TUYỆT ĐỐI",
        badge_color="#9333ea",
        advice="Nước ngập lút nóc xe, dòng chảy xiết nguy hiểm đến tính mạng. Cấm tất cả phương tiện và người đi bộ.",
    ),
}


@dataclass
class FusionResult:
    fused_depth_cm: float | None
    used_gauges: List[str]
    rejected_gauges: List[str]
    fusion_method: str
    fusion_quality: str                       # 'GOOD', 'LOW', 'UNKNOWN'
    severity: SeverityLevel
    measurements: List[GaugeMeasurement] = field(default_factory=list)

    # Backward compatibility properties
    @property
    def max_depth_cm(self) -> float:
        valid_depths = [m.local_depth_cm for m in self.measurements if m.status == GaugeStatus.VALID]
        return float(np.max(valid_depths)) if valid_depths else 0.0

    @property
    def mean_depth_cm(self) -> float:
        valid_depths = [m.local_depth_cm for m in self.measurements if m.status == GaugeStatus.VALID]
        return float(np.mean(valid_depths)) if valid_depths else 0.0

    @property
    def flood_level(self) -> str:
        return self.severity.level_code

    @property
    def alert_status(self) -> str:
        return self.severity.level_name

    @property
    def alert_color_bgr(self) -> List[int]:
        hex_c = self.severity.badge_color.lstrip("#")
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        return [rgb[2], rgb[1], rgb[0]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fused_depth_cm": round(self.fused_depth_cm, 1) if self.fused_depth_cm is not None else None,
            "max_depth_cm": round(self.max_depth_cm, 1),
            "mean_depth_cm": round(self.mean_depth_cm, 1),
            "fusion_method": self.fusion_method,
            "fusion_quality": self.fusion_quality,
            "used_gauges": self.used_gauges,
            "rejected_gauges": self.rejected_gauges,
            "severity": self.severity.to_dict(),
        }


def classify_severity(fused_depth_cm: float | None, flood_area_pct: float) -> SeverityLevel:
    if fused_depth_cm is None:
        return SeverityLevel(
            level_code="UNKNOWN",
            level_name="Chưa xác định",
            depth_range="--",
            severity="KHÔNG RÕ",
            badge_color="#64748b",
            advice="Không có cột mốc hợp lệ để đo độ sâu.",
        )

    if fused_depth_cm <= 0.5 or flood_area_pct < 1.0:
        return SEVERITY_LEVELS["L0"]
    elif fused_depth_cm <= 20.0:
        return SEVERITY_LEVELS["L1"]
    elif fused_depth_cm <= 50.0:
        return SEVERITY_LEVELS["L2"]
    elif fused_depth_cm <= 100.0:
        return SEVERITY_LEVELS["L3"]
    else:
        return SEVERITY_LEVELS["L4"]


def fuse_gauge_measurements(
    measurements: Sequence[GaugeMeasurement],
    flood_area_pct: float = 0.0,
    config: dict[str, Any] | None = None,
) -> FusionResult:
    """
    Hợp nhất các phép đo độ sâu từ các cột mốc với cơ chế Robust Median Fusion và Outlier Rejection.
    """
    cfg = config or {}
    outlier_abs_cm = float(cfg.get("outlier_abs_cm", 25.0))
    fusion_method = str(cfg.get("method", "median")).lower()

    if not measurements:
        severity = classify_severity(None, flood_area_pct)
        return FusionResult(
            fused_depth_cm=None,
            used_gauges=[],
            rejected_gauges=[],
            fusion_method=fusion_method,
            fusion_quality="UNKNOWN",
            severity=severity,
            measurements=[],
        )

    # 1. Thu thập các gauge hợp lệ
    valid_submerged = [
        m for m in measurements if m.status == GaugeStatus.VALID and m.local_depth_cm > 0.5
    ]
    dry_or_no_water = [
        m for m in measurements if m.status == GaugeStatus.NO_WATER_AT_BASE or (m.status == GaugeStatus.VALID and m.local_depth_cm <= 0.5)
    ]
    invalid_or_insufficient = [
        m for m in measurements if m.status not in (GaugeStatus.VALID, GaugeStatus.NO_WATER_AT_BASE)
    ]

    rejected_gauges: List[str] = [m.gauge_id for m in invalid_or_insufficient]
    used_gauges: List[str] = []

    # 2. Xử lý trường hợp không có mốc nào ngập nước
    if not valid_submerged:
        # Nếu toàn bộ là khô ráo
        fused_depth = 0.0
        used_gauges = [m.gauge_id for m in dry_or_no_water]
        severity = classify_severity(0.0, flood_area_pct)
        return FusionResult(
            fused_depth_cm=0.0,
            used_gauges=used_gauges,
            rejected_gauges=rejected_gauges,
            fusion_method=fusion_method,
            fusion_quality="GOOD" if len(used_gauges) >= 2 else "LOW",
            severity=severity,
            measurements=list(measurements),
        )

    # 3. Xử lý trường hợp chỉ có 1 mốc ngập nước hợp lệ
    if len(valid_submerged) == 1:
        single_m = valid_submerged[0]
        fused_depth = single_m.local_depth_cm
        used_gauges = [single_m.gauge_id]
        severity = classify_severity(fused_depth, flood_area_pct)
        return FusionResult(
            fused_depth_cm=round(fused_depth, 1),
            used_gauges=used_gauges,
            rejected_gauges=rejected_gauges,
            fusion_method=fusion_method,
            fusion_quality="LOW",
            severity=severity,
            measurements=list(measurements),
        )

    # 4. Có từ 2 mốc ngập nước trở lên -> Áp dụng Outlier Rejection trên Median
    depths = [m.local_depth_cm for m in valid_submerged]
    median_d = float(np.median(depths))

    inliers: List[GaugeMeasurement] = []
    for m in valid_submerged:
        diff = abs(m.local_depth_cm - median_d)
        if diff > outlier_abs_cm:
            # Phát hiện mốc đo lệch bất thường -> Đánh dấu OUTLIER
            m.status = GaugeStatus.OUTLIER
            m.warning = f"Outlier rejected: depth {m.local_depth_cm:.1f}cm deviates by {diff:.1f}cm from median {median_d:.1f}cm"
            rejected_gauges.append(m.gauge_id)
        else:
            inliers.append(m)
            used_gauges.append(m.gauge_id)

    # 5. Tính toán độ sâu hợp nhất từ các Inliers
    if not inliers:
        # Nếu tất cả đều lệch, lấy median_d với chất lượng LOW
        fused_depth = median_d
        fusion_quality = "LOW"
    elif fusion_method == "mean":
        fused_depth = float(np.mean([m.local_depth_cm for m in inliers]))
        fusion_quality = "GOOD" if len(inliers) >= 2 else "LOW"
    elif fusion_method == "max":
        # Fallback chế độ max nếu user chủ động cấu hình
        fused_depth = float(np.max([m.local_depth_cm for m in inliers]))
        fusion_quality = "GOOD" if len(inliers) >= 2 else "LOW"
    else:
        # Default: Robust Median
        fused_depth = float(np.median([m.local_depth_cm for m in inliers]))
        fusion_quality = "GOOD" if len(inliers) >= 2 else "LOW"

    severity = classify_severity(fused_depth, flood_area_pct)

    return FusionResult(
        fused_depth_cm=round(fused_depth, 1),
        used_gauges=used_gauges,
        rejected_gauges=rejected_gauges,
        fusion_method=fusion_method,
        fusion_quality=fusion_quality,
        severity=severity,
        measurements=list(measurements),
    )

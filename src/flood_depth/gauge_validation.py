from __future__ import annotations

from enum import Enum


class GaugeStatus(str, Enum):
    VALID = "VALID"
    NO_WATER_AT_BASE = "NO_WATER_AT_BASE"
    NO_INTERSECTION = "NO_INTERSECTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OUT_OF_CALIBRATION_RANGE = "OUT_OF_CALIBRATION_RANGE"
    OUTLIER = "OUTLIER"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    INSUFFICIENT_VALID_RAYS = "INSUFFICIENT_VALID_RAYS"
    OCCLUDED = "OCCLUDED"

    def is_usable_for_fusion(self) -> bool:
        return self == GaugeStatus.VALID

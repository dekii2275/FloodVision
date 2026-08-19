from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence, Tuple


def create_roi_mask(
    image_shape: tuple[int, int],
    roi_polygon_normalized: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """
    Tạo mặt nạ nhị phân ROI (Region of Interest) cho mặt đường từ đa giác chuẩn hóa [0..1, 0..1].
    """
    height, width = image_shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    if roi_polygon_normalized is None or len(roi_polygon_normalized) < 3:
        roi_mask[:] = 1
        return roi_mask

    pts_px = np.array([
        [int(np.clip(p[0] * width, 0, width - 1)), int(np.clip(p[1] * height, 0, height - 1))]
        for p in roi_polygon_normalized
    ], dtype=np.int32)

    cv2.fillPoly(roi_mask, [pts_px], 1)
    return roi_mask


def filter_connected_components(
    binary_mask: np.ndarray,
    roi_mask: np.ndarray | None = None,
    min_component_ratio: float = 0.001,
    min_component_px: int = 50,
    morph_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lọc các đốm nhiễu nhỏ không liên thông, chỉ giữ lại các mảng nước ngập chính.
    
    Returns:
        (filtered_binary_mask, labels_map, stats_array)
    """
    if binary_mask.max() == 0:
        h, w = binary_mask.shape[:2]
        return np.zeros_like(binary_mask), np.zeros((h, w), dtype=np.int32), np.zeros((1, 5), dtype=np.int32)

    # 1. Hậu xử lý hình thái học (Morphological Closing & Opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 2. Tính diện tích tối thiểu theo ROI ratio hoặc lower bound tuyệt đối
    roi_pixels = float(np.sum(roi_mask == 1)) if roi_mask is not None else float(binary_mask.size)
    effective_min_area = max(int(min_component_px), int(round(roi_pixels * float(min_component_ratio))))

    # 3. Phân tích các thành phần liên thông (Connected Components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    filtered_mask = np.zeros_like(cleaned)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= effective_min_area:
            filtered_mask[labels == label_idx] = 1

    return filtered_mask, labels, stats


def find_component_near_gauge_base(
    labels_map: np.ndarray,
    base_pt: Tuple[int, int],
    search_radius_px: int = 8,
) -> int | None:
    """
    Xác định connected component ID của vùng nước ngập tiếp xúc trực tiếp hoặc nằm sát chân cột mốc (base).
    """
    h, w = labels_map.shape[:2]
    bx, by = base_pt

    x1 = max(0, bx - search_radius_px)
    x2 = min(w, bx + search_radius_px + 1)
    y1 = max(0, by - search_radius_px)
    y2 = min(h, by + search_radius_px + 1)

    region = labels_map[y1:y2, x1:x2]
    water_labels = region[region > 0]

    if water_labels.size == 0:
        return None

    # Lấy component label xuất hiện nhiều nhất quanh chân mốc
    vals, counts = np.unique(water_labels, return_counts=True)
    dominant_label = int(vals[np.argmax(counts)])
    return dominant_label


def extract_water_contours(binary_mask: np.ndarray) -> list[np.ndarray]:
    """
    Trích xuất danh sách đường viền ranh giới mặt nước (Water Boundary Contours).
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

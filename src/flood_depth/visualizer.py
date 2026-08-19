from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Sequence

from src.flood_depth.gauge_sampling import GaugeMeasurement
from src.flood_depth.gauge_validation import GaugeStatus
from src.flood_depth.fusion import FusionResult


def make_depth_dashboard_visualization(
    orig_bgr: np.ndarray,
    prob_map: np.ndarray,
    final_water_mask: np.ndarray,
    roi_mask: np.ndarray,
    roi_polygon_normalized: Sequence[Sequence[float]] | None,
    measurements: Sequence[GaugeMeasurement],
    fusion_result: FusionResult | dict[str, Any],
    flood_area_pct: float,
    output_path: str,
) -> None:
    """
    Tạo Dashboard trực quan hóa 4 khung hình chuyên nghiệp cho hệ thống giám sát ngập.
    """
    height, width = orig_bgr.shape[:2]

    # Normalize fusion_result
    if isinstance(fusion_result, dict):
        fused_depth = float(fusion_result.get("fused_depth_cm", 0.0))
        alert_txt = str(fusion_result.get("alert_status", "KHÔ RÁO"))
        alert_color_bgr = tuple(fusion_result.get("alert_color_bgr", [0, 255, 0]))
        used_gauges = list(fusion_result.get("used_gauges", []))
        rejected_gauges = list(fusion_result.get("rejected_gauges", []))
    else:
        fused_depth = float(fusion_result.fused_depth_cm) if fusion_result.fused_depth_cm is not None else 0.0
        alert_txt = fusion_result.severity.level_name
        alert_color_bgr = tuple(fusion_result.alert_color_bgr)
        used_gauges = fusion_result.used_gauges
        rejected_gauges = fusion_result.rejected_gauges

    # -------------------------------------------------------------
    # Khung 1: Ảnh gốc + Đường bao ROI Camera + Vị trí các Cột đo ảo
    # -------------------------------------------------------------
    panel1_bgr = orig_bgr.copy()
    if roi_polygon_normalized and len(roi_polygon_normalized) >= 3:
        roi_pts = np.array([
            [int(p[0] * width), int(p[1] * height)] for p in roi_polygon_normalized
        ], dtype=np.int32)
        cv2.polylines(panel1_bgr, [roi_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    for m in measurements:
        bx, by = m.base_pt
        tx, ty = m.top_pt

        # Vẽ dải đo (Gauge Band) nếu có geometry
        if m.gauge_geometry:
            band_rays = m.gauge_geometry.get_band_rays(num_rays=2)
            if len(band_rays) == 2:
                r1_b, r1_t, _ = band_rays[0]
                r2_b, r2_t, _ = band_rays[1]
                band_poly = np.array([r1_b, r1_t, r2_t, r2_b], dtype=np.int32)
                overlay = panel1_bgr.copy()
                cv2.fillPoly(overlay, [band_poly], (255, 200, 100))
                cv2.addWeighted(overlay, 0.25, panel1_bgr, 0.75, 0, panel1_bgr)

        # Thân cột mốc
        cv2.line(panel1_bgr, (bx, by), (tx, ty), (255, 255, 255), 2)
        # Điểm đáy và điểm đỉnh
        cv2.circle(panel1_bgr, (bx, by), 5, (0, 255, 0), -1)
        cv2.circle(panel1_bgr, (tx, ty), 5, (0, 0, 255), -1)
        # Nhãn tên cột
        cv2.putText(panel1_bgr, m.gauge_id, (bx - 15, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    panel1_rgb = cv2.cvtColor(panel1_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------
    # Khung 2: Heatmap xác suất mặt nước (áp dụng ROI)
    # -------------------------------------------------------------
    prob_masked = prob_map * roi_mask

    # -------------------------------------------------------------
    # Khung 3: Final Flood Mask + Ranh giới mực nước (Yellow Contours)
    # -------------------------------------------------------------
    panel3_bgr = orig_bgr.copy()
    mask_idx = (final_water_mask == 1)
    if np.any(mask_idx):
        water_color = np.zeros_like(orig_bgr)
        water_color[mask_idx] = [255, 180, 0]  # Cyan-Blue
        blended = cv2.addWeighted(orig_bgr, 0.55, water_color, 0.45, 0)
        panel3_bgr[mask_idx] = blended[mask_idx]
        # Vẽ viền ranh giới mực nước
        contours, _ = cv2.findContours(final_water_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel3_bgr, contours, -1, (0, 255, 255), 2)

    panel3_rgb = cv2.cvtColor(panel3_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------
    # Khung 4: Trực quan hóa đo độ sâu tại các Cột mốc & Badge cảnh báo
    # -------------------------------------------------------------
    panel4_bgr = orig_bgr.copy()
    if np.any(mask_idx):
        water_color = np.zeros_like(orig_bgr)
        water_color[mask_idx] = [255, 180, 0]
        panel4_bgr[mask_idx] = cv2.addWeighted(orig_bgr, 0.7, water_color, 0.3, 0)[mask_idx]

    # Vẽ thước đo độ sâu cho từng cột mốc
    for m in measurements:
        bx, by = m.base_pt
        tx, ty = m.top_pt

        # Vẽ dải đo mờ
        if m.gauge_geometry:
            band_rays = m.gauge_geometry.get_band_rays(num_rays=2)
            if len(band_rays) == 2:
                r1_b, r1_t, _ = band_rays[0]
                r2_b, r2_t, _ = band_rays[1]
                band_poly = np.array([r1_b, r1_t, r2_t, r2_b], dtype=np.int32)
                overlay = panel4_bgr.copy()
                band_color = (0, 0, 255) if m.status == GaugeStatus.OUTLIER else (100, 200, 255)
                cv2.fillPoly(overlay, [band_poly], band_color)
                cv2.addWeighted(overlay, 0.20, panel4_bgr, 0.80, 0, panel4_bgr)

        # Thân thước
        gauge_line_color = (50, 50, 220) if m.status == GaugeStatus.OUTLIER else (120, 120, 120)
        cv2.line(panel4_bgr, (bx, by), (tx, ty), gauge_line_color, 4)

        # Vẽ candidate waterline points của từng ray
        for r_res in m.ray_results:
            if r_res.is_valid:
                rx, ry = r_res.intersection_pt
                cv2.circle(panel4_bgr, (rx, ry), 2, (0, 255, 255), -1)

        if m.status == GaugeStatus.VALID and m.intersection_pt is not None:
            ix, iy = m.intersection_pt
            # Thân ngập nước màu đỏ
            cv2.line(panel4_bgr, (bx, by), (ix, iy), (0, 0, 255), 6)
            # Điểm median waterline màu vàng
            cv2.circle(panel4_bgr, (ix, iy), 6, (0, 255, 255), -1)
            cv2.circle(panel4_bgr, (ix, iy), 8, (0, 0, 0), 2)

            label = f"{m.gauge_id}: {m.local_depth_cm:.1f} cm"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            lbl_x = max(10, min(width - t_size[0] - 15, ix + 12))
            lbl_y = max(25, iy)
            cv2.rectangle(panel4_bgr, (lbl_x - 4, lbl_y - t_size[1] - 4), (lbl_x + t_size[0] + 4, lbl_y + 4), (0, 0, 0), -1)
            cv2.putText(panel4_bgr, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

        elif m.status == GaugeStatus.OUTLIER and m.intersection_pt is not None:
            ix, iy = m.intersection_pt
            # Outlier hiển thị màu đỏ cam gạch chéo
            cv2.line(panel4_bgr, (bx, by), (ix, iy), (0, 100, 255), 4)
            cv2.circle(panel4_bgr, (ix, iy), 6, (0, 0, 255), -1)
            label = f"{m.gauge_id}: {m.local_depth_cm:.1f} cm (OUTLIER)"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            lbl_x = max(10, min(width - t_size[0] - 15, ix + 12))
            lbl_y = max(25, iy)
            cv2.rectangle(panel4_bgr, (lbl_x - 4, lbl_y - t_size[1] - 4), (lbl_x + t_size[0] + 4, lbl_y + 4), (0, 0, 0), -1)
            cv2.putText(panel4_bgr, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)

        else:
            # Khô ráo / NO_WATER_AT_BASE
            cv2.circle(panel4_bgr, (bx, by), 5, (0, 255, 0), -1)
            label = f"{m.gauge_id}: 0.0 cm"
            cv2.putText(panel4_bgr, label, (bx + 8, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Banner tổng hợp
    used_str = ", ".join(used_gauges) if used_gauges else "None"
    rej_str = f" | REJECTED: {', '.join(rejected_gauges)}" if rejected_gauges else ""
    banner_txt = f"DEPTH: {fused_depth:.1f} cm | {alert_txt} | USED: {used_str}{rej_str}"

    cv2.rectangle(panel4_bgr, (0, 0), (width, 42), (20, 20, 20), -1)
    cv2.putText(panel4_bgr, banner_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, alert_color_bgr, 2, cv2.LINE_AA)

    panel4_rgb = cv2.cvtColor(panel4_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------
    # Xuất Figure 4 panel
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)

    axes[0, 0].imshow(panel1_rgb)
    axes[0, 0].set_title("1. Camera ROI & Virtual Gauges (G1, G2, G3)", fontsize=11, fontweight='bold', pad=6)
    axes[0, 0].axis('off')

    im_heat = axes[0, 1].imshow(prob_masked, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title("2. Water Probability Heatmap (Inside ROI)", fontsize=11, fontweight='bold', pad=6)
    axes[0, 1].axis('off')
    plt.colorbar(im_heat, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(panel3_rgb)
    axes[1, 0].set_title(f"3. Connected Flood Mask (Area: {flood_area_pct:.2f}%)", fontsize=11, fontweight='bold', pad=6)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(panel4_rgb)
    axes[1, 1].set_title(f"4. Water Depth Gauges & Alert Dashboard ({fused_depth:.1f} cm)", fontsize=11, fontweight='bold', pad=6)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu Dashboard phân tích độ sâu tại: {output_path}")

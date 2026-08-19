#!/usr/bin/env python3
"""
FloodVision - Phân đoạn CHUYÊN BIỆT MẶT NƯỚC (Water Surface Segmentation)
1. Sử dụng Zero-Shot Text-Prompt Segmentation (CLIPSeg).
2. Hỗ trợ 3 chế độ:
   - 'contrast' (Khuyên dùng): Đối sánh tương phản giữa Vùng Nước (Water/Puddle) và Mặt Đường Khô (Dry Road)
     giúp phát hiện tốt từ vũng nước nông đến nước lũ đục, đồng thời triệt tiêu False Positive trên đường khô.
   - 'ensemble': Kết hợp xác suất cực đại từ nhiều từ khóa nước (puddle, flood water, muddy water...).
   - 'single': Phân đoạn theo một prompt duy nhất.
3. Xuất kết quả trực quan hóa 3 panel và file mặt nạ nhị phân (Binary Mask: 0 = Nền, 255 = Mặt nước).
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt


DEFAULT_WATER_PROMPTS = [
    "puddle",
    "water puddle",
    "water on the road",
    "flooded street",
    "flood water",
    "muddy water",
    "standing water",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "dry road",
    "dry asphalt",
    "dry pavement",
]


def segment_water_zero_shot(
    image_path: str,
    prompt: str = "puddle, water on the road, flood water, muddy water",
    negative_prompt: str = "dry road, dry asphalt, dry pavement",
    threshold: float = 0.40,
    mode: str = "contrast",
    device: str = "cuda",
):
    """
    Sử dụng CLIPSeg để phân đoạn vùng nước với các cơ chế tối ưu (Contrast / Ensemble / Single).
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    print(f"🔄 Đang tải mô hình Zero-shot Water Segmenter (CLIPSeg)...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    model.eval()

    orig_img_bgr = cv2.imread(image_path)
    if orig_img_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    orig_h, orig_w = orig_img_bgr.shape[:2]
    img_pil = Image.open(image_path).convert("RGB")

    print(f"🎯 Chế độ phân đoạn (Mode): {mode.upper()}")
    print(f"⚙️ Ngưỡng nhận diện (Threshold): {threshold}")

    with torch.no_grad():
        if mode == "contrast":
            # Đối sánh 2 lớp: Nước vs Đường khô
            pos_text = prompt
            neg_text = negative_prompt
            print(f"💧 Positive Prompt: '{pos_text}'")
            print(f"🛣️ Negative Prompt: '{neg_text}'")

            inputs = processor(
                text=[pos_text, neg_text],
                images=[img_pil, img_pil],
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # logits: [2, H, W] -> channel 0: water, channel 1: dry road
            water_logit = outputs.logits[0]
            dry_logit = outputs.logits[1]

            # Softmax đối sánh tương đối giữa nước và đường khô
            two_class_logits = torch.stack([dry_logit, water_logit], dim=0)  # [2, H, W]
            water_prob = torch.softmax(two_class_logits, dim=0)[1]  # [H, W]

            prob_map_resized = torch.nn.functional.interpolate(
                water_prob.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

        elif mode == "ensemble":
            # Tập hợp các prompt nước phổ biến và lấy Max probability
            prompts = [p.strip() for p in prompt.split(",") if p.strip()]
            if not prompts:
                prompts = DEFAULT_WATER_PROMPTS
            print(f"💧 Ensemble Prompts: {prompts}")

            inputs = processor(
                text=prompts,
                images=[img_pil] * len(prompts),
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)  # [N, H, W]
            max_prob = torch.max(probs, dim=0).values  # [H, W]

            prob_map_resized = torch.nn.functional.interpolate(
                max_prob.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

        else:  # single
            print(f"💧 Single Prompt: '{prompt}'")
            inputs = processor(
                text=[prompt],
                images=[img_pil],
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            prob_map = torch.sigmoid(outputs.logits)  # [1, H, W]
            prob_map_resized = torch.nn.functional.interpolate(
                prob_map.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

    # Nhị phân hóa mặt nạ nước (1 = Water, 0 = Non-water)
    binary_mask = (prob_map_resized >= threshold).astype(np.uint8)

    # Khử nhiễu nhỏ bằng phép toán hình thái học (Morphological opening/closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Tính tỉ lệ % diện tích mặt nước
    water_pixels = np.sum(binary_mask == 1)
    total_pixels = orig_h * orig_w
    water_pct = (water_pixels / total_pixels) * 100.0

    return orig_img_bgr, prob_map_resized, binary_mask, water_pct


def save_water_visualization(orig_bgr, prob_map, binary_mask, water_pct, output_path):
    """Trực quan hóa bề mặt nước: Heatmap xác suất, Mặt nạ nhị phân, Overlay ranh giới mực nước"""
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = binary_mask.shape

    # 1. Overlay màu xanh Cyan lên vùng nước + vẽ viền ranh giới mực nước
    overlay = orig_bgr.copy()
    mask_idx = (binary_mask == 1)

    if np.any(mask_idx):
        water_color = np.zeros_like(orig_bgr)
        water_color[mask_idx] = [255, 180, 0]  # BGR: Xanh dương sáng (Cyan-Blue)
        alpha = 0.45
        blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, water_color, alpha, 0)
        overlay[mask_idx] = blended[mask_idx]

        # Vẽ đường viền (contour) ranh giới mực nước (màu vàng)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 2. Tạo hình ảnh 3 panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Ảnh Gốc (Original Image)", fontsize=12, fontweight='bold', pad=8)
    axes[0].axis('off')

    # Heatmap xác suất bề mặt nước
    im_heat = axes[1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("2. Bản Đồ Xác Suất Mặt Nước (Water Heatmap)", fontsize=12, fontweight='bold', pad=8)
    axes[1].axis('off')
    plt.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay + Viền ranh giới ngập
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"3. Vùng Mặt Nước (Diện tích ngập: {water_pct:.2f}%)", fontsize=12, fontweight='bold', pad=8)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Đã lưu kết quả trực quan hóa tại: {output_path}")

    # Đồng thời lưu riêng file binary mask nhị phân (0 và 255) phục vụ huấn luyện/xử lý tiếp
    mask_only_path = output_path.replace('.png', '_binary_mask.png')
    cv2.imwrite(mask_only_path, binary_mask * 255)
    print(f"💾 Đã lưu file mặt nạ nhị phân (Binary Mask) tại: {mask_only_path}")


def main():
    parser = argparse.ArgumentParser(description="FloodVision - Phân đoạn bề mặt nước (Water Surface Segmentation)")
    parser.add_argument(
        "--image", 
        type=str, 
        default="/home/dekii2275/FloodVision/data/ngapmuc2/ngapnang.png",
        help="Đường dẫn ảnh đầu vào"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["contrast", "ensemble", "single"],
        default="contrast",
        help="Chế độ phân đoạn: 'contrast' (đối sánh Nước vs Đường khô - khuyên dùng), 'ensemble' (max prob nhiều từ khóa), 'single' (1 prompt)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="puddle, water puddle, water on the road, flooded street, flood water, muddy water",
        help="Từ khóa văn bản mô tả mặt nước"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="dry road, dry asphalt, dry pavement",
        help="Từ khóa mô tả nền đường khô (chỉ dùng cho mode 'contrast')"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.45,
        help="Ngưỡng xác suất nhận diện mặt nước (Mặc định: 0.45 cho contrast mode, 0.25-0.35 cho ensemble/single mode)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Thư mục lưu kết quả (mặc định: cùng thư mục với ảnh đầu vào)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị: cuda hoặc cpu"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy ảnh: {args.image}")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    out_vis = os.path.join(output_dir, f"{base_name}_water_surface_seg.png")

    orig_bgr, prob_map, binary_mask, water_pct = segment_water_zero_shot(
        args.image, 
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        threshold=args.threshold, 
        mode=args.mode,
        device=args.device,
    )

    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ PHÂN ĐOẠN MẶT NƯỚC (WATER SURFACE)")
    print("="*50)
    print(f"💧 Tỉ lệ diện tích mặt nước: {water_pct:.2f}%")
    print(f"📐 Kích thước ảnh: {binary_mask.shape[1]} x {binary_mask.shape[0]} px")
    print("="*50)

    save_water_visualization(orig_bgr, prob_map, binary_mask, water_pct, out_vis)


if __name__ == '__main__':
    main()

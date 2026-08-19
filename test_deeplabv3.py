#!/usr/bin/env python3
"""
FloodVision - Segmentation Evaluation Script (DeepLabV3 / Scene Parsing)
Cho phép chạy thử các mô hình phân đoạn ngữ nghĩa (DeepLabV3, Scene Parsing ADE20K)
trên bất kỳ ảnh nào mà không cần huấn luyện lại.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# COCO VOC 21 Classes (Dùng cho DeepLabV3 torchvision)
COCO_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Tạo bảng màu cố định trực quan
def get_fixed_colormap():
    np.random.seed(123)
    colors = np.random.randint(40, 240, size=(256, 3), dtype=np.uint8)
    colors[0] = [20, 20, 20]      # Background -> Đen xám
    # Gán một số màu đặc trưng dễ nhìn
    colors[1] = [180, 100, 100]   # Building (nâu hồng)
    colors[6] = [220, 120, 60]    # Road (xám cam)
    colors[11] = [240, 180, 50]   # Sidewalk (vàng cam)
    colors[12] = [255, 60, 60]    # Person (đỏ tươi)
    colors[14] = [255, 140, 0]    # Motorbike (cam)
    colors[15] = [255, 50, 50]    # Person in COCO (đỏ)
    colors[21] = [0, 180, 255]    # Water (xanh dương sáng)
    colors[26] = [0, 220, 220]    # Sea/Lake (xanh cyan)
    return colors

COLOR_PALETTE = get_fixed_colormap()


def load_deeplabv3_torchvision(backbone='resnet50', device='cuda'):
    """Tải mô hình DeepLabV3 pre-trained từ torchvision (COCO 21 classes)"""
    import torchvision.models.segmentation as seg
    print(f"🔄 Đang khởi tạo Torchvision DeepLabV3 (Backbone: {backbone})...")
    if backbone == 'resnet50':
        weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT
        model = seg.deeplabv3_resnet50(weights=weights)
    elif backbone == 'resnet101':
        weights = seg.DeepLabV3_ResNet101_Weights.DEFAULT
        model = seg.deeplabv3_resnet101(weights=weights)
    elif backbone == 'mobilenet_v3':
        weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = seg.deeplabv3_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Backbone không hợp lệ: {backbone}")

    model.to(device)
    model.eval()
    return model, COCO_CLASSES


def load_ade20k_model(model_name='nvidia/segformer-b0-finetuned-ade-512-512', device='cuda'):
    """Tải mô hình Scene Parsing ADE20K (150 classes gồm water, road, sidewalk, building, v.v.)"""
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    print(f"🔄 Đang tải mô hình ADE20k ({model_name})...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    classes = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return (model, processor), classes


def infer_torchvision_deeplabv3(model, img_pil, device='cuda'):
    """Chạy inference cho torchvision DeepLabV3"""
    from torchvision import transforms
    orig_w, orig_h = img_pil.size
    
    # Scale ảnh nếu kích thước quá lớn để tiết kiệm bộ nhớ GPU
    max_dim = 1280
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img_pil.resize((w, h), Image.BILINEAR)
    else:
        img_resized = img_pil

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']  # [1, 21, H, W]
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Đưa mask về kích thước ảnh gốc
    if (w, h) != (orig_w, orig_h):
        preds = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return preds


def infer_ade20k(model_tuple, img_pil, device='cuda'):
    """Chạy inference cho mô hình ADE20K"""
    model, processor = model_tuple
    orig_w, orig_h = img_pil.size
    
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, 150, H_feat, W_feat]
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # Resize mask về kích thước gốc bằng Nearest Neighbor (nhanh, chuẩn ranh giới)
    pred_mask = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def visualize_and_save(img_pil, pred_mask, class_names, output_path, model_type_name="DeepLabV3"):
    """Tạo bảng biểu đồ trực quan hóa 3 khung hình đẹp mắt và lưu ra file"""
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape
    total_pixels = h * w

    # 1. Thống kê các lớp xuất hiện
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    detected_info = []
    
    print("\n" + "="*65)
    print(f"📊 KẾT QUẢ PHÂN ĐOẠN NGỮ NGHĨA ({model_type_name})")
    print("="*65)
    print(f"{'Class ID':<10} | {'Tên Lớp (Class Name)':<26} | {'Pixel':<10} | {'Tỉ lệ (%)':<10}")
    print("-"*65)

    for cid, cnt in zip(unique_ids, counts):
        cid_int = int(cid)
        if cid_int < len(class_names):
            cname = class_names[cid_int]
        else:
            cname = f"Class_{cid_int}"
        pct = (float(cnt) / total_pixels) * 100.0
        detected_info.append((cid_int, cname, cnt, pct))
        print(f"{cid_int:<10} | {cname:<26} | {cnt:<10} | {pct:>7.2f}%")
    print("="*65)

    # 2. Tạo colored mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid_int, cname, cnt, pct in detected_info:
        color = COLOR_PALETTE[cid_int % 256]
        color_mask[pred_mask == cid_int] = color

    # 3. Tạo overlay ảnh gốc + mask (alpha blending)
    alpha = 0.45
    overlay = cv2.addWeighted(img_np, 1.0 - alpha, color_mask, alpha, 0)

    # 4. Vẽ Matplotlib 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    axes[0].imshow(img_np)
    axes[0].set_title("1. Ảnh Gốc (Input)", fontsize=12, fontweight='bold', pad=8)
    axes[0].axis('off')

    axes[1].imshow(color_mask)
    axes[1].set_title(f"2. Mặt Nạ Phân Đoạn ({model_type_name})", fontsize=12, fontweight='bold', pad=8)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("3. Phủ Mặt Nạ Lên Ảnh (Overlay)", fontsize=12, fontweight='bold', pad=8)
    axes[2].axis('off')

    # Thêm Legend cho các class phát hiện được (> 0.2% diện tích)
    legend_patches = []
    # Sắp xếp các class theo tỉ lệ % giảm dần
    sorted_detected = sorted(detected_info, key=lambda x: x[3], reverse=True)
    for cid_int, cname, cnt, pct in sorted_detected:
        if pct >= 0.2:  # Bỏ qua các class nhiễu quá nhỏ
            rgb_norm = COLOR_PALETTE[cid_int % 256] / 255.0
            legend_patches.append(mpatches.Patch(color=rgb_norm, label=f"{cname} ({pct:.1f}%)"))
    
    if legend_patches:
        axes[2].legend(
            handles=legend_patches,
            bbox_to_anchor=(1.03, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=9,
            title="Đối tượng nhận diện",
            title_fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Đã lưu ảnh trực quan hóa tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FloodVision - Test Segmentation Model Pre-trained")
    parser.add_argument(
        "--image", 
        type=str, 
        default="/home/dekii2275/FloodVision/data/ngapmuc2/ngapnang.png",
        help="Đường dẫn tới ảnh đầu vào bất kỳ"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['deeplabv3', 'ade20k', 'both'], 
        default='both',
        help="Chọn mô hình: 'deeplabv3' (COCO objects), 'ade20k' (Scene/Water parsing), hoặc 'both' để so sánh cả 2"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='resnet50',
        choices=['resnet50', 'resnet101', 'mobilenet_v3'],
        help="Backbone cho DeepLabV3"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/dekii2275/FloodVision/data/ngapmuc2",
        help="Thư mục lưu ảnh kết quả"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán: cuda hoặc cpu"
    )
    args = parser.parse_args()

    # Kiểm tra ảnh đầu vào
    if not os.path.exists(args.image):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại '{args.image}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    print(f"\n🖼️ Đang xử lý ảnh: {args.image}")
    img_pil = Image.open(args.image).convert("RGB")
    print(f"📐 Kích thước ảnh: {img_pil.size[0]} x {img_pil.size[1]} pixels")
    print(f"⚡ Thiết bị sử dụng: {args.device.upper()}")

    # 1. Chạy DeepLabV3 (COCO Objects: Xe cộ, Người, v.v.)
    if args.model in ['deeplabv3', 'both']:
        try:
            model_deep, classes_deep = load_deeplabv3_torchvision(backbone=args.backbone, device=args.device)
            mask_deep = infer_torchvision_deeplabv3(model_deep, img_pil, device=args.device)
            out_deep = os.path.join(args.output_dir, f"{base_name}_deeplabv3_{args.backbone}_eval.png")
            visualize_and_save(img_pil, mask_deep, classes_deep, out_deep, model_type_name=f"DeepLabV3-{args.backbone}")
        except Exception as e:
            print(f"❌ Lỗi khi chạy DeepLabV3: {e}")
            import traceback; traceback.print_exc()

    # 2. Chạy ADE20K Scene Parsing (Nhận diện Water, Road, Sidewalk, v.v.)
    if args.model in ['ade20k', 'both']:
        try:
            model_ade, classes_ade = load_ade20k_model(device=args.device)
            mask_ade = infer_ade20k(model_ade, img_pil, device=args.device)
            out_ade = os.path.join(args.output_dir, f"{base_name}_ade20k_scene_eval.png")
            visualize_and_save(img_pil, mask_ade, classes_ade, out_ade, model_type_name="ADE20K-SceneParsing")
        except Exception as e:
            print(f"❌ Lỗi khi chạy ADE20K: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()

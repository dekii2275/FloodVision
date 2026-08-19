# 🐳 Hướng Dẫn Triển Khai & Đóng Gói Docker (FloodVision AI)

Hệ thống FloodVision đã được đóng gói toàn diện bao gồm:
- **Core SDK (`FloodVisionPipeline`):** Thư viện Python tái sử dụng cho các dự án khác, luồng xử lý video RTSP hoặc Edge AI.
- **Backend API (FastAPI):** Cung cấp RESTful API phân đoạn và ước lượng độ sâu ngập.
- **Frontend Dashboard:** Giao diện trực quan hóa, cấu hình mốc $G_1, G_2, G_3$ và đo lường tức thì.
- **Docker & Docker Compose:** Đóng gói trọn gói môi trường CUDA, PyTorch, CLIPSeg và OpenCV.

---

## 1. Khởi động nhanh với Docker Compose (Khuyên dùng)

### Yêu cầu tiên quyết:
- Đã cài đặt [Docker](https://docs.docker.com/engine/install/) & [Docker Compose](https://docs.docker.com/compose/install/).
- Đã cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (để tận dụng GPU).

### Lệnh chạy:

```bash
# 1. Build và khởi chạy ngầm toàn bộ hệ thống
docker compose up --build -d

# 2. Xem logs hoạt động
docker compose logs -f

# 3. Dừng hệ thống khi không sử dụng
docker compose down
```

Sau khi khởi động, truy cập giao diện tại: 👉 **`http://localhost:8000`**

---

## 2. Khởi động với Standalone Docker

```bash
# 1. Build Docker Image
docker build -t floodvision:latest .

# 2. Chạy Container (với GPU)
docker run -d \
  --name floodvision-app \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/configs:/app/configs \
  -v floodvision_hf_cache:/root/.cache/huggingface \
  floodvision:latest

# 3. Chạy Container (chỉ CPU nếu không có GPU)
docker run -d \
  --name floodvision-app \
  -e DEVICE=cpu \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  floodvision:latest
```

---

## 3. Hướng dẫn Tái sử dụng Core Pipeline SDK trong Python

Bạn có thể nhập trực tiếp `FloodVisionPipeline` vào bất kỳ module hoặc ứng dụng Python nào khác:

```python
from pathlib import Path
from src.flood_depth import FloodVisionPipeline, PipelineConfig

# 1. Khởi tạo Pipeline từ file cấu hình Camera
pipeline = FloodVisionPipeline.from_config(
    config_path="configs/camera_gauges_default.yaml",
    device="cuda",  # hoặc "cpu"
)

# 2. Phân tích 1 bức ảnh (hỗ trợ Path, numpy BGR/RGB hoặc PIL Image)
output, img_bgr = pipeline.predict(
    image="data/ngapmuc2/ngapnang.png",
    threshold=0.30,
)

# 3. Đọc các số liệu đo đạc chuẩn xác
print(f"🌊 Diện tích ngập: {output.flood_area_pct:.1f}%")
print(f"📏 Độ sâu hợp nhất (Fused Depth): {output.fused_depth_cm:.1f} cm")
print(f"🚨 Cấp độ cảnh báo: [{output.level.level_code}] {output.level.level_name}")

# 4. Chi tiết từng cột mốc
for g in output.measurements:
    print(f"  • {g.gauge_id} ({g.gauge_name}): {g.local_depth_cm:.1f} cm (Trạng thái: {g.status.value})")

# 5. Xuất ảnh Dashboard 4 khung hình
output.save_dashboard(img_bgr, "outputs/my_dashboard_result.png")
```

---

## 4. Cấu trúc RESTful API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/` | Giao diện Web tương tác (Frontend) |
| `GET` | `/api/health` | Kiểm tra tình trạng server & GPU |
| `GET` | `/api/sample_images` | Lấy danh sách ảnh mẫu kiểm thử |
| `GET` | `/api/image_raw?path=...` | Tải file ảnh tĩnh |
| `POST` | `/api/upload_image` | Tải ảnh mới lên server |
| `POST` | `/api/analyze` | Gửi yêu cầu phân đoạn & đo độ sâu ngập |

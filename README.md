# 🌊 FloodVision AI - Giám Sát & Ước Lượng Độ Sâu Ngập Lụt Qua Camera Giao Thông

<p align="center">
  <img src="https://img.shields.io/badge/Model-CLIPSeg%20Zero--Shot-blue?style=for-the-badge&logo=pytorch" alt="CLIPSeg">
  <img src="https://img.shields.io/badge/Architecture-Virtual%20Gauges%20%2B%20Robust%20Fusion-green?style=for-the-badge" alt="Virtual Gauges">
  <img src="https://img.shields.io/badge/Backend-FastAPI%20%2B%20Uvicorn-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready%20(CUDA%20%26%20CPU)-2496ED?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen?style=for-the-badge" alt="Tests">
</p>

---

## 📌 Giới Thiệu Tổng Quan

**FloodVision AI** là giải pháp thị giác máy tính và AI giám sát ngập lụt đô thị theo thời gian thực từ camera giao thông. Hệ thống kết hợp mô hình phân đoạn **CLIPSeg (Zero-Shot Multi-Prompt Ensemble)** cùng tầng đo lường hình học **3 Cột mốc ảo ($G_1, G_2, G_3$)**, cơ chế **Quét ranh giới mực nước kết nối chân mốc (Base-Connected Waterline)** và thuật toán **Hợp nhất Robust Median Fusion** để đưa ra độ sâu ngập chính xác (cm) và cảnh báo 5 cấp độ chuẩn (`L0` $\to$ `L4`).

```
                              KIẾN TRÚC PIPELINE
                              
Ảnh Camera Giao Thông + Tọa độ 3 Mốc (G1, G2, G3)
                        │
                        ▼
       CLIPSeg Parallel Water-Centric Prompts
     ['flood water', 'water', 'muddy water', ...]
                        │
                        ▼
             Water Probability Map
                        │
                        ▼
                Camera ROI Masking
                        │
                        ▼
      Clean Flood Mask (Morphology + CC Filter)
                        │
                        ▼
           Base-Connected Flood Component
                        │
                        ▼
     Gauge Band Multi-Ray Sampling (15 tia song song)
                        │
                        ▼
           Base-Connected Waterline Scan
                        │
                        ▼
     Multi-Point Piecewise Calibration (depth_cm)
                        │
                        ▼
      Robust Median Fusion + Outlier Rejection
     (|depth - median| > 25cm -> Loại bỏ Outlier)
                        │
                        ▼
       Độ Sâu Hợp Nhất & Cảnh Báo 5 Cấp Độ (L0 - L4)
```

---

## ✨ Tính Năng Nổi Bật

- **Phân đoạn không phụ thuộc nhãn cứng (Zero-Shot CLIPSeg):** Nhận diện chính xác cả vũng nước nông sau mưa, mặt đường ngập đục phù sa lẫn lũ lụt diện rộng.
- **Triệt tiêu báo ngập giả trên đường khô:** Sử dụng bộ từ khóa chuyên biệt mặt nước (Water-Centric) loại bỏ đặc trưng mặt đường nhựa.
- **Dải đo đa tia (Gauge Band 15 Rays):** Lấy mẫu 15 tia song song trên bề rộng thước, tính Median Waterline để triệt tiêu nhiễu pixel cục bộ.
- **Quét ranh giới mực nước từ chân mốc (Base-Connected Scan):** Chỉ đo vùng nước liên tục kết nối từ chân mốc đi lên, bỏ qua hoàn toàn các đốm nước ảo ở trên cao.
- **Hiệu chuẩn đa điểm phi tuyến (Piecewise Linear Calibration):** Khắc phục sai số do góc nhìn phối cảnh camera nghiêng/xa/gần.
- **Hợp nhất chống nhiễu (Robust Median Fusion & Outlier Rejection):** Tự động phát hiện và loại bỏ mốc bị đo lệch bất thường ($|d_i - \text{median}| > 25\text{ cm}$).
- **Giao diện Web tương tác trực quan:** Kéo thả định vị mốc $G_1, G_2, G_3$, tùy chỉnh chiều cao, độ nhạy và xem kết quả Dashboard 4 khung hình tức thì.
- **Đóng gói Docker trọn gói:** Sẵn sàng triển khai nhanh cho production với GPU NVIDIA hoặc CPU.

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
FloodVision/
├── app/                              # Ứng dụng Web & Giao diện tương tác
│   ├── server.py                     # FastAPI Backend Server
│   └── static/                       # Frontend HTML / CSS / JS
│       ├── index.html                # Giao diện chính
│       ├── style.css                 # Dark-mode Glassmorphism styling
│       └── app.js                    # Canvas kéo thả mốc & gọi API
├── configs/
│   └── camera_gauges_default.yaml    # Cấu hình ROI, 3 Mốc ảo, Ngưỡng lọc & Fusion
├── src/
│   ├── common/
│   │   └── io.py                     # Xử lý đọc / ghi ảnh
│   └── flood_depth/                  # Core SDK đo độ sâu ngập
│       ├── pipeline.py               # FloodVisionPipeline SDK chính
│       ├── gauge_geometry.py         # Hình học cột mốc, vector & dải đo đa tia
│       ├── gauge_sampling.py         # Quét Base-Connected Waterline
│       ├── gauge_validation.py       # Định nghĩa trạng thái mốc (VALID, OUTLIER,...)
│       ├── calibration.py            # Hiệu chuẩn đa điểm phi tuyến tính
│       ├── fusion.py                 # Robust Median Fusion & Phân loại L0-L4
│       ├── roi_filter.py             # Lọc ROI và Connected Components
│       └── visualizer.py             # Sinh ảnh Dashboard 4 khung hình
├── scripts/
│   ├── run_app.py                    # Script khởi chạy Web Server (tự giải phóng port)
│   └── run_water_gauge_depth_baseline.py # CLI chạy kiểm thử ảnh đơn / batch thư mục
├── tests/
│   └── test_water_gauge_depth.py     # Bộ 7 Unit Tests (Cases A, B, C, D, E, F, G)
├── Dockerfile                        # File đóng gói Container production
├── docker-compose.yml                # File cấu hình Docker Compose (hỗ trợ GPU)
├── requirements.txt                  # Danh mục thư viện Python cần thiết
└── DOCKER_GUIDE.md                   # Hướng dẫn chi tiết triển khai Docker
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### Cách 1: Chạy trực tiếp trên Web UI (Python / Conda)

```bash
# 1. Kích hoạt môi trường
conda activate floodvision

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi động Web Server (mặc định mở tại port 8000)
python scripts/run_app.py --port 8000
```

Truy cập giao diện Web tại: 👉 **`http://localhost:8000`**

---

### Cách 2: Triển khai với Docker Compose (Khuyên dùng cho Production)

```bash
# 1. Build và khởi chạy ngầm hệ thống
docker compose up --build -d

# 2. Xem logs hoạt động
docker compose logs -f

# 3. Dừng hệ thống khi cần
docker compose down
```

---

### Cách 3: Chạy Baseline CLI trên Terminal

#### Phân tích 1 bức ảnh:
```bash
python scripts/run_water_gauge_depth_baseline.py \
  --image data/ngapmuc2/ngapnang.png \
  --output-dir outputs/demo
```

#### Phân tích hàng loạt ảnh trong một thư mục:
```bash
python scripts/run_water_gauge_depth_baseline.py \
  --input-dir data/ngapmuc2 \
  --output-dir outputs/batch_eval
```

---

## 💻 Hướng Dẫn Tái Sử Dụng Python SDK Trong Dự Án Khác

Bạn có thể nhập trực tiếp `FloodVisionPipeline` vào script hoặc luồng phân tích camera khác:

```python
from src.flood_depth import FloodVisionPipeline

# 1. Khởi tạo Pipeline (tự động nạp CLIPSeg và cấu hình camera)
pipeline = FloodVisionPipeline.from_config("configs/camera_gauges_default.yaml")

# 2. Dự đoán độ sâu ngập trên ảnh bất kỳ (Path, numpy BGR/RGB, hoặc PIL Image)
output, img_bgr = pipeline.predict("data/ngapmuc2/ngapnang.png", threshold=0.30)

# 3. Đọc các kết quả đo lường
print(f"🌊 Diện tích ngập: {output.flood_area_pct:.1f}%")
print(f"📏 Độ sâu hợp nhất (Fused Depth): {output.fused_depth_cm:.1f} cm")
print(f"🚨 Cấp độ cảnh báo: [{output.level.level_code}] {output.level.level_name}")

# 4. Trạng thái đo chi tiết từng cột mốc
for g in output.measurements:
    print(f"  • {g.gauge_id} ({g.gauge_name}): {g.local_depth_cm:.1f} cm | Trạng thái: {g.status.value}")

# 5. Lưu ảnh Dashboard 4 khung hình
output.save_dashboard(img_bgr, "outputs/dashboard_result.png")
```

---

## 📊 Bảng 5 Cấp Độ Cảnh Báo Ngập Chuẩn

| Cấp độ | Tên gọi | Dải độ sâu | Ý nghĩa & Cảnh báo an toàn | Màu cảnh báo |
|:---:|---|:---:|---|:---:|
| **`L0`** | **Không ngập** | $0\text{ cm}$ | Mặt đường khô ráo hoặc ướt nhẹ sau mưa. Xe cộ lưu thông an toàn tuyệt đối. | 🟢 `#10b981` |
| **`L1`** | **Ngập nông** | $0 – 20\text{ cm}$ | Mực nước mấp mé mép lốp xe. Xe máy và ô tô con có thể di chuyển chậm qua vùng ngập. | 🟡 `#84cc16` |
| **`L2`** | **Ngập vừa** | $20 – 50\text{ cm}$ | Nước ngập nửa bánh xe, gần miệng ống xả xe máy. Khuyến cáo xe gầm thấp không đi qua. | 🟠 `#f59e0b` |
| **`L3`** | **Ngập sâu** | $50 – 100\text{ cm}$ | Nước ngập qua yên xe máy / nắp capo ô tô con. Nguy cơ thủy kích phá hủy động cơ cực cao. Cấm xe máy và ô tô con. | 🔴 `#ef4444` |
| **`L4`** | **Ngập nghiêm trọng** | $> 100\text{ cm}$ | Nước lũ dâng cao lút nóc xe, dòng chảy xiết nguy hiểm đến tính mạng. Cấm tất cả phương tiện. | 🟣 `#9333ea` |

---

## 🧪 Chạy Kiểm Thử Tự Động (Unit Tests)

Bộ kiểm thử đảm bảo hệ thống vượt qua tất cả các ca kiểm thử đo lường khó (Outlier Rejection, False Blob Ignored, Dry Base, Multi-Point Calibration):

```bash
PYTHONPATH=. pytest tests/test_water_gauge_depth.py -v
```

Kết quả: **`7/7 unit tests PASSED (100%)`**.

---

## 📄 Giấy Phép & Bản Quyền

Dự án được phát triển phục vụ mục đích nghiên cứu và triển khai hệ thống giao thông thông minh (ITS) cảnh báo ngập lụt đô thị.

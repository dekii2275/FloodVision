# ==============================================================================
# FloodVision AI - Production Docker Container
# CLIPSeg Zero-Shot Flood Segmentation & Virtual Gauges Depth Perception System
# ==============================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Thiết lập môi trường
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PORT=8000 \
    DEVICE=cuda

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và hình ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt Python Dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào Container
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Tạo các thư mục lưu trữ dữ liệu và output
RUN mkdir -p /app/data /app/outputs/web_runs/uploads /app/outputs/web_runs/results /root/.cache/huggingface

# Mở cổng dịch vụ
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Khởi động Uvicorn Web Server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

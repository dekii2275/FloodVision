#!/usr/bin/env python3
"""
FloodVision - Launcher cho Ứng Dụng Web Hiệu Chuẩn Mốc & Ước Lượng Độ Sâu Ngập Lụt

Sử dụng:
    python scripts/run_app.py --port 8000
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(port: int) -> None:
    try:
        # Tìm và tắt tiến trình cũ đang chiếm cổng
        cmd = f"lsof -ti:{port} | xargs -r kill -9"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="FloodVision Interactive Web App Launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng chạy ứng dụng (mặc định: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()

    port = args.port

    # Tự động giải phóng cổng nếu đang bị chiếm dụng
    if is_port_in_use(port):
        print(f"⚠️ Cổng {port} đang bận. Đang giải phóng tiến trình cũ...")
        free_port(port)

    import uvicorn

    url = f"http://localhost:{port}"
    print("\n" + "=" * 65)
    print("🌊 FLOODVISION INTERACTIVE CALIBRATION & DEPTH ESTIMATOR")
    print("=" * 65)
    print(f"🚀 Máy chủ đang khởi động tại: {url}")
    print("💡 Mở trình duyệt và truy cập đường dẫn trên để cấu hình mốc & phân tích.")
    print("=" * 65 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("app.server:app", host=args.host, port=port, reload=False)


if __name__ == "__main__":
    main()

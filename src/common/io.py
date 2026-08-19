from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_image(path: str | Path) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path_obj}")
    # Đọc ảnh hỗ trợ cả ký tự tiếng Việt / unicode
    img = cv2.imdecode(np.fromfile(str(path_obj), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path_obj}")
    return img


def write_image(path: str | Path, image: np.ndarray) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix.lower()
    if not ext:
        ext = ".png"
        path_obj = path_obj.with_suffix(ext)
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Could not encode image for writing to {path_obj}")
    with path_obj.open("wb") as handle:
        handle.write(encoded.tobytes())


def list_images(directory: str | Path) -> list[Path]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    return sorted(
        path for path in dir_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def safe_stem(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_").replace(".", "_")

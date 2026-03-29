"""
Download the MediaPipe Pose Landmarker model file.

Usage:
    cd backend
    python scripts/download_model.py

Downloads pose_landmarker_lite.task into backend/models/.
Switch MODEL_VARIANT below to 'full' or 'heavy' for higher accuracy
(larger file, slower inference).
"""

import os
import urllib.request

MODEL_VARIANT = "full"  # options: lite | full | heavy
MODEL_FILENAME = f"pose_landmarker_{MODEL_VARIANT}.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    f"pose_landmarker/pose_landmarker_{MODEL_VARIANT}/float16/latest/"
    f"{MODEL_FILENAME}"
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEST_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(DEST_PATH):
        print(f"Model already exists: {DEST_PATH}")
        return

    print(f"Downloading {MODEL_FILENAME} from:\n  {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, DEST_PATH, _progress)
    print(f"\nSaved to: {DEST_PATH}")


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  {pct:.1f}%  ({downloaded // 1024} / {total_size // 1024} KB)", end="", flush=True)


if __name__ == "__main__":
    main()

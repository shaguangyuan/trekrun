"""
Minimal validation script for cloud/local pose extraction.

Usage:
  python scripts/validate_pose_extract.py --video /path/to/test.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from app.services.pose_extractor import extract_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a short test video.")
    parser.add_argument(
        "--output-dir",
        default="./uploads",
        help="Directory for intermediate extraction outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"FAIL: video not found: {args.video}")
        return 1

    video_id = f"validate-{uuid.uuid4()}"
    res = extract_landmarks(
        video_path=args.video,
        video_id=video_id,
        output_dir=args.output_dir,
    )
    if not res.success:
        print(f"FAIL: pose_extract failed: {res.error}")
        return 2

    print("OK: pose_extract succeeded")
    print(f"video_id={video_id}")
    print(f"frames={res.total_frames} frames_with_pose={res.frames_with_pose}")
    print(f"pose_ratio={res.pose_ratio:.3f}")
    print(f"delegate={res.pose_runtime.get('delegate_used')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

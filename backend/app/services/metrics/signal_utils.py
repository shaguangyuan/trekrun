"""Lightweight signal helpers (no scipy)."""

from __future__ import annotations

from typing import List

import numpy as np


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.astype(float).copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def local_minima_indices(y: List[float] | np.ndarray) -> List[int]:
    arr = np.asarray(y, dtype=float)
    if len(arr) < 3:
        return []
    out: List[int] = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            out.append(i)
    return out


def pick_strikes_with_min_separation(indices: List[int], min_sep_frames: int) -> List[int]:
    if not indices:
        return []
    kept: List[int] = []
    last = -10**9
    for i in sorted(indices):
        if i - last >= min_sep_frames:
            kept.append(i)
            last = i
    return kept


def intervals_from_indices(indices: List[int], fps: float) -> List[float]:
    """Convert sorted frame indices to inter-event intervals in seconds."""
    if len(indices) < 2:
        return []
    fps = max(fps, 1e-6)
    return [(indices[j] - indices[j - 1]) / fps for j in range(1, len(indices))]

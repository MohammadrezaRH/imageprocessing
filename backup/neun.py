from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage import color, io, filters, measure, util

__all__ = [
    "load_image",
    "preprocess",
    "entropy_threshold",
    "segment",
    "filter_regions",
    "analyze_image",
    "batch_analyze",
]


# -----------------------------
# Constants mirroring MATLAB settings
# -----------------------------

GAUSSIAN_SIGMA: float = 2.0  # fspecial('gaussian', [5 5], 2)
AREA_MIN: int = 7
AREA_MAX: int = 40_000
THRESHOLD_SCALE: float = 0.81  # NeuN script scales entropy threshold by 0.81


# -----------------------------
# I/O helpers
# -----------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load *path* and convert to grayscale float32 in \[0, 1]."""
    img = io.imread(str(path))
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return util.img_as_float32(img)


# -----------------------------
# Pre-processing
# -----------------------------

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian smoothing exactly like the MATLAB 5×5, σ=2 kernel."""
    return filters.gaussian(gray, sigma=GAUSSIAN_SIGMA, truncate=1.0)


# -----------------------------
# Threshold
# -----------------------------

def entropy_threshold(img: np.ndarray) -> float:
    """Compute entropy-based global threshold and apply NeuN scaling."""
    counts, _ = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()

    cumulative = np.cumsum(p)
    cumulative_bg_entropy = np.cumsum(-p * np.log2(p + np.finfo(float).eps))
    fg_entropy = (
        cumulative_bg_entropy[-1] - cumulative_bg_entropy
    ) / (1 - cumulative + np.finfo(float).eps)
    total_entropy = cumulative_bg_entropy + fg_entropy
    total_entropy = np.nan_to_num(total_entropy, nan=-np.inf)

    t = int(np.argmax(total_entropy[:-1]))  # ignore last bin
    return (t / 255.0) * THRESHOLD_SCALE


# -----------------------------
# Segmentation & filtering
# -----------------------------

def segment(img: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary mask of *img* > *threshold*."""
    return img > threshold


def filter_regions(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Filter connected components by area range \[AREA_MIN, AREA_MAX].

    Returns
    -------
    Tuple[np.ndarray, int]
        final_mask, total_valid_area
    """
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    final_mask = np.zeros_like(binary, dtype=bool)
    total_area = 0

    for r in regions:
        if AREA_MIN <= r.area <= AREA_MAX:
            final_mask[labeled == r.label] = True
            total_area += r.area
    return final_mask, total_area


# -----------------------------
# High-level API
# -----------------------------

def analyze_image(path: Path | str) -> dict:
    gray = load_image(path)
    smoothed = preprocess(gray)
    thresh = entropy_threshold(smoothed)
    binary = segment(smoothed, thresh)
    final_mask, total_valid_area = filter_regions(binary)

    image_area = gray.size
    coverage_percent = (total_valid_area / image_area) * 100.0

    return {
        "filepath": str(path),
        "total_valid_area": total_valid_area,
        "coverage_percent": coverage_percent,
    }


def batch_analyze(paths: List[str | Path], output: Path | None = None) -> pd.DataFrame:
    records = [analyze_image(p) for p in paths]
    df = pd.DataFrame.from_records(records)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
    return df


# -----------------------------
# CLI
# -----------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuN analysis — Python port of MATLAB NeuN.m",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more image paths to analyze.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Optional Parquet filepath to save the results.",
    )
    args = parser.parse_args()

    df = batch_analyze(args.paths, args.out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    _main() 
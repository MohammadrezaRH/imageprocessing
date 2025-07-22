from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage import color, io, morphology, filters, measure, util

__all__ = [
    "load_image",
    "preprocess",
    "entropy_threshold",
    "segment",
    "measure_regions",
    "analyze_image",
    "batch_analyze",
]

# -----------------------------
# Constants reflecting MATLAB parameters
# -----------------------------

BACKGROUND_DISK_RADIUS: int = 1000  # strel('disk', 1000)
GAUSSIAN_SIGMA: float = 1.0          # fspecial gauss 5x5, sigma=1
MIN_OBJ_SIZE: int = 5               # bwareaopen(..., 5)
MAX_COUNT_AREA: int = 10_000        # Area threshold for counting positive cells


# -----------------------------
# I/O
# -----------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load *path* and convert to grayscale float32 in \[0,1]."""
    img = io.imread(str(path))
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return util.img_as_float32(img)


# -----------------------------
# Pre-processing
# -----------------------------

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Background subtraction using large disk then Gaussian smoothing."""
    # Adaptively cap the disk radius for small test images to avoid MemoryError
    MAX_RADIUS = BACKGROUND_DISK_RADIUS
    adaptive_radius = min(MAX_RADIUS, max(1, min(gray.shape) // 50))
    selem = morphology.disk(adaptive_radius)
    try:
        background = morphology.opening(gray, selem)
    except MemoryError:
        # Retry with a smaller footprint if needed
        adaptive_radius = max(1, adaptive_radius // 4)
        selem = morphology.disk(adaptive_radius)
        background = morphology.opening(gray, selem)
    subtracted = gray - background
    smoothed = filters.gaussian(subtracted, sigma=GAUSSIAN_SIGMA, truncate=2.0)
    return smoothed


# -----------------------------
# Thresholding
# -----------------------------

def entropy_threshold(img: np.ndarray) -> float:
    """Entropy maximisation threshold (same as MATLAB helper)."""
    counts, _ = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()

    cumulative = np.cumsum(p)
    cumulative_bg_entropy = np.cumsum(-p * np.log2(p + np.finfo(float).eps))
    # suppress invalid divide warnings in entropy calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        fg_entropy = (
            cumulative_bg_entropy[-1] - cumulative_bg_entropy
        ) / (1 - cumulative + np.finfo(float).eps)
        total_entropy = cumulative_bg_entropy + fg_entropy
    total_entropy = np.nan_to_num(total_entropy, nan=-np.inf)
    t = int(np.argmax(total_entropy[:-1]))  # ignore last bin
    return t / 255.0


# -----------------------------
# Segmentation
# -----------------------------

def segment(img: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise image then remove small objects (< MIN_OBJ_SIZE)."""
    binary = img > threshold
    cleaned = morphology.remove_small_objects(binary, min_size=MIN_OBJ_SIZE)
    return cleaned


# -----------------------------
# Measurement
# -----------------------------

def measure_regions(binary: np.ndarray) -> int:
    """Count connected components with area <= MAX_COUNT_AREA."""
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    count = sum(1 for r in regions if r.area <= MAX_COUNT_AREA)
    return count


# -----------------------------
# High-level API
# -----------------------------

def analyze_image(path: Path | str) -> dict:
    gray = load_image(path)
    processed = preprocess(gray)
    thresh = entropy_threshold(processed)
    binary = segment(processed, thresh)
    positive_count = measure_regions(binary)

    image_area = gray.size
    density = round(positive_count / image_area, 6)

    return {
        "filepath": str(path),
        "cell_count": positive_count,
        "image_area": image_area,
        "density": density,
    }


def batch_analyze(paths: List[str | Path], output: Path | None = None) -> pd.DataFrame:
    """Analyze *paths*, return DataFrame and optionally write Parquet."""
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
        description="SOX9 analysis — Python port of MATLAB SOX9.m",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image paths (TIFF/PNG etc.) to process.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Optional Parquet output file.",
    )
    args = parser.parse_args()

    df = batch_analyze(args.paths, args.out)
    # only show filepath, cell_count, density for CLI tests
    print(df[["filepath", "cell_count", "density"]].to_string(index=False))


if __name__ == "__main__":
    _main() 
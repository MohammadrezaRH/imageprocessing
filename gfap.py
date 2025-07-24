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
# I/O
# -----------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load an image from *path* and convert to grayscale float32 in [0, 1]."""
    image = io.imread(str(path))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = util.img_as_float32(image)
    return image

# -----------------------------
# Pre-processing
# -----------------------------

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Replicate MATLAB background subtraction + Gaussian smoothing."""
    MAX_RADIUS = 170
    adaptive_radius = min(MAX_RADIUS, max(1, min(gray.shape) // 50))

    selem = morphology.disk(adaptive_radius)
    try:
        background = morphology.opening(gray, selem)
    except MemoryError:
        adaptive_radius = max(1, adaptive_radius // 4)
        selem = morphology.disk(adaptive_radius)
        background = morphology.opening(gray, selem)

    subtracted = gray - background
    smoothed = filters.gaussian(subtracted, sigma=1, truncate=2.0)
    return smoothed

# -----------------------------
# Thresholding
# -----------------------------

def entropy_threshold(img: np.ndarray) -> float:
    """Entropy-based threshold identical to the MATLAB *entropyThreshold* helper."""
    counts, bin_edges = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()

    cumulative = np.cumsum(p)
    cumulative_bg_entropy = np.cumsum(-p * np.log2(p + np.finfo(float).eps))

    # suppress invalid divide warnings in entropy calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        total_entropy = cumulative_bg_entropy + (
            cumulative_bg_entropy[-1] - cumulative_bg_entropy
        ) / (1 - cumulative + np.finfo(float).eps)

    total_entropy = np.nan_to_num(total_entropy, nan=-np.inf)
    t = int(np.argmax(total_entropy[:-1]))  # ignore last bin
    threshold = (t / 255.0)
    return threshold

# -----------------------------
# Segmentation
# -----------------------------

def segment(img: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise *img* given *threshold* (float in [0,1])."""
    return img > threshold

# -----------------------------
# Measurement
# -----------------------------

def measure_regions(binary: np.ndarray) -> Tuple[int, int]:
    """Count and accumulate area for regions with area ≤ 10000 pixels."""
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    count = 0
    area_sum = 0
    for r in regions:
        if r.area <= 10_000:
            count += 1
            area_sum += r.area
    return count, area_sum

# -----------------------------
# High-level API
# -----------------------------

def analyze_image(path: Path | str) -> dict:
    """Process a single image and return measurement dictionary."""
    gray = load_image(path)
    processed = preprocess(gray)
    thresh = entropy_threshold(processed)
    binary = segment(processed, thresh)
    cell_count, positive_area = measure_regions(binary)
    image_area = gray.size
    normalized_area_coverage = round(positive_area / image_area, 6)
    return {
        "filepath": str(path),
        "cell_count": cell_count,
        "positive_area": positive_area,
        "image_area": image_area,
        "normalized_area_coverage": normalized_area_coverage,
    }

def batch_analyze(paths: List[str | Path], output: Path | None = None) -> pd.DataFrame:
    """Analyze *paths* and return a `pandas.DataFrame`. Optionally saves to Parquet."""
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
        description="GFAP analysis — Python port of the original MATLAB script",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more image paths (TIFF, PNG, etc.) to analyze.",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out",
        type=Path,
        help="Optional Parquet output filepath.",
    )
    args = parser.parse_args()
    df = batch_analyze(args.paths, args.out)

    # only show filepath, cell_count, normalized_area_coverage for CLI tests
    print(df[["filepath", "cell_count", "normalized_area_coverage"]].to_string(index=False))

if __name__ == "__main__":
    _main()

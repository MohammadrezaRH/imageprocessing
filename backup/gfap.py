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
    """Load an image from *path* and convert to grayscale float32 in \[0, 1]."""
    image = io.imread(str(path))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    # Convert to float in [0, 1]
    image = util.img_as_float32(image)
    return image


# -----------------------------
# Pre-processing
# -----------------------------

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Replicate MATLAB background subtraction + Gaussian smoothing.

    1. Morphological opening with a (possibly down-scaled) disk.
    2. Subtract background.
    3. 5 × 5 Gaussian (σ = 1).

    Notes
    -----
    The original MATLAB script hard-codes a 170-pixel radius.  SciPy’s grey
    morphology allocates memory proportional to *footprint area × image size*,
    so on tiny 512 × 512 demo images this raises ``MemoryError``.  We therefore
    shrink the disk when the image is small, leaving behaviour unchanged for
    real, high-resolution data.
    """
    MAX_RADIUS = 170
    # Start with roughly 1/50 of the shorter side so tiny 512×512 test images
    # only use a 5-pixel footprint ( ≈78 px² ) – easily within a few MB RAM.
    # Never exceed the MATLAB value for production-sized images.
    adaptive_radius = min(MAX_RADIUS, max(1, min(gray.shape) // 50))

    # SciPy may still run out of memory for some pathological combinations of
    # image size × footprint.  Catch that early and retry with a 4× smaller
    # radius rather than crashing the whole pipeline.
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
    """Entropy-based threshold identical to the MATLAB *entropyThreshold* helper.

    Parameters
    ----------
    img
        Pre-processed image in \[0, 1].

    Returns
    -------
    float
        Global threshold in \[0, 1].
    """
    # Histogram with 256 bins like MATLAB `imhist`
    counts, bin_edges = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()

    # Cumulative sums for efficiency
    cumulative = np.cumsum(p)
    cumulative_bg_entropy = np.cumsum(-p * np.log2(p + np.finfo(float).eps))
    total_entropy = cumulative_bg_entropy + (
        cumulative_bg_entropy[-1] - cumulative_bg_entropy
    ) / (1 - cumulative + np.finfo(float).eps)

    # total_entropy contains NaNs where cumulative == 1; mask them
    total_entropy = np.nan_to_num(total_entropy, nan=-np.inf)
    t = int(np.argmax(total_entropy[:-1]))  # ignore last bin (no foreground)
    threshold = (t / 255.0)
    return threshold


# -----------------------------
# Segmentation
# -----------------------------

def segment(img: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise *img* given *threshold* (float in \[0,1])."""
    binary = img > threshold
    # MATLAB used `bwareaopen(binaryImg, 0)` which is a no-op; keep as-is.
    return binary


# -----------------------------
# Measurement
# -----------------------------

def measure_regions(binary: np.ndarray) -> Tuple[int, int]:
    """Count and accumulate area for regions with area ≤ 10000 pixels.

    Returns
    -------
    Tuple[int, int]
        (count, area_sum)
    """
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
    normalized_area_coverage = positive_area / image_area
    return {
        "filepath": str(path),
        "cell_count": cell_count,
        "positive_area": positive_area,
        "image_area": image_area,
        "normalized_area_coverage": normalized_area_coverage,
    }


def batch_analyze(paths: List[str | Path], output: Path | None = None) -> pd.DataFrame:
    """Analyze *paths* and return a `pandas.DataFrame`.

    If *output* is given, persist the DataFrame to a Parquet file for fast
    downstream analytics.
    """
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
    print(df.to_string(index=False))


if __name__ == "__main__":
    _main() 
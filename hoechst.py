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
# I/O
# -----------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load image at *path* converting to grayscale float32 in \[0,1]."""
    img = io.imread(str(path))
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return util.img_as_float32(img)


# -----------------------------
# Pre-processing
# -----------------------------

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply 5×5 Gaussian smoothing with σ=2 (as in MATLAB fspecial)."""
    # For kernel size 5 with sigma 2, we need truncate=1 ⇒ radius=2
    return filters.gaussian(gray, sigma=2.0, truncate=1.0)


# -----------------------------
# Thresholding
# -----------------------------

def entropy_threshold(img: np.ndarray) -> float:
    """Global entropy maximisation threshold (MATLAB analogue)."""
    counts, _ = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()

    cumulative = np.cumsum(p)
    cumulative_bg_entropy = np.cumsum(-p * np.log2(p + np.finfo(float).eps))
    # Foreground entropy for each threshold position
    fg_entropy = (
        cumulative_bg_entropy[-1] - cumulative_bg_entropy
    ) / (1 - cumulative + np.finfo(float).eps)
    total_entropy = cumulative_bg_entropy + fg_entropy
    total_entropy = np.nan_to_num(total_entropy, nan=-np.inf)
    t = int(np.argmax(total_entropy[:-1]))
    return t / 255.0


# -----------------------------
# Segmentation & Region Filtering
# -----------------------------

def segment(img: np.ndarray, threshold: float) -> np.ndarray:
    """Convert *img* to binary mask using *threshold*."""
    return img > threshold


VALID_AREA_RANGE: Tuple[int, int] = (5, 20_000)


def filter_regions(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Filter connected components by area and build final mask.

    Returns
    -------
    Tuple[np.ndarray, int]
        final_mask, positive_count
    """
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    final_mask = np.zeros_like(binary, dtype=bool)
    positive_count = 0
    area_min, area_max = VALID_AREA_RANGE

    for r in regions:
        if area_min <= r.area <= area_max:
            positive_count += 1
            final_mask[labeled == r.label] = True
    return final_mask, positive_count


# -----------------------------
# High-level API
# -----------------------------

def analyze_image(path: Path | str) -> dict:
    gray = load_image(path)
    smoothed = preprocess(gray)
    thresh = entropy_threshold(smoothed)
    binary = segment(smoothed, thresh)
    final_mask, positive_count = filter_regions(binary)

    image_area = gray.size
    cell_density = round(positive_count / image_area, 6)
    total_positive_pixels = int(final_mask.sum())

    return {
        "filepath": str(path),
        "cell_count": positive_count,
        "image_area": image_area,
        "density": cell_density,
        "total_positive_pixels": total_positive_pixels,
    }


def batch_analyze(paths: List[str | Path], output: Path | None = None) -> pd.DataFrame:
    """Analyze multiple images, returning a DataFrame and optionally saving to Parquet."""
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
        description="Hoechst analysis — Python port of MATLAB Hoechst.m",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more image paths to process.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Optional Parquet output filepath.",
    )
    args = parser.parse_args()
    df = batch_analyze(args.paths, args.out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    _main() 
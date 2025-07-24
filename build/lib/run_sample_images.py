"""
Quick sanity-check runner for the four analysis pipelines.

Usage
-----
$ python run_sample_images.py
"""
from pathlib import Path

import pandas as pd
from skimage import data, img_as_ubyte, io

import gfap
import hoechst
import neun
import sox9


def _prepare_sample_images(out_dir: Path) -> list[Path]:
    """Save a handful of scikit-image sample pictures and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "camera":      img_as_ubyte(data.camera()),
        "coins":       img_as_ubyte(data.coins()),
        "moon":        img_as_ubyte(data.moon()),
        "text":        img_as_ubyte(data.text()),
        "page":        img_as_ubyte(data.page()),
        "horse":       img_as_ubyte(data.horse()),
        "checker":     img_as_ubyte(data.checkerboard()),
        "rocket":      img_as_ubyte(data.rocket()),
        "clock":       img_as_ubyte(data.clock()),
        # A 2-D slice from the 3-D cells volume
        "cells":       img_as_ubyte(data.cells3d()[30, 1]),
    }

    paths: list[Path] = []
    for name, img in samples.items():
        path = out_dir / f"{name}.png"
        io.imsave(path, img, check_contrast=False)
        paths.append(path)

    print(f"Saved {len(paths)} test images to {out_dir.resolve()}")
    return paths


def _run_pipeline(name: str, batch_fn, img_paths: list[Path]) -> pd.DataFrame:
    """Execute one analysis batch function and pretty-print the result."""
    print(f"\n=== {name} ===")
    df = batch_fn(img_paths)
    print(df.to_string(index=False))
    return df


def main() -> None:
    img_dir = Path("sample_images")
    images = _prepare_sample_images(img_dir)

    _run_pipeline("GFAP",    gfap.batch_analyze,    images)
    _run_pipeline("Hoechst", hoechst.batch_analyze, images)
    _run_pipeline("NeuN",    neun.batch_analyze,    images)
    _run_pipeline("SOX9",    sox9.batch_analyze,    images)


if __name__ == "__main__":
    main() 
import pytest
from pathlib import Path
from pytest import approx

import gfap
import hoechst
import neun
import sox9

# -----------------------------------------------------------------------------
# Expected results from `python run_sample_images.py`
# -----------------------------------------------------------------------------
EXPECTED = {
    "gfap": {
        "camera.png": {"cell_count": 1,  "normalized_area_coverage": 0.000004},
        "coins.png":  {"cell_count": 1,  "normalized_area_coverage": 0.000009},
        "moon.png":   {"cell_count": 1,  "normalized_area_coverage": 0.000004},
        "text.png":   {"cell_count": 3,  "normalized_area_coverage": 0.000130},
        "page.png":   {"cell_count": 1,  "normalized_area_coverage": 0.000014},
        "horse.png":  {"cell_count": 18, "normalized_area_coverage": 0.014924},
        "checker.png":{"cell_count": 49, "normalized_area_coverage": 0.004900},
        "rocket.png": {"cell_count": 1,  "normalized_area_coverage": 0.000004},
        "clock.png":  {"cell_count": 1,  "normalized_area_coverage": 0.000075},
        "cells.png":  {"cell_count": 4,  "normalized_area_coverage": 0.000122},
    },
    "hoechst": {
        "camera.png": {"cell_count": 0, "density": 0.0},
        "coins.png":  {"cell_count": 0, "density": 0.0},
        "moon.png":   {"cell_count": 0, "density": 0.0},
        "text.png":   {"cell_count": 0, "density": 0.0},
        "page.png":   {"cell_count": 1, "density": 0.000014},
        "horse.png":  {"cell_count": 0, "density": 0.0},
        "checker.png":{"cell_count": 0, "density": 0.0},
        "rocket.png": {"cell_count": 0, "density": 0.0},
        "clock.png":  {"cell_count": 1, "density": 0.000008},
        "cells.png":  {"cell_count": 0, "density": 0.0},
    },
    "neun": {
        "camera.png": {"total_valid_area": 38747.0, "coverage_percent": 14.780807},
        "coins.png":  {"total_valid_area": 5176.0,  "coverage_percent": 4.448570},
        "moon.png":   {"total_valid_area": 379.0,   "coverage_percent": 0.144577},
        "text.png":   {"total_valid_area": 40558.0, "coverage_percent": 52.634448},
        "page.png":   {"total_valid_area": 29425.0, "coverage_percent": 40.119164},
        "horse.png":  {"total_valid_area":    0.0,  "coverage_percent": 0.0},
        "checker.png":{"total_valid_area": 22702.0, "coverage_percent": 56.755000},
        "rocket.png": {"total_valid_area":  1208.0, "coverage_percent": 0.442037},
        "clock.png":  {"total_valid_area":  5829.0, "coverage_percent": 4.857500},
        "cells.png":  {"total_valid_area":    65.0, "coverage_percent": 0.099182},
    },
    "sox9": {
        "camera.png": {"cell_count": 0,  "density": 0.0},
        "coins.png":  {"cell_count": 0,  "density": 0.0},
        "moon.png":   {"cell_count": 0,  "density": 0.0},
        "text.png":   {"cell_count": 1,  "density": 0.000013},
        "page.png":   {"cell_count": 0,  "density": 0.0},
        "horse.png":  {"cell_count": 18, "density": 0.000137},
        "checker.png":{"cell_count": 0,  "density": 0.0},
        "rocket.png": {"cell_count": 0,  "density": 0.0},
        "clock.png":  {"cell_count": 1,  "density": 0.000008},
        "cells.png":  {"cell_count": 0,  "density": 0.0},
    },
}

MODULES = [
    (gfap,    "gfap",    ["cell_count", "normalized_area_coverage"]),
    (hoechst, "hoechst", ["cell_count", "density"]),
    (neun,    "neun",    ["total_valid_area", "coverage_percent"]),
    (sox9,    "sox9",    ["cell_count", "density"]),
]

@pytest.mark.parametrize("module,name,fields", MODULES)
def test_batch_analyze(sample_image_paths, module, name, fields):
    df = module.batch_analyze(sample_image_paths)
    df = df.set_index(df["filepath"].map(lambda p: Path(p).name))
    for img_path in sample_image_paths:
        fname = Path(img_path).name
        for field in fields:
            actual = df.loc[fname][field]
            expected = EXPECTED[name][fname][field]
            assert actual == approx(expected, rel=1e-6), (
                f"{name}:{fname}:{field} expected {expected}, got {actual}"
            ) 
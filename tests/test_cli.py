#!/usr/bin/env python3
import sys
import importlib
from pathlib import Path
import pytest

# Fixture to supply test image paths
@pytest.fixture
def sample_image_paths():
    return [
        "sample_images/checker.png"
    ]

# Parametrize the test for each CLI module and its expected output columns
@pytest.mark.parametrize("module_name, expected_cols", [
    ("gfap", ["filepath", "cell_count", "normalized_area_coverage"]),
    ("hoechst", ["filepath", "cell_count", "density"]),
    ("neun", ["filepath", "total_valid_area", "coverage_percent"]),
    ("sox9", ["filepath", "cell_count", "density"]),
])
def test_cli_module(sample_image_paths, capsys, monkeypatch, module_name, expected_cols):
    # Set project root as the working directory
    project_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(project_root)

    # Mock command-line arguments
    monkeypatch.setattr(sys, "argv", [module_name] + sample_image_paths)

    # Dynamically import module and run its CLI entrypoint
    module = importlib.import_module(module_name)
    module._main()

    # Capture output and validate
    out = capsys.readouterr().out
    lines = out.strip().splitlines()

    # Check header matches expected columns
    header = lines[0].split()
    assert header == expected_cols

    # Confirm output row count matches number of images + header
    assert len(lines) == len(sample_image_paths) + 1

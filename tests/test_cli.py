#!/usr/bin/env python3

import sys
import importlib
from pathlib import Path
import pytest

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_image_paths():
    return [
        "sample_images/checker.png"
    ]


# -----------------------------
# Parametrized CLI Test
# -----------------------------

@pytest.mark.parametrize("module_name, expected_cols", [
    ("gfap", ["filepath", "cell_count", "normalized_area_coverage"]),
    ("hoechst", ["filepath", "cell_count", "density"]),
    ("neun", ["filepath", "total_valid_area", "coverage_percent"]),
    ("sox9", ["filepath", "cell_count", "density"]),
])
def test_cli_module(sample_image_paths, capsys, monkeypatch, module_name, expected_cols):
    # Ensure project root is the working directory
    project_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(project_root)

    # Mock command-line arguments for CLI invocation
    monkeypatch.setattr(sys, "argv", [module_name] + sample_image_paths)

    # Dynamically import the module and invoke its CLI entrypoint
    module = importlib.import_module(module_name)
    module._main()

    # Capture output and verify
    out = capsys.readouterr().out
    lines = out.strip().splitlines()

    # First line should match expected header columns
    header = lines[0].split()
    assert header == expected_cols

    # Total output lines should match number of images + header
    assert len(lines) == len(sample_image_paths) + 1

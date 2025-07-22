#!/usr/bin/env python3
import sys
import importlib
from pathlib import Path
import pytest

@pytest.mark.parametrize("module_name, expected_cols", [
    ("gfap", ["filepath", "cell_count", "normalized_area_coverage"]),
    ("hoechst", ["filepath", "cell_count", "density"]),
    ("neun", ["filepath", "total_valid_area", "coverage_percent"]),
    ("sox9", ["filepath", "cell_count", "density"]),
])
def test_cli_module(sample_image_paths, capsys, monkeypatch, module_name, expected_cols):
    # Ensure project root is cwd
    project_root = Path(__file__).parent.parent
    monkeypatch.chdir(project_root)
    # Monkeypatch sys.argv for CLI invocation
    monkeypatch.setattr(sys, "argv", [module_name] + sample_image_paths)
    # Import module and run its CLI main function
    module = importlib.import_module(module_name)
    module._main()
    # Capture CLI output
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    # First line is header with expected columns
    header = lines[0].split()
    assert header == expected_cols
    # Number of output rows equals number of images + header
    assert len(lines) == len(sample_image_paths) + 1 
1  Project Goal (why we’re doing this)
 • Replace the legacy MATLAB loop that loads one TIFF at a time, thresholds, labels cells, and writes a .mat file.
 • Deliver a modern, production-ready Python package that runs 3× faster, is easier to debug, and outputs an analytics-friendly Parquet file so DarkVision can drop it straight into Pandas, Power BI, or Dask.
 • Keep behaviour identical to the MATLAB reference run on 10 verification images (same cell counts, same normalized coverage).


2  Quality bar we need to hit


Speed Pure NumPy (vectorised) or Cython on any tight loops; end-to-end ≤ ⅓ MATLAB time.
 Separate I/O, preprocessing, thresholding, measurement, and CLI.
Reproducibility pytest unit tests; deterministic threshold results; fixed random seeds if used.
 Docstrings, typed function signatures, PEP 8 / flake8 clean.
 Able to batch thousands of TIFFs without blowing RAM; Parquet output for downstream tools.
CI/CD ready Minimal GitLab CI job: install deps → run tests → lint.

3. Convert Matlab Files To Python That Fulfills Above Requirements

3.1. Convert GFAP.m to Python

3.2. Convert Hoechst.m to Python

3.3. Convert NeuN.m to Python

3.4. Convert SOX9.m to Python
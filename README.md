Image Processing Pipelines (GFAP, Hoechst, NeuN, SOX9)
=====================================================

This repository contains four production-grade Python pipelines that replicate and enhance MATLAB-based image analysis workflows for immunostaining quantification. Each pipeline supports batch image processing, robust thresholding, region filtering, and CLI execution.

Originally developed for neuroscience research, these pipelines are adaptable to many grayscale image segmentation tasks.

---------------------------------------------------------------------
1. Setup and Usage
---------------------------------------------------------------------
---

## 1. Installation

### 1.1 Clone the repository

```bash
git clone https://github.com/MohammadrezaRH/neuroimage-quantification.git
cd neuroimage-quantification
```

### 1.2 (Optional but recommended) Create and activate a virtual environment

```bash
# Create the environment
py -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1
```

If you see a script execution error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 2. Running the Pipelines

Each stain has its own CLI command:

```bash
# GFAP example
gfap sample_images/cells.png

# Hoechst example
hoechst sample_images/coins.png

# NeuN example
neun sample_images/page.png

# SOX9 example
sox9 sample_images/rocket.png
```

You can provide multiple images or use wildcards (`*.png`). Output is printed to the console or can be saved using `--out`.

```bash
sox9 --out sox9_results.parquet sample_images/*.png
```

---

## 3. Running All Pipelines on Sample Images

To run all pipelines on default sample images (using `skimage.data`), run:

```bash
py run_sample_images.py
```

This executes all four analysis pipelines and prints their output tables.

---

## 4. Output Format

Each pipeline returns a table with standardized columns:

- **GFAP**: filepath, cell_count, normalized_area_coverage  
- **Hoechst**: filepath, cell_count, density  
- **NeuN**: filepath, total_valid_area, coverage_percent  
- **SOX9**: filepath, cell_count, density

---

## 5. Running Tests

```bash
# Run all tests
pytest

# Verbose mode
pytest -v
```

---------------------------------------------------------------------
5. Tech Stack
---------------------------------------------------------------------

- Python ≥ 3.8
- numpy, pandas, scikit-image
- pytest, mypy, ruff, bandit
- CLI packaging with setuptools

---------------------------------------------------------------------
6. What These Pipelines Are For
---------------------------------------------------------------------

These tools were built to analyze microscope images of brain tissue from animals infected with mild COVID-19. The goal was to measure how specific brain cells respond to infection.

We used four fluorescent markers:

    GFAP    - Active support cells (astrocytes)
    SOX9    - Total number of astrocytes
    NeuN    - Neurons
    Hoechst - All cell nuclei

These tools:
- Automatically detect and count cells in large images
- Replace manual work with reliable, fast analysis
- Work from the command line and output clean tabular data

They were used in a neuroscience study to show that female animals had a temporary increase in astrocyte activity during infection — possibly helping them recover faster.

Although developed for brain images, these pipelines can be adapted to many other scientific or technical image processing tasks.

---------------------------------------------------------------------
7. Publication
---------------------------------------------------------------------

This pipeline was used in a published preprint:

    A sex-specific and transient astrocyte response to mild respiratory COVID-19 in the hamster brain  
    DOI: https://doi.org/10.1101/2025.04.08.647811

---------------------------------------------------------------------
8. License
---------------------------------------------------------------------

MIT License. Free to use, adapt, and distribute.

---------------------------------------------------------------------
9. Author
---------------------------------------------------------------------

Developed by MohammadrezaRH  
GitHub: https://github.com/MohammadrezaRH

Image Processing Pipelines (GFAP, Hoechst, NeuN, SOX9)
=====================================================

This repository contains four production-grade Python pipelines that replicate and enhance MATLAB-based image analysis workflows for immunostaining quantification. Each pipeline supports batch image processing, robust thresholding, region filtering, and CLI execution.

Originally developed for neuroscience research, these pipelines are adaptable to many grayscale image segmentation tasks.

---------------------------------------------------------------------
1. Setup and Usage
---------------------------------------------------------------------

1.1 Installation

Clone the repository:
   git clone https://github.com/MohammadrezaRH/neuroimage-quantification.git
   cd neuroimage-quantification

Create and activate a virtual environment (recommended):
    python -m venv venv

    # On Windows:
    .\venv\Scripts\activate

    # On Unix/macOS:
    source venv/bin/activate

Install the package and its dependencies:
    pip install -e .

---------------------------------------------------------------------
2. Running the Analysis
---------------------------------------------------------------------

This package provides four command-line tools:

    gfap       - Quantify GFAP signal (astrocyte reactivity)
    hoechst    - Count total cells using Hoechst stain
    neun       - Count neurons using NeuN stain
    sox9       - Count astrocyte nuclei using SOX9

Example usage:

    gfap path/to/image.tiff
    hoechst image1.tiff image2.tiff image3.tiff
    neun --out results.parquet image*.tiff
    run-sample-images

---------------------------------------------------------------------
3. Output Format
---------------------------------------------------------------------

Each pipeline returns a structured table. Columns vary by stain:

GFAP:
    filepath, cell_count, positive_area, image_area, normalized_area_coverage

Hoechst:
    filepath, cell_count, image_area, density

NeuN:
    filepath, total_valid_area, image_area, coverage_percent

SOX9:
    filepath, cell_count, image_area, density

Output can be:
    - Printed to the console
    - Saved to Parquet (--out)
    - Loaded into pandas, Power BI, Dask, etc.

---------------------------------------------------------------------
4. Running Tests
---------------------------------------------------------------------

Run the full test suite:
    pytest

Run a specific test:
    pytest tests/test_pipelines.py

Run with verbose output:
    pytest -v

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

"""
Setup script for pip installation of the cell analysis pipelines.
"""
<<<<<<< HEAD

from pathlib import Path
from setuptools import setup

this_directory = Path(__file__).parent
=======
from pathlib import Path
from setuptools import setup

# read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
>>>>>>> 7055819 (Module Built)

setup(
    name="cell-analysis-pipelines",
    version="0.1.0",
    description="Python port of MATLAB cell analysis pipelines (GFAP, Hoechst, NeuN, SOX9)",
<<<<<<< HEAD
    author="Reza Rahmani",
    author_email="mrahmanimanesh@uvic.ca",
    license="MIT",
=======
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
>>>>>>> 7055819 (Module Built)
    py_modules=["gfap", "hoechst", "neun", "sox9", "run_sample_images"],
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scikit-image>=0.17.0",
    ],
    entry_points={
        "console_scripts": [
            "gfap=gfap:_main",
            "hoechst=hoechst:_main",
            "neun=neun:_main",
            "sox9=sox9:_main",
            "run-sample-images=run_sample_images:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
<<<<<<< HEAD
        "Operating System :: OS Independent",
    ],
)
=======
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
>>>>>>> 7055819 (Module Built)

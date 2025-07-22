import sys, os
from pathlib import Path
import pytest
from pytest import approx

# Ensure project root is on PYTHONPATH so we can import gfap, hoechst, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

@pytest.fixture(scope="session")
def sample_image_paths():
    """Return sorted list of all 10 PNGs in sample_images/"""
    data_dir = Path(__file__).parent.parent / "sample_images"
    return sorted(str(p) for p in data_dir.glob("*.png")) 
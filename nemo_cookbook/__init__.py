"""
NEMO Cookbook

Reproducible analysis of NEMO ocean general circulation model outputs using xarray.
"""
__author__ = "Ollie Tooth (oliver.tooth@noc.ac.uk)"
__credits__ = "National Oceanography Centre (NOC), Southampton, UK"

from importlib.metadata import version as _version

from nemo_cookbook import (
    examples,
    extract,
    integrate,
    interpolate,
    masks,
    stats,
    transform,
)
from nemo_cookbook.nemodataarray import NEMODataArray
from nemo_cookbook.nemodatatree import NEMODataTree

try:
    __version__ = _version("nemo_cookbook")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999.0.0"

__all__ = ("NEMODataArray", "NEMODataTree", "examples", "extract", "masks", "transform", "stats", "integrate", "interpolate")
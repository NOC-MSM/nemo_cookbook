"""
utils.py

Description:
This module includes utility functions for the core data structures
used in the NEMO Cookbook library.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import functools
import warnings
from typing import Callable, TypedDict

import xarray as xr

# -- Type Aliases and TypedDict Defintions -- #
GridPaths = dict[str, str]
NestedGridPaths = dict[str, GridPaths]

class PathsDict(TypedDict, total=False):
    parent: GridPaths
    child: NestedGridPaths
    grandchild: NestedGridPaths

GridDatasets = dict[str, xr.Dataset]
NestedGridDatasets = dict[str, GridDatasets]

class DatasetsDict(TypedDict, total=False):
    parent: GridDatasets
    child: NestedGridDatasets
    grandchild: NestedGridDatasets

NestConfig = dict[str, str | int | bool]
NestsDict = dict[str, NestConfig]


# -- Decorators -- #
def deprecated(
    version_since: str,
    version_removed : str,
    alternative : str | None = None
    ) -> Callable:
    """
    Utility function to issue a deprecation warning.

    Parameters:
    version_since : str
        Version since which the function or class has been deprecated.
    version_removed : str
        Version in which the deprecated function or class will be removed.
    alternative : str, optional
        Name of the alternative function or class that should be used instead.
     """
    def decorator(func):
        message = (
            f"{func.__qualname__} is deprecated since v{version_since} "
            f"and will be removed in v{version_removed}."
        )

        if alternative:
            message += f"\n Use {alternative} instead."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                FutureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator

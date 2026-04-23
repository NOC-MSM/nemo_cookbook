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
from typing import Callable


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

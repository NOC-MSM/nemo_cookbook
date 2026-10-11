"""
integrate.py

Description:
This module includes functions to perform integration operations
on NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import numpy.typing as npt
from numba import guvectorize


@guvectorize(
    "(float64[:], float64[:], float64, float64, float64[:])",
    "(n), (n), (), () -> ()",
    nopython=True,
)
def compute_depth_integral(
    e3_in: npt.NDArray[np.float64],
    var_in: npt.NDArray[np.float64],
    z_max: float,
    z_min: float,
    result: npt.NDArray[np.float64],
):
    """
    Compute vertical integral of variable in depth coordinates.

    Parameters
    ----------
    e3_in : ndarray[np.float64]
        Vertical grid cell thicknesses of the input grid.
    var_in : ndarray[np.float64]
        Values of variable defined at the centre of each vertical
        grid cell on the input grid.
    z_max : float
        Upper limit of depth integral (m).
    z_min : float
        Lower limit of depth integral (m).

    Returns
    -------
    result : float
        Integral of chosen variable between depth surfaces (z_min, z_max).
    """
    # Mask NaNs from input:
    mask = ~np.isnan(e3_in)
    e3_masked = e3_in[mask]
    var_masked = var_in[mask]

    if (e3_masked.size == 0) or np.isnan(z_min) or np.isnan(z_max):
        result[0] = np.nan
        return

    # Accumulate values contained within (z_min <= z <= z_max):
    z_top = 0.0
    total = 0.0
    for k in range(e3_masked.size):
        if z_top >= z_max:
            break
        z_bottom = z_top + e3_masked[k]
        dz = min(z_bottom, z_max) - max(z_top, z_min)
        if dz > 0.0:
            total += dz * var_masked[k]
        z_top = z_bottom

    result[0] = total

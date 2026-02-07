"""
core.py

Description:
This module includes functions to perform core scalar and vector
operations on NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

from numba import guvectorize
import numpy.typing as npt
import numpy as np


@guvectorize(
    "(float64[:], float64[:], float64[:], float64[:], float64[:])",
    "(n), (n), (m), (m) -> (m)",
    nopython=True,
)
def compute_depth_integral(
    e3_in: npt.NDArray[np.float64],
    var_in: npt.NDArray[np.float64],
    z_upper: npt.NDArray[np.float64],
    z_lower: npt.NDArray[np.float64],
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
    z_upper : ndarray[np.float64]
        Upper limit of depth integral (m).
    z_lower : ndarray[np.float64]
        Lower limit of depth integral (m).

    Returns
    -------
    result : float
        Integral of chosen variable between depth surfaces (z_lower, z_upper).
    """

    # Mask NaNs from input:
    mask = ~np.isnan(e3_in)
    e3t_masked = e3_in[mask]
    var_masked = var_in[mask]

    n_ini = e3t_masked.size
    if n_ini == 0:
        result[:] = np.nan
        return

    else:
        # Compute depth boundaries:
        z_ini = np.zeros(n_ini + 1)
        z_ini[1:] = np.cumsum(e3t_masked)

        # Vertically integrate variable:
        dz = np.maximum(
            0.0,
            np.minimum(z_ini[1:], z_upper) - np.maximum(z_ini[:-1], z_lower),
        )

        result[:] = np.sum(dz * var_masked)


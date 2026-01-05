"""
transform.py

Description:
This module includes functions to perform vertical grid
transformations for NEMO ocean general circulation model
domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import numpy.typing as npt
from numba import guvectorize, prange


# -- Internal Functions -- #
@guvectorize(
    "(float64[:], float64[:], float64[:], float64[:], float64[:])",
    "(n), (n), (m) -> (m), (m)",
    nopython=True,
)
def transform_vertical_coords(
    e3_in: npt.NDArray[np.float64],
    var_in: npt.NDArray[np.float64],
    e3_target: npt.NDArray[np.float64],
    var_tout: npt.NDArray[np.float64],
    e3_tout: npt.NDArray[np.float64],
):
    """
    Conservatively remaps a vertical profile of a variable to a new vertical grid.

    Parameters
    ----------
    e3_in : ndarray
        Vertical grid cell thicknesses of the input grid.
    var_in : ndarray
        Values of variable defined at the centre of each vertical
        grid cell on the input grid.
    e3_target : ndarray
        Vertical grid cell thicknesses of the target grid.

    Outputs
    -------
    var_tout : ndarray
        Variable values remapped to target vertical grid.
    e3_tout : ndarray
        Adjusted target grid accounting for partial cells.
    """

    # Mask NaNs from input:
    mask = ~np.isnan(e3_in)
    e3t_masked = e3_in[mask]
    var_masked = var_in[mask]

    n_ini = e3t_masked.size
    if n_ini == 0:
        var_tout[:] = np.nan
        e3_tout[:] = np.nan
        return

    # Compute cumulative thickness for matching target grid:
    total_thickness = e3t_masked.sum()
    cumulative_target = e3_target.cumsum()
    valid_indices = np.argwhere(cumulative_target < total_thickness)

    # Adjust target grid to match the total thickness:
    if valid_indices.size == 0:
        e3t_adjusted = np.zeros(1)
        e3t_adjusted[0] = total_thickness
    else:
        last_index = valid_indices[-1].item()
        e3t_adjusted = e3_target[: last_index + 2].copy()
        e3t_adjusted[-1] = total_thickness - e3t_adjusted[:-1].sum()

    n_target = e3t_adjusted.size

    # Compute depth boundaries:
    z_ini = np.zeros(n_ini + 1)
    z_target = np.zeros(n_target + 1)
    z_ini[1:] = np.cumsum(e3t_masked)
    z_target[1:] = np.cumsum(e3t_adjusted)

    var_out = np.zeros(n_target)

    # Remap variable to target vertical grid:
    for p in prange(n_target):
        z_top = z_target[p]
        z_bot = z_target[p + 1]

        dz = np.maximum(
            0.0,
            np.minimum(z_ini[1:], z_bot) - np.maximum(z_ini[:-1], z_top),
        )

        dz_sum = np.sum(dz)
        if dz_sum == 0:
            var_out[p] = np.nan
        else:
            var_out[p] = np.sum(dz * var_masked) / dz_sum

    # Fill outputs with results or pad with NaNs:
    var_tout[:] = np.nan
    e3_tout[:] = np.nan
    var_tout[:n_target] = var_out
    e3_tout[:n_target] = e3t_adjusted

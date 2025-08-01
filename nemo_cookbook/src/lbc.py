"""
lbc.py

Description:
This module includes functions to handle Lateral Boundary Conditions (LBCs),
including the North Fold (NFD) in NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import numpy as np
import xarray as xr

def _apply_lbc_nfd(
    array_in: xr.DataArray,
    sgn_in: int,
    ipj: int,
    jpiglo: int,
    nn_hls: int,
    grid_type: str = "V",
) -> xr.DataArray:
    """
    Apply North Fold (NFD) to a DataArray following the NEMO
    DOMAINcfg subroutine src/lbc_nfd_generic.h90.

    Parameters
    ----------
    array_in : xr.DataArray
        DataArray to which NFD is applied.
    sgn_in : int
        Sign indicating the direction of the NFD. Options are 1 or -1.
    ipj : int
        Size of domain (j-points), including halos.
    jpiglo : int
        Size of domain (i-points), including halos.
    nn_hls : int
        Number of halo grid points.
    grid_type: str
        Type of grid point to apply NFD. Options are only "V" for meridional
        velocity points currently.

    Returns
    -------
    xr.DataArray
        DataArray with NFD applied.
    """
    if not isinstance(array_in, xr.DataArray):
        raise TypeError("array_in must be an xarray DataArray")
    if not isinstance(sgn_in, int):
        raise TypeError("sgn_in must be an integer")
    if sgn_in not in [1, -1]:
        raise ValueError("sgn_in must be either 1 or -1")
    if not isinstance(ipj, int) or not isinstance(jpiglo, int) or not isinstance(nn_hls, int):
        raise TypeError("ipj, jpiglo, and nn_hls must be integers")

    # Define domain size (i-points) excluding halos:
    ni0glo = jpiglo - 2 * nn_hls

    # TODO: Handle F-folds as well as T-folds for eORCA1.
    match grid_type:
        case "V":
            # Handle typical output with no halos:
            if nn_hls == 0:
                nn_hls_max = nn_hls + 2
            else:
                nn_hls_max = nn_hls + 1

            # Iterate over northernmost j-points & apply NFD:
            for jj in range(1, nn_hls_max):
                ij1 = ipj - jj + 1
                ij2 = ipj - 2 * nn_hls + jj - 2

                # First nn_hls points
                for ji in range(1, nn_hls_max):
                    ii1 = ji
                    ii2 = 2 * nn_hls + 2 - ji
                    array_in[:, ij1-1, ii1-1] = sgn_in * array_in[:, ij2-1, ii2-1]

                # Point nn_hls+1
                for ji in range(1, 2):
                    ii1 = nn_hls + ji
                    ii2 = ii1
                    array_in[:, ij1-1, ii1-1] = sgn_in * array_in[:, ij2-1, ii2-1]

                # Points from nn_hls+2 to jpiglo - nn_hls
                for ji in range(1, ni0glo):
                    ii1 = 2 + nn_hls + ji - 1
                    ii2 = jpiglo - nn_hls - ji + 1
                    array_in[:, ij1-1, ii1-1] = sgn_in * array_in[:, ij2-1, ii2-1]

                # Last nn_hls - 1 points
                for ji in range(1, nn_hls):
                    ii1 = jpiglo - nn_hls + ji + 1
                    ii2 = jpiglo - nn_hls - ji + 1
                    array_in[:, ij1-1, ii1-1] = sgn_in * array_in[:, ij2-1, ii2-1]

            # Assign coordinates to the output DataArray:
            array_out = array_in.assign_coords({'y': np.arange(1, ipj + 1)})

    return array_out
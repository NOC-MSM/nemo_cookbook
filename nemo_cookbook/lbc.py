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


def _lbc_nfd(
    ptab:np.ndarray,
    c_NFtype:str,
    cd_nat:str,
    ihls:int,
    psgn:int,
    ipi:int,
    ipj:int,
    Ni0glo:int
    ) -> np.ndarray:
    """
    Apply NEMO North Fold (NFD) Lateral Boundary Condition
    to a input DataArray.

    This is a Python vectorised implementation of the original NEMO subroutine:
    /nemo/src/OCE/LBC/lbc_nfd_generic.h90

    Parameters
    ----------
    ptab : np.ndarray
        Array on which apply the boundary condition.
    c_NFtype : str
        Type of North Fold condition to apply.
        Options are "T" for T-point pivot or "F" for F-point pivot.
    cd_nat : str
        Nature of array grid-points.
        Options are "T", "W", "U", "V" or "F".
    ihls : int
        Number of halo points.
    psgn : int
        Sign used across the north fold boundary.
        Options are 1 for positive and -1 for negative.
    ipi : int
        Size of the i-dimension of the array.
    ipj : int
        Size of the j-dimension of the array.
    Ni0glo : int
        Number of interior points in the i-dimension (excluding halos).

    Returns
    -------
    np.ndarray
        Array with north fold boundary condition applied.
    """
    # === Apply north fold boundary condition === #
    # North fold  T-point pivot:
    if c_NFtype == "T":
        match cd_nat:
            case "T" | "W":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 2 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Point ihls+1:
                    ii1 = ihls + 1
                    ii2 = ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Points from ihls+2 to ipi - ihls:
                    ii1 = 2 + ihls + np.arange(1, Ni0glo) - 1
                    ii2 = ipi - ihls - np.arange(1, Ni0glo) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Point ipi - ihls + 1:
                    ii1 = ipi - ihls + 1
                    ii2 = ihls + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls-1 points:
                    ii1 = ipi - ihls + 1 + np.arange(1, ihls)
                    ii2 = ipi - ihls + 1 - np.arange(1, ihls)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                # -- Line number ipj - ihls : right half --- #
                for jj in range(1, 2):  # only once
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+2 to ipi - ihls:
                    ii1 = ipi // 2 + np.arange(1, Ni0glo // 2) + 1
                    ii2 = ipi // 2 - np.arange(1, Ni0glo // 2) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # First ihls points: redo just in case:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 2 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls points (DO ji = 1, ihls):
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                # --- Line number ipj - ihls : right half --- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls  (DO ji = 1, Ni0glo/2):
                    ii1 = ipi // 2 + np.arange(1, Ni0glo // 2 + 1)
                    ii2 = ipi // 2 - np.arange(1, Ni0glo // 2 + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls points: redo them just in case:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "V":
                # -- Last ihls+1 lines (from ipj to ipj-ihls) : full -- #
                for jj in range(1, ihls + 2):   # Fortran: DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 2 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls+1:
                    ii1 = ihls + 1
                    ii2 = ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls+2 to ipi - ihls  (Fortran: DO ji = 1, Ni0glo - 1):
                    ii1 = 2 + ihls + np.arange(1, Ni0glo) - 1
                    ii2 = ipi - ihls - np.arange(1, Ni0glo) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # IF( ihls > 0 ) THEN ; DO ji = 1, COUNT( (/ihls > 0/) ) -> one iteration if ihls>0:
                    if ihls > 0:
                        ii1 = ipi - ihls + 1
                        ii2 = ihls + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                            )

                    # Last ihls-1 points:
                    ii1 = ipi - ihls + 1 + np.arange(1, ihls)
                    ii2 = ipi - ihls + 1 - np.arange(1, ihls)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

            case "F":
                # -- Last ihls+1 lines (from ipj to ipj - ihls) : full -- #
                for jj in range(1, ihls + 2):  # DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls points (DO ji = 1, ihls):
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

    # North fold  F-point pivot:
    elif c_NFtype == "F":
        match cd_nat:
            case "T" | "W":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):    # DO jj = 1, ihls
                    ij1 = ipj + 1 - jj
                    ij2 = ipj - 2 * ihls + jj

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls:
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj + 1 - jj
                    ij2 = ipj - 2 * ihls + jj

                    # First ihls-1 points:
                    ii1 = np.arange(1, ihls)
                    ii2 = 2 * ihls - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls (ihls > 0):
                    ii1 = ihls
                    ii2 = ipi - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Points from ihls+1 to ipi - ihls - 1:
                    ii1 = ihls + np.arange(1, Ni0glo)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Point ipi - ihls:
                    ii1 = ipi - ihls
                    ii2 = ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

            case "V":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls:
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                # -- Line number ipj - ihls : right half -- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls:
                    ii1 = ipi // 2 + np.arange(1, Ni0glo // 2 + 1)
                    ii2 = ipi // 2 - np.arange(1, Ni0glo // 2 + 1) + 1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls points: redo them just in case:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "F":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls - 1 points:
                    ii1 = np.arange(1, ihls)
                    ii2 = 2 * ihls - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Point ihls:
                    ii1 = ihls
                    ii2 = ipi - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Points from ihls+1 to ipi - ihls - 1:
                    ii1 = ihls + np.arange(1, Ni0glo)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # Point ipi - ihls:
                    ii1 = ipi - ihls
                    ii2 = ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                # -- Line number ipj - ihls : right half -- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls - 1:
                    ii1 = ipi // 2 + np.arange(1, Ni0glo // 2)
                    ii2 = ipi // 2 - np.arange(1, Ni0glo // 2)
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                    )

                    # First ihls - 1 points: redo them just in case:
                    ii1 = np.arange(1, ihls)
                    ii2 = 2 * ihls - ii1
                    ptab[:, ij1-1, ii1-1] = (
                        psgn * ptab[:, ij2-1, ii2-1]
                        )

    return ptab


def lbc_nfd(
    c_NFtype:str,
    cd_nat:str,
    ihls:int,
    ptab:xr.DataArray,
    psgn:int
    ) -> xr.DataArray:
    """
    Apply NEMO North Fold (NFD) Lateral Boundary Condition
    to a input DataArray.

    This is a Python implementation of the original NEMO subroutine:
    /nemo/src/OCE/LBC/lbc_nfd_generic.h90

    Parameters
    ----------
    c_NFtype : str
        Type of North Fold condition to apply.
        Options are "T" for T-point pivot or "F" for F-point pivot.
    cd_nat : str
        Nature of array grid-points.
        Options are "T", "W", "U", "V" or "F".
    ihls : int
        Number of halo points.
    ptab : xr.DataArray
        DataArray on which apply the boundary condition.
    psgn : int
        Sign used across the north fold boundary.
        Options are 1 for positive and -1 for negative.

    Returns
    -------
    xr.DataArray
        DataArray with north fold boundary condition applied.
    """
    # === Validate Inputs === #
    if c_NFtype not in ["T", "F"]:
        raise ValueError("Invalid c_NFtype. Options are 'T' or 'F'.")
    if cd_nat not in ["T", "W", "U", "V", "F"]:
        raise ValueError("Invalid cd_nat. Options are 'T', 'W', 'U', 'V' or 'F'.")
    if psgn not in [1, -1]:
        raise ValueError("Invalid psgn. Options are 1 for positive and -1 for negative.")

    if "x" not in ptab.coords:
        raise ValueError("DataArray must have 'x' coordinate.")
    if "y" not in ptab.coords:
        raise ValueError("DataArray must have 'y' coordinate.")

    # === Define domain parameters === #
    ipi = ptab.coords["x"].size
    ipj = ptab.coords["y"].size
    Ni0glo = ipi - 2 * ihls

    # === Apply North Fold LBC === #
    # Convert mask to contiguous ndarray & apply LBC:
    array = np.ascontiguousarray(ptab.data)
    array = _lbc_nfd(array, c_NFtype, cd_nat, ihls, psgn, ipi, ipj, Ni0glo)

    # Convert updated mask to DataArray:
    ptab = xr.DataArray(
        data=array,
        dims=ptab.dims,
        coords=ptab.coords,
        attrs=ptab.attrs
    )

    return ptab

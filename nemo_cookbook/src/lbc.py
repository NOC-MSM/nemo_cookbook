"""
lbc.py

Description:
This module includes functions to handle Lateral Boundary Conditions (LBCs),
including the North Fold (NFD) in NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import xarray as xr

def lbc_nfd(
    c_NFtype:str,
    cd_nat:str,
    ihls:int,
    ptab:xr.DataArray,
    psgn:int
    ):
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
                    for ji in range(1, ihls + 1):
                        ii1 = ji
                        ii2 = 2 * ihls + 2 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls+1:
                    for ji in range(1, 2):  # only ji = 1
                        ii1 = ihls + ji
                        ii2 = ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls+2 to ipi - ihls:
                    for ji in range(1, Ni0glo):  # 1..Ni0glo-1
                        ii1 = 2 + ihls + ji - 1
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ipi - ihls + 1:
                    for ji in range(1, 2):  # COUNT((/ihls > 0/)) == 1 if ihls>0
                        ii1 = ipi - ihls + ji
                        ii2 = ihls + ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls-1 points:
                    for ji in range(1, ihls):
                        ii1 = ipi - ihls + 1 + ji
                        ii2 = ipi - ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                # -- Line number ipj - ihls : right half --- #
                for jj in range(1, 2):  # only once
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+2 to ipi - ihls:
                    for ji in range(1, Ni0glo // 2):
                        ii1 = ipi // 2 + ji + 1
                        ii2 = ipi // 2 - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls points: redo just in case:
                    for ji in range(1, ihls + 1):
                        ii1 = ji
                        ii2 = 2 * ihls + 2 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    for ji in range(1, ihls + 1):
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    for ji in range(1, Ni0glo + 1):
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points (DO ji = 1, ihls):
                    for ji in range(1, ihls + 1):
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                # --- Line number ipj - ihls : right half --- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls  (DO ji = 1, Ni0glo/2):
                    for ji in range(1, Ni0glo // 2 + 1):
                        ii1 = ipi // 2 + ji
                        ii2 = ipi // 2 - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls points: redo them just in case:
                    for ji in range(1, ihls + 1):
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "V":
                # -- Last ihls+1 lines (from ipj to ipj-ihls) : full -- #
                for jj in range(1, ihls + 2):   # Fortran: DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    for ji in range(1, ihls + 1):
                        ii1 = ji
                        ii2 = 2 * ihls + 2 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls+1:
                    for ji in range(1, 2):  # only ji = 1
                        ii1 = ihls + ji
                        ii2 = ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls+2 to ipi - ihls  (Fortran: DO ji = 1, Ni0glo - 1):
                    for ji in range(1, Ni0glo):
                        ii1 = 2 + ihls + ji - 1
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # IF( ihls > 0 ) THEN ; DO ji = 1, COUNT( (/ihls > 0/) ) -> one iteration if ihls>0:
                    if ihls > 0:
                        for ji in range(1, 2):  # one iteration: ji = 1
                            ii1 = ipi - ihls + ji
                            ii2 = ihls + ji
                            ptab[:, ij1-1, ii1-1] = (
                                psgn * ptab[:, ij2-1, ii2-1]
                            )

                    # Last ihls-1 points:
                    for ji in range(1, ihls):
                        ii1 = ipi - ihls + 1 + ji
                        ii2 = ipi - ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "F":
                # -- Last ihls+1 lines (from ipj to ipj - ihls) : full -- #
                for jj in range(1, ihls + 2):  # DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    for ji in range(1, Ni0glo + 1):
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points (DO ji = 1, ihls):
                    for ji in range(1, ihls + 1):
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji + 1
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
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls:
                    for ji in range(1, Ni0glo + 1):  # DO ji = 1, Ni0glo
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj + 1 - jj
                    ij2 = ipj - 2 * ihls + jj

                    # First ihls-1 points:
                    for ji in range(1, ihls):  # DO ji = 1, ihls-1
                        ii1 = ji
                        ii2 = 2 * ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls (ihls > 0):
                    for ji in range(1, 2):  # DO ji = 1, 1
                        ii1 = ihls + ji - 1
                        ii2 = ipi - ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls+1 to ipi - ihls - 1:
                    for ji in range(1, Ni0glo):  # DO ji = 1, Ni0glo - 1
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ipi - ihls:
                    for ji in range(1, 2):  # DO ji = 1, 1
                        ii1 = ipi - ihls + ji - 1
                        ii2 = ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

            case "V":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls to ipi - ihls:
                    for ji in range(1, Ni0glo + 1):  # DO ji = 1, Ni0glo
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                # -- Line number ipj - ihls : right half -- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls:
                    for ji in range(1, Ni0glo // 2 + 1):  # DO ji = 1, Ni0glo/2
                        ii1 = ipi // 2 + ji
                        ii2 = ipi // 2 - ji + 1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls points: redo them just in case:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ji
                        ii2 = 2 * ihls + 1 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                    )

            case "F":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls - 1 points:
                    for ji in range(1, ihls):  # DO ji = 1, ihls-1
                        ii1 = ji
                        ii2 = 2 * ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ihls:
                    for ji in range(1, 2):  # DO ji = 1, 1
                        ii1 = ihls + ji - 1
                        ii2 = ipi - ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Points from ihls+1 to ipi - ihls - 1:
                    for ji in range(1, Ni0glo):  # DO ji = 1, Ni0glo - 1
                        ii1 = ihls + ji
                        ii2 = ipi - ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Point ipi - ihls:
                    for ji in range(1, 2):  # DO ji = 1, 1
                        ii1 = ipi - ihls + ji - 1
                        ii2 = ii1
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # Last ihls points:
                    for ji in range(1, ihls + 1):  # DO ji = 1, ihls
                        ii1 = ipi - ihls + ji
                        ii2 = ipi - ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                # -- Line number ipj - ihls : right half -- #
                for jj in range(1, 2):  # DO jj = 1, 1
                    ij1 = ipj - ihls
                    ij2 = ij1  # same line

                    # Points from ipi/2+1 to ipi - ihls - 1:
                    for ji in range(1, Ni0glo // 2):  # DO ji = 1, Ni0glo/2 - 1
                        ii1 = ipi // 2 + ji
                        ii2 = ipi // 2 - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

                    # First ihls - 1 points: redo them just in case:
                    for ji in range(1, ihls):  # DO ji = 1, ihls-1
                        ii1 = ji
                        ii2 = 2 * ihls - ji
                        ptab[:, ij1-1, ii1-1] = (
                            psgn * ptab[:, ij2-1, ii2-1]
                        )

    return ptab

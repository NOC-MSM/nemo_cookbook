"""
lbc.py.

Description:
This module includes functions to handle Lateral Boundary Conditions (LBCs),
including the North Fold (NFD) in NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import xarray as xr


def _apply_lbc_nfd(
    da: xr.DataArray,
    c_NFtype: str,
    cd_nat: str,
    psgn: int,
    ihls: int = 1,
) -> xr.DataArray:
    """
    Apply NEMO North Fold (NFD) lateral boundary condition to a DataArray.

    This is a lazy xarray implementation of the NEMO subroutine lbc_nfd
    (src/OCE/LBC/lbc_nfd_generic.h90).

    Note: NEMO lbc_nfd also overwrites the last interior row (Nj) for T-pivot
    T/W/U-points and F-pivot V/F-points. This is not reproduced here, since
    NEMO model outputs already satisfy the north fold condition along row Nj.

    Parameters
    ----------
    da : xr.DataArray
        DataArray defined on the interior domain on which to apply
        the north fold lateral boundary condition with dimensions
        (j, i) spanning the full global domain.
    c_NFtype : str
        Type of North Fold condition to apply.
        Options are "T" for T-point pivot or "F" for F-point pivot.
    cd_nat : str
        Nature of array grid-points.
        Options are "T", "W", "U", "V" or "F".
    psgn : int
        Sign used across the north fold. Use 1 for scalars (e.g., land-sea masks,
        tracers, scale factors) and -1 for vector components (e.g., velocities).
    ihls : int, optional
        Number of halo points to add along the lateral boundaries.
        Default is 1.

    Returns
    -------
    xr.DataArray
        DataArray with north fold lateral boundary condition applied, of size
        (..., Nj + ihls, Ni + 2 * ihls) in the (j, i) dimensions.
    """
    # === Validate Inputs === #
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray.")
    if c_NFtype not in ["T", "F"]:
        raise ValueError("Invalid c_NFtype. Options are 'T' or 'F'.")
    if cd_nat not in ["T", "W", "U", "V", "F"]:
        raise ValueError("Invalid cd_nat. Options are 'T', 'W', 'U', 'V' or 'F'.")
    if psgn not in [1, -1]:
        raise ValueError("Invalid psgn. Options are 1 (scalar) or -1 (vector).")
    if (psgn == -1) and (da.dtype == bool):
        raise ValueError("psgn = -1 cannot be applied to a boolean DataArray.")
    if ("i" not in da.dims) or ("j" not in da.dims):
        raise ValueError("da must have dimensions 'i' and 'j'.")
    if isinstance(ihls, bool) or not isinstance(ihls, (int, np.integer)):
        raise TypeError("ihls must be an integer.")
    if ihls < 1:
        raise ValueError("ihls must be >= 1.")
    if (ihls > da.sizes["i"]) or (ihls + 2 > da.sizes["j"]):
        raise ValueError(
            f"ihls = {ihls} is too large for domain of size (j={da.sizes['j']}, "
            f"i={da.sizes['i']})."
        )
    if da.sizes["i"] % 2 != 0:
        raise ValueError(
            f"north fold requires an even i-dimension size, got {da.sizes['i']}."
        )

    # === Apply North Fold LBC === #
    # North fold defined for a T-point:
    if c_NFtype == "T":
        match cd_nat:
            case "T" | "W":
                da_jhls = (da
                           .isel(j=slice(-ihls-1, -1)) # Select (jpj-2, jpj-1) rows
                           .roll(i=-1, roll_coords=True) # Roll (periodic) row
                           .isel(i=slice(None, None, -1)) # Reverse the order of the row
                           .assign_coords(
                               i=da.coords["i"],
                               j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                               )
                            .sortby("j")
                            )
            case "U":
                da_jhls = (da
                            .isel(j=slice(-ihls-1, -1)) # Select (jpj-2, jpj-1) rows
                            .isel(i=slice(None, None, -1)) # Reverse the order of the row
                            .assign_coords(
                            i=da.coords["i"],
                            j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                            )
                            .sortby("j")
                            )
            case "V":
                da_jhls = (da
                            .isel(j=slice(-ihls-2, -2)) # Select (jpj-3, jpj-2) rows
                            .roll(i=-1, roll_coords=True) # Roll (periodic) row
                            .isel(i=slice(None, None, -1)) # Reverse the order of the row
                            .assign_coords(
                            i=da.coords["i"],
                            j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                            )
                            .sortby("j")
                            )
            case "F":
                da_jhls = (da
                           .isel(j=slice(-ihls-2, -2)) # Select (jpj-3, jpj-2) rows
                           .isel(i=slice(None, None, -1)) # Reverse the order of the row
                           .assign_coords(
                               i=da.coords["i"],
                               j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                               )
                           .sortby("j")
                           )
    # North fold defined for an F-point:
    elif c_NFtype == "F":
        match cd_nat:
            case "T" | "W":
                da_jhls = (da
                           .isel(j=slice(-ihls, None)) # Select (jpj-1, jpj) rows
                           .isel(i=slice(None, None, -1)) # Reverse the order of the row
                           .assign_coords(
                               i=da.coords["i"],
                               j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                               )
                           .sortby("j")
                           )
            case "U":
                da_jhls = (da
                            .isel(j=slice(-ihls, None)) # Select (jpj-1, jpj) rows
                            .roll(i=1, roll_coords=True) # Roll (periodic) row
                            .isel(i=slice(None, None, -1)) # Reverse the order of the row
                            .assign_coords(
                                i=da.coords["i"],
                                j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                            )
                            .sortby("j")
                            )
            case "V":
                da_jhls = (da
                            .isel(j=slice(-ihls-1, -1)) # Select (jpj-2, jpj-1) rows
                            .isel(i=slice(None, None, -1)) # Reverse the order of the row
                            .assign_coords(
                                i=da.coords["i"],
                                j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                            )
                            .sortby("j")
                            )
            case "F":
                da_jhls = (da
                           .isel(j=slice(-ihls-1, -1)) # Select (jpj-2, jpj-1) rows
                           .roll(i=1, roll_coords=True) # Roll (periodic) row
                           .isel(i=slice(None, None, -1)) # Reverse the order of the row
                           .assign_coords(
                               i=da.coords["i"],
                               j=da.coords["j"].values[-1] + np.arange(ihls, 0, -1)
                               )
                           .sortby("j")
                           )

    # Apply sign across the north fold (e.g., -1 for vector components):
    if psgn == -1:
        da_jhls = -da_jhls

    # Concatenate j-halos to interior domain (north fold):
    da_nf = xr.concat(objs=[da, da_jhls], dim="j")

    # Add halos along i-dimension (zonal periodicity):
    d_ihls = {
        "lhs": da_nf.isel(i=slice(-ihls, None)).assign_coords(i=(da_nf.coords["i"].isel(i=slice(0,  ihls)) - ihls)),
        "rhs": da_nf.isel(i=slice(0, ihls)).assign_coords(i=(da_nf.coords["i"].isel(i=slice(-ihls, None)) + ihls)),
    }
    # Concatenate i-halos to north-folding domain:
    da_nfhls = xr.concat(objs=[d_ihls["lhs"], da_nf, d_ihls["rhs"]], dim="i")

    return da_nfhls

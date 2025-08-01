"""
mask.py

Description:
This module includes functions to compute land/sea masks 
for NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import xarray as xr
from .lbc import _apply_lbc_nfd


def _add_dom_msk(
    ka: xr.DataArray,
    top_level: xr.DataArray,
    bottom_level: xr.DataArray,
    msk: str = "T",
    iperio: bool = False,
    nfold: bool = False,
) -> xr.DataArray:
    """
    Compute land/ocean mask arrays at tracer points, horizontal velocity
    points (u & v) and vorticity points (f) points.

    Expected dimensions are: (y, x) for 2D arrays and (nav_lev) for
    1D vertical level array. All dimension arrays are expected to use
    zero-based indexing.

    See p.36 of the NEMO manual (DOI: 10.5281/zenodo.1464186) for details
    on the mask computation.

    Parameters
    ----------
    ka : xr.DataArray
        1D array of vertical levels indexes (Fortran-based indexing).
    top_level : xr.DataArray
        Top wet level in each grid column.
    bottom_level : xr.DataArray
        Bottom wet level in each grid column.
    msk : str, optional
        Type of mask to compute, by default "T" for tracer points.
    iperio : bool, optional
        If True, the domain is zonally periodic, by default False.
    nfold : bool, optional
        If True, the North Fold (NFold) is applied, by default False.

    Returns
    -------
    xr.DataArray
        The computed land/ocean mask for the specified grid point type.
    """
    if not isinstance(ka, xr.DataArray):
        raise TypeError("ka must be an xarray DataArray")
    if ka.ndim != 1:
        raise ValueError("ka must be a 1D xarray DataArray")
    if 'nav_lev' not in ka.dims:
        raise ValueError("ka must have a 'nav_lev' dimension")
    if not isinstance(top_level, xr.DataArray):
        raise TypeError("top_level must be an xarray DataArray")
    if 'x' not in top_level.dims or 'y' not in top_level.dims:
        raise ValueError("top_level must have dimensions 'x' and 'y'")
    if not isinstance(bottom_level, xr.DataArray):
        raise TypeError("bottom_level must be an xarray DataArray")
    if 'x' not in bottom_level.dims or 'y' not in bottom_level.dims:
        raise ValueError("bottom_level must have dimensions 'x' and 'y'")
    if msk not in ["T", "U", "V", "W", "F"]:
        raise ValueError("msk must be one of 'T', 'U', 'V', 'W', or 'F'")

    # -- Define t_mask from top/bottom_level -- #
    # Use Fortran-based indexing:
    top_level = top_level.assign_coords({"x": top_level["x"] + 1, "y": top_level["y"] + 1})
    bottom_level = bottom_level.assign_coords({"x": bottom_level["x"] + 1, "y": bottom_level["y"] + 1})

    ka = (ka + 1).expand_dims({"x": top_level["x"], "y": top_level["y"]})
    ka = ka.assign_coords({"nav_lev": ka["nav_lev"] + 1})

    # 1. Exclude land points:
    mask_1 = ~(ka < top_level)
    # 2. Keep wet cells between top and bottom levels:
    mask_2 = (ka >= top_level) & (bottom_level >= ka)
    # 3. Exclude points below the sea floor:
    mask_3 = ~(bottom_level < ka)

    t_mask = (mask_1 & mask_2 & mask_3).T

    # TODO: Handle F-folds as well as T-folds for eORCA1.
    match msk:
        case "T":
            mask = t_mask

        case "U":
            # Zonally Periodic Parent Domain:
            if iperio:
                tmask_end = t_mask.isel(x=0)
                tmask_end["x"] = t_mask["x"].max() + 1
                tmask = xr.concat([t_mask, tmask_end], dim="x")

                tmask_1 = tmask.isel({"x": slice(None, -1)})
                tmask_2 = tmask.isel({"x": slice(1, None)})
                tmask_2.coords["x"] = tmask_2.coords["x"] - 1

                mask = (tmask_1 * tmask_2)

            else:
                # Closed Parent / Child / Grandchild Domain - masked at first and last U-points:
                tmask_1 = t_mask.isel({"x": slice(None, -1)})
                tmask_2 = t_mask.isel({"x": slice(1, None)})
                tmask_2.coords["x"] = tmask_2.coords["x"] - 1

                mask = (tmask_1 * tmask_2).pad(x=(0, 1), constant_values=0)

        case "V":
            tmask_1 = t_mask.isel({"y": slice(None, -1)})
            tmask_2 = t_mask.isel({"y": slice(1, None)})
            tmask_2.coords["y"] = tmask_2.coords["y"] - 1

            mask = (tmask_1 * tmask_2).pad(y=(0, 1), constant_values=0)

            # North Fold (NFold) Lateral Boundary Condition:
            if nfold:
                mask = _apply_lbc_nfd(array_in=mask,
                                      sgn_in=1,
                                      ipj=mask.sizes["y"],
                                      jpiglo=mask.sizes["x"],
                                      nn_hls=0,
                                      grid_type="V"
                                      )

        case "W":
            wmask_1 = t_mask.isel({"nav_lev": slice(None, -1)})
            wmask_2 = t_mask.isel({"nav_lev": slice(1, None)})
            wmask_1.coords["nav_lev"] = wmask_1.coords["nav_lev"] + 1
            w_mask = (wmask_1 * wmask_2)

            # At k=1 -> sea surface, w_mask(i, j, 1) = tmask(i, j, 1)
            tmask_st = t_mask.isel({"nav_lev": 0})
            tmask_st.coords["nav_lev"] = w_mask.coords["nav_lev"].min() - 1
            mask = xr.concat([tmask_st, w_mask], dim="nav_lev").transpose("nav_lev", "y", "x")

        case "F":
            if iperio:
                # Zonally Periodic Parent Domain:
                tmask_end = t_mask.isel(x=0)
                tmask_end["x"] = t_mask["x"].max() + 1
                tmask = xr.concat([t_mask, tmask_end], dim="x")

                tmask_x1y1 = tmask.isel({"x": slice(None, -1), "y": slice(None, -1)})

                tmask_x2y1 = tmask.isel({"x": slice(1, None), "y": slice(None, -1)})
                tmask_x2y1.coords["x"] = tmask_x2y1.coords["x"] - 1

                tmask_x1y2= tmask.isel({"y": slice(1, None), "x": slice(None, -1)})
                tmask_x1y2.coords["y"] = tmask_x1y2.coords["y"] - 1

                tmask_x2y2 = tmask.isel({"x": slice(1, None), "y": slice(1, None)})
                tmask_x2y2.coords["x"] = tmask_x2y2.coords["x"] - 1
                tmask_x2y2.coords["y"] = tmask_x2y2.coords["y"] - 1

                mask = (tmask_x1y1 * tmask_x2y1 * tmask_x1y2 * tmask_x2y2).pad(y=(0, 1), constant_values=0)

            else:
                # Closed Parent / Child / Grandchild Domain:
                tmask_x1y1 = t_mask.isel({"x": slice(None, -1), "y": slice(None, -1)})

                tmask_x2y1 = t_mask.isel({"x": slice(1, None), "y": slice(None, -1)})
                tmask_x2y1.coords["x"] = tmask_x2y1.coords["x"] - 1

                tmask_x1y2= t_mask.isel({"y": slice(1, None), "x": slice(None, -1)})
                tmask_x1y2.coords["y"] = tmask_x1y2.coords["y"] - 1

                tmask_x2y2 = t_mask.isel({"x": slice(1, None), "y": slice(1, None)})
                tmask_x2y2.coords["x"] = tmask_x2y2.coords["x"] - 1
                tmask_x2y2.coords["y"] = tmask_x2y2.coords["y"] - 1

                mask = (tmask_x1y1 * tmask_x2y1 * tmask_x1y2 * tmask_x2y2).pad(x=(0, 1), y=(0, 1), constant_values=0)

    # Reassign coordinates to use zero-based indexing:
    mask = mask.assign_coords(
        {"nav_lev": mask["nav_lev"] - 1,
         "y": mask["y"] - 1,
         "x": mask["x"] - 1
         })

    return mask

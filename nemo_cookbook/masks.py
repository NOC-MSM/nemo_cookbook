"""
mask.py

Description:
This module includes functions to compute land/sea masks,
polygon masks and masked statistics for NEMO ocean general
circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import xarray as xr
from skimage import measure
from matplotlib.path import Path

from .lbc import lbc_nfd


def read_dom_mask(
    ka: xr.DataArray,
    ds_domain: xr.Dataset,
    cd_nat: str,
) -> xr.DataArray:
    """
    Read land/ocean mask arrays at tracer points, horizontal velocity
    points (u & v) and vorticity points (f) points from domain dataset.

    Expected dimensions are: (y, x) for 2D arrays and (nav_lev) for
    1D vertical level array. All dimensions are expected to use
    zero-based indexing.

    Parameters
    ----------
    ka : xr.DataArray
        1D array of vertical levels indexes (Fortran-based indexing).
    ds_domain : xr.Dataset
        NEMO model domain dataset.
    cd_nat : str
        Nature of array grid-points to read land/sea mask for.
        Options are 'T', 'W', 'U', 'V' or 'F'.

    Returns
    -------
    xr.DataArray
        Read land/ocean mask for the specified grid point type.
    """
    if not isinstance(ka, xr.DataArray):
        raise TypeError("ka must be an xarray DataArray")
    if ka.ndim != 1:
        raise ValueError("ka must be a 1D xarray DataArray")
    if not isinstance(ds_domain, xr.Dataset):
        raise TypeError("ds_domain must be an xarray Dataset")
    if cd_nat not in ["T", "U", "V", "W", "F"]:
        raise ValueError("cd_nat must be one of 'T', 'U', 'V', 'W', or 'F'")

    # -- Read mask from domain dataset -- #
    try:
        mask = ds_domain[f"{cd_nat.lower()}mask"]
    except AttributeError as e:
        raise AttributeError(
            f"missing '{cd_nat.lower()}mask' variable in domain dataset -> {e}"
        )

    # Update coordinates to use zero-based indexing:
    mask = mask.assign_coords({"nav_lev": ka, "y": mask["y"], "x": mask["x"]})

    return mask


def create_dom_mask(
    ka: xr.DataArray,
    top_level: xr.DataArray,
    bottom_level: xr.DataArray,
    cd_nat: str,
    c_NFtype: str = None,
    iperio: bool = False,
    mask_opensea: xr.DataArray = None,
) -> xr.DataArray:
    """
    Compute land/ocean mask arrays at tracer points, horizontal velocity
    points (u & v) and vorticity points (f) points.

    Expected dimensions are: (y, x) for 2D arrays and (nav_lev) for
    1D vertical level array. All dimensions are expected to use
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
    cd_nat : str
        Nature of array grid-points to compute land/sea mask for.
        Options are 'T', 'W', 'U', 'V' or 'F'.
    c_NFtype : str, optional
        Type of North Fold boundary condition to apply.
        Options are 'T' for T-point pivot or 'F' for F-point pivot, by default
        no North Fold boundary condition is applied.
    iperio : bool, optional
        If True, the domain is zonally periodic, by default False.
    mask_opensea : xr.DataArray, optional
        All closed seas are masked using mask_opensea, by default no masking is
        applied.

    Returns
    -------
    xr.DataArray
        The computed land/ocean mask for the specified grid point type.
    """
    if not isinstance(ka, xr.DataArray):
        raise TypeError("ka must be an xarray DataArray")
    if ka.ndim != 1:
        raise ValueError("ka must be a 1D xarray DataArray")
    if "nav_lev" not in ka.dims:
        raise ValueError("ka must have a 'nav_lev' dimension")
    if not isinstance(top_level, xr.DataArray):
        raise TypeError("top_level must be an xarray DataArray")
    if "x" not in top_level.dims or "y" not in top_level.dims:
        raise ValueError("top_level must have dimensions 'x' and 'y'")
    if not isinstance(bottom_level, xr.DataArray):
        raise TypeError("bottom_level must be an xarray DataArray")
    if "x" not in bottom_level.dims or "y" not in bottom_level.dims:
        raise ValueError("bottom_level must have dimensions 'x' and 'y'")
    if cd_nat not in ["T", "U", "V", "W", "F"]:
        raise ValueError("cd_nat must be one of 'T', 'U', 'V', 'W', or 'F'")
    if c_NFtype not in ["T", "F", None]:
        raise ValueError("c_NFtype must be one of 'T', 'F', or None")
    if not isinstance(iperio, bool):
        raise TypeError("iperio must be a boolean")
    if mask_opensea is not None and not isinstance(mask_opensea, xr.DataArray):
        raise TypeError("mask_opensea must be an xarray DataArray or None")

    # -- Define t_mask from top/bottom_level -- #
    # Use Fortran-based indexing for (nav_lev, x, y) coordinates:
    top_level = top_level.assign_coords(
        {"x": top_level["x"] + 1, "y": top_level["y"] + 1}
    )
    bottom_level = bottom_level.assign_coords(
        {"x": bottom_level["x"] + 1, "y": bottom_level["y"] + 1}
    )

    # Mask closed seas:
    if mask_opensea is not None:
        # Use Fortran-based indexing for (nav_lev, x, y) coordinates:
        mask_opensea = mask_opensea.assign_coords(
            {"x": mask_opensea["x"] + 1, "y": mask_opensea["y"] + 1}
        )
        # Apply open-sea mask to top & bottom levels:
        top_level = top_level.where(mask_opensea)
        bottom_level = bottom_level.where(mask_opensea)

    # Calculate tmask from vertical level indexes:
    ka = (ka + 1).expand_dims({"x": top_level["x"], "y": top_level["y"]})
    ka = ka.assign_coords({"nav_lev": ka["nav_lev"] + 1})

    # 1. Exclude land points:
    mask_1 = ~(ka < top_level)
    # 2. Keep wet cells between top and bottom levels:
    mask_2 = (ka >= top_level) & (bottom_level >= ka)
    # 3. Exclude points below the sea floor:
    mask_3 = ~(bottom_level < ka)

    # Define tmask for interior domain only:
    t_mask = (mask_1 & mask_2 & mask_3).transpose("nav_lev", "y", "x").load()

    if c_NFtype is not None:
        # Add 2 halos to tmask (x-direction & y-direction):
        t_mask_hls = t_mask.pad(x=(2, 2), constant_values=0)
        t_mask_hls["x"] = np.arange(-1, t_mask_hls["x"].size - 1)
        t_mask_hls = t_mask_hls.pad(y=(0, 2), constant_values=0)
        t_mask_hls["y"] = np.arange(1, t_mask_hls["y"].size + 1)

        # Apply NFold boundary condition to tmask:
        tmask = lbc_nfd(c_NFtype=c_NFtype, cd_nat="T", ihls=2, ptab=t_mask_hls, psgn=1)
    else:
        tmask = t_mask

    match cd_nat:
        case "T":
            mask = t_mask

        case "U":
            # NFold Domain:
            if c_NFtype is not None:
                # Zonally Periodic Domain:
                if iperio:
                    tmask[:, :, -2] = tmask[:, :, 2]

                # Calculate umask from tmask w halos & nfd applied:
                tmask_1 = tmask.isel({"x": slice(None, -1)})
                tmask_2 = tmask.isel({"x": slice(1, None)})
                tmask_2.coords["x"] = tmask_2.coords["x"] - 1

                u_mask_hls = (tmask_1 * tmask_2).pad(x=(0, 1), constant_values=0)
                u_mask_hls["x"] = np.arange(-1, u_mask_hls["x"].size - 1)

                # Apply NFold boundary condition to umask:
                u_mask_hls_nfd = lbc_nfd(
                    c_NFtype=c_NFtype, cd_nat="U", ihls=2, ptab=u_mask_hls, psgn=1
                )

                # Select umask for interior domain only:
                mask = u_mask_hls_nfd.isel(x=slice(2, -2), y=slice(None, -2))

            else:
                # Zonally Periodic Domain:
                if iperio:
                    tmask[:, :, -1] = tmask[:, :, 1]

                # Calculate umask from tmask w/o halos:
                tmask_1 = tmask.isel({"x": slice(None, -1)})
                tmask_2 = tmask.isel({"x": slice(1, None)})
                tmask_2.coords["x"] = tmask_2.coords["x"] - 1

                mask = (tmask_1 * tmask_2).pad(x=(0, 1), constant_values=0)
                mask["x"] = np.arange(1, mask["x"].size + 1)

        case "V":
            # Calculate vmask from tmask (w | w/o halos & nfd applied):
            tmask_1 = tmask.isel({"y": slice(None, -1)})
            tmask_2 = tmask.isel({"y": slice(1, None)})
            tmask_2.coords["y"] = tmask_2.coords["y"] - 1

            v_mask = (tmask_1 * tmask_2).pad(y=(0, 1), constant_values=0)
            v_mask["y"] = np.arange(1, v_mask["y"].size + 1)

            # NFold Domain:
            if c_NFtype is not None:
                # Apply NFold boundary condition to vmask:
                vmask_hls_nfd = lbc_nfd(
                    c_NFtype=c_NFtype, cd_nat="V", ihls=2, ptab=v_mask, psgn=1
                )

                # Select vmask for interior domain only:
                mask = vmask_hls_nfd.isel(x=slice(2, -2), y=slice(None, -2))

            else:
                mask = v_mask

        case "W":
            # Calculate wmask from tmask (w | w/o halos & nfd applied):
            wmask_1 = tmask.isel({"nav_lev": slice(None, -1)})
            wmask_2 = tmask.isel({"nav_lev": slice(1, None)})
            wmask_1.coords["nav_lev"] = wmask_1.coords["nav_lev"] + 1
            w_mask = wmask_1 * wmask_2

            # At k=1 -> sea surface, w_mask(i, j, 1) = tmask(i, j, 1)
            tmask_st = tmask.isel({"nav_lev": 0})
            tmask_st.coords["nav_lev"] = w_mask.coords["nav_lev"].min() - 1
            w_mask = xr.concat([tmask_st, w_mask], dim="nav_lev").transpose(
                "nav_lev", "y", "x"
            )

            # NFold Domain:
            if c_NFtype is not None:
                # Apply NFold boundary condition to wmask:
                wmask_hls_nfd = lbc_nfd(
                    c_NFtype=c_NFtype, cd_nat="W", ihls=2, ptab=w_mask, psgn=1
                )

                # Select wmask for interior domain only:
                mask = wmask_hls_nfd.isel(x=slice(2, -2), y=slice(None, -2))

            else:
                mask = w_mask

        case "F":
            # NFold Domain:
            if c_NFtype is not None:
                # Zonally Periodic Domain:
                if iperio:
                    # Zonally Periodic Parent Domain:
                    tmask[:, :, -2] = tmask[:, :, 2]

                # Calculate fmask from tmask w halos & nfd lbc:
                tmask_x1y1 = tmask.isel({"x": slice(None, -1), "y": slice(None, -1)})

                tmask_x2y1 = tmask.isel({"x": slice(1, None), "y": slice(None, -1)})
                tmask_x2y1.coords["x"] = tmask_x2y1.coords["x"] - 1

                tmask_x1y2 = tmask.isel({"y": slice(1, None), "x": slice(None, -1)})
                tmask_x1y2.coords["y"] = tmask_x1y2.coords["y"] - 1

                tmask_x2y2 = tmask.isel({"x": slice(1, None), "y": slice(1, None)})
                tmask_x2y2.coords["x"] = tmask_x2y2.coords["x"] - 1
                tmask_x2y2.coords["y"] = tmask_x2y2.coords["y"] - 1

                fmask_hls = (tmask_x1y1 * tmask_x2y1 * tmask_x1y2 * tmask_x2y2).pad(
                    x=(0, 1), y=(0, 1), constant_values=0
                )
                fmask_hls["x"] = np.arange(-1, fmask_hls["x"].size - 1)
                fmask_hls["y"] = np.arange(1, fmask_hls["y"].size + 1)

                # Apply NFold boundary condition to fmask:
                fmask_hls_nfd = lbc_nfd(
                    c_NFtype=c_NFtype, cd_nat="F", ihls=2, ptab=fmask_hls, psgn=1
                )

                # Select fmask for interior domain only:
                mask = fmask_hls_nfd.isel(x=slice(2, -2), y=slice(None, -2))

            else:
                # Zonally Periodic Domain:
                if iperio:
                    tmask[:, :, -1] = tmask[:, :, 1]

                # Calculate fmask from tmask w/o halos:
                tmask_x1y1 = tmask.isel({"x": slice(None, -1), "y": slice(None, -1)})

                tmask_x2y1 = tmask.isel({"x": slice(1, None), "y": slice(None, -1)})
                tmask_x2y1.coords["x"] = tmask_x2y1.coords["x"] - 1

                tmask_x1y2 = tmask.isel({"y": slice(1, None), "x": slice(None, -1)})
                tmask_x1y2.coords["y"] = tmask_x1y2.coords["y"] - 1

                tmask_x2y2 = tmask.isel({"x": slice(1, None), "y": slice(1, None)})
                tmask_x2y2.coords["x"] = tmask_x2y2.coords["x"] - 1
                tmask_x2y2.coords["y"] = tmask_x2y2.coords["y"] - 1

                mask = (tmask_x1y1 * tmask_x2y1 * tmask_x1y2 * tmask_x2y2).pad(
                    x=(0, 1), y=(0, 1), constant_values=0
                )
                mask["x"] = np.arange(1, mask["x"].size + 1)
                mask["y"] = np.arange(1, mask["y"].size + 1)

    # Update coordinates to use zero-based indexing:
    mask = mask.astype(bool).assign_coords(
        {"nav_lev": mask["nav_lev"] - 1, "y": mask["y"] - 1, "x": mask["x"] - 1}
    )

    return mask


def create_polygon_mask(
    lon_grid: xr.DataArray,
    lat_grid: xr.DataArray,
    lon_poly: list | np.ndarray,
    lat_poly: list | np.ndarray,
    dims: tuple[str] = ("j", "i"),
):
    """
    Create a mask of NEMO model grid points contained within a polygon.

    Parameters
    ----------
    lon_grid : DataArray
        Longitudes of NEMO model grid points.
    lat_grid : DataArray
        Latitudes of NEMO model grid points.
    lon_poly : ndarray
        Longitudes of polygon.
    lat_poly : ndarray
        Latitudes of polygon.
    dims : tuple[str]
        Dimensions of the output mask.

    Returns
    -------
    mask : DataArray
        Boolean mask identifying the model grid cells points which are
        contained (True) within the polygon.
    """
    # -- Validate Inputs -- #
    if not isinstance(lon_grid, xr.DataArray):
        raise TypeError("lon_grid must be an xarray DataArray")
    if not isinstance(lat_grid, xr.DataArray):
        raise TypeError("lat_grid must be an xarray DataArray")
    if not isinstance(lon_poly, (np.ndarray, list)):
        raise TypeError("lon_poly must be a numpy ndarray or list")
    if not isinstance(lat_poly, (np.ndarray, list)):
        raise TypeError("lat_poly must be a numpy ndarray or list")
    if not isinstance(dims, tuple) or len(dims) != 2:
        raise TypeError("dims must be a tuple of length 2")

    # Define model grid arrays:
    mdl_shape = lon_grid.shape
    lon_grid = lon_grid.values
    lat_grid = lat_grid.values

    # Prepare polygon:
    poly_coords = list(zip(lon_poly, lat_poly, strict=True))
    polygon = Path(poly_coords)

    # Define coords of polygon bbox:
    lon_min = np.nanmin(lon_poly)
    lon_max = np.nanmax(lon_poly)
    lat_min = np.nanmin(lat_poly)
    lat_max = np.nanmax(lat_poly)

    # Determine model coords within bbox:
    ind_bbox = np.argwhere(
        (lon_grid >= lon_min)
        & (lon_grid <= lon_max)
        & (lat_grid >= lat_min)
        & (lat_grid <= lat_max)
    )
    i_bbox = ind_bbox[:, 1]
    j_bbox = ind_bbox[:, 0]

    if i_bbox.size > 0:
        # Find model coordinates within bbox & polygon:
        lon_mdl = lon_grid[j_bbox, i_bbox]
        lat_mdl = lat_grid[j_bbox, i_bbox]
        mdl_coords = np.array(list(zip(lon_mdl, lat_mdl, strict=True)))
        mask_points = polygon.contains_points(mdl_coords)

        # Populate mask where model grid cells are within polygon:
        mask = np.zeros(mdl_shape, dtype=np.bool)
        mask[j_bbox[mask_points], i_bbox[mask_points]] = True
    else:
        mask = np.zeros(mdl_shape, dtype=np.bool)

    return xr.DataArray(data=mask, dims=dims)


def get_mask_boundary(mask: xr.DataArray) -> list[list[int]]:
    """
    Get the boundary coordinates of a mask.

    Parameters
    ----------
    mask : xr.DataArray
        Boolean mask identifying the model grid cells.

    Returns
    -------
    boundary : list[list[int]]
        List of boundary coordinates in the format [y, x].
    """
    # -- Extract mask boundary -- #
    mask_data = mask.astype("int").values
    boundary = measure.find_contours(image=mask_data, level=0.5, fully_connected="high")

    # -- Classify boundary grid points -- #
    # Store (i, j) coordinates on NEMO model grid.
    i_bdy, j_bdy, flux_type, flux_dir = [], [], [], []
    if len(boundary) == 1:
        bdy_points = boundary[0]
    else:
        raise RuntimeError(
            "multiple mask boundaries identified. Only single mask boundaries are supported."
        )

    for point in bdy_points:
        jp, ip = point[0], point[1]
        # Classify grid cell face of point - zonal:
        if (jp % 1 == 0) & (ip % 1 != 0):
            flux_type.append("U")
            # Classify flux direction across grid cell face:
            i_w, i_e = int(ip - 0.5), int(ip + 0.5)
            if mask_data[int(jp), i_w] == 1:
                i_bdy.append(i_w + 1.5)
                j_bdy.append(jp + 1)
                flux_dir.append(1)
            elif mask_data[int(jp), i_e] == 1:
                i_bdy.append(i_w + 1.5)
                j_bdy.append(jp + 1)
                flux_dir.append(-1)
            else:
                raise RuntimeError(
                    "cannot determine flux direction across grid cell face"
                )

        # Classify grid cell face of point - meridional:
        elif (jp % 1 != 0) & (ip % 1 == 0):
            flux_type.append("V")
            # Classify flux direction across meridional grid cell face:
            j_s, j_n = int(jp - 0.5), int(jp + 0.5)
            if mask_data[int(j_s), int(ip)] == 1:
                i_bdy.append(ip + 1)
                j_bdy.append(j_s + 1.5)
                flux_dir.append(1)

            elif mask_data[int(j_n), int(ip)] == 1:
                i_bdy.append(ip + 1)
                j_bdy.append(j_s + 1.5)
                flux_dir.append(-1)
            else:
                raise RuntimeError(
                    "cannot determine flux direction across grid cell face"
                )
        else:
            raise RuntimeError("boundary coordinate is not on a grid cell face")

    return i_bdy, j_bdy, flux_type, flux_dir

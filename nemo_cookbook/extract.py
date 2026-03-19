"""
extract.py

Description:
This module includes functions to extract hydrographic sections
and mask boundaries from NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime:
    from .nemodatatree import NEMODataTree

import dask
import numpy as np
import xarray as xr
from xoak import SklearnGeoBallTreeAdapter


def create_section_polygon(
    lon_sec: np.ndarray,
    lat_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a polygon using the geographical coordinates
    of a hydrographic section.

    Parameters
    ----------
    lon_sec: np.ndarray
        Longitudes defining the hydrographic section.
    lat_sec: np.ndarray
        Latitudes defining the hydrographic section.

    Returns
    -------
    lon_poly: np.ndarray
        Longitudes defining a closed polygon using the vertices
        of the hydrographic section.
    lat_poly: np.ndarray
        Latitudes defining a closed polygon using the vertices
        of the hydrographic section.
    """
    # Ensure longitudes are monotonically increasing:
    if lon_sec[0] > lon_sec[-1]:
        lon_sec = lon_sec[::-1]
        lat_sec = lat_sec[::-1]

    # Create closed polygon using endpoints of the hydrographic section:
    dlat = 1.0
    if lat_sec[0] < lat_sec[-1]:
        lon_poly = np.concatenate(
            [lon_sec, np.array([lon_sec[-1], lon_sec[0], lon_sec[0]])]
        )
        lat_poly = np.concatenate(
            [lat_sec, np.array([lat_sec[0] - dlat, lat_sec[0] - dlat, lat_sec[0]])]
        )
    elif lat_sec[0] > lat_sec[-1]:
        lon_poly = np.concatenate(
            [lon_sec, np.array([lon_sec[-1], lon_sec[0], lon_sec[0]])]
        )
        lat_poly = np.concatenate(
            [lat_sec, np.array([lat_sec[-1] - dlat, lat_sec[-1] - dlat, lat_sec[0]])]
        )
    elif all(lat == lat_sec[0] for lat in lat_sec):
        raise ValueError("extracting zonal hydrographic sections is not supported.")

    return lon_poly, lat_poly


def create_boundary_dataset(
    nemo: NEMODataTree,
    dom: str,
    i_bdy: list,
    j_bdy: list,
    flux_type: list,
    flux_dir: list,
) -> xr.Dataset:
    """
    Create a Dataset to store NEMO model coordinates and variables
    extracted along the boundary of a mask.

    Parameters
    ----------
    nemo : NEMODataTree
        NEMODataTree storing NEMO model outputs.
    dom : str
        Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
    i_bdy : list
        List of i-indexes along the mask boundary.
    j_bdy : list
        List of j-indexes along the mask boundary.
    flux_type : list
        List of flux types ('U' or 'V') along the mask boundary.
    flux_dir : list
        List of flux directions (+1 or -1) along the mask boundary.
    """
    # -- Get NEMO model grid properties -- #
    dom_prefix, _ = nemo._get_properties(dom=dom)
    grid_paths = nemo._get_grid_paths(dom=dom)
    gridT, gridU, gridV = grid_paths["gridT"], grid_paths["gridU"], grid_paths["gridV"]
    ijk_names = nemo._get_ijk_names(dom=dom)
    k_name = ijk_names["k"]
    time_name = [dim for dim in nemo[gridT].dims if "time" in dim][0]

    ds_bdy = xr.Dataset(
        data_vars={
            "i_bdy": (["bdy"], i_bdy[::-1]),
            "j_bdy": (["bdy"], j_bdy[::-1]),
            "flux_type": (["bdy"], flux_type[::-1]),
            "flux_dir": (["bdy"], flux_dir[::-1]),
        },
        coords={
            time_name: nemo[gridT][time_name].values,
            k_name: nemo[gridT][k_name].values,
            "bdy": np.arange(len(i_bdy)),
        },
    )

    # -- Add geographical coordinates along-section -- #
    ubdy_mask = ds_bdy["flux_type"] == "U"
    vbdy_mask = ds_bdy["flux_type"] == "V"
    dim_sizes = [
        nemo[gridU][time_name].size,
        nemo[gridU][k_name].size,
        ds_bdy["bdy"].size,
    ]

    ds_bdy = ds_bdy.assign_coords(
        {
            f"{dom_prefix}glamb": (["bdy"], np.zeros(ds_bdy["bdy"].size)),
            f"{dom_prefix}gphib": (["bdy"], np.zeros(ds_bdy["bdy"].size)),
            f"{dom_prefix}depthb": ((k_name, "bdy"), np.zeros(dim_sizes[1:])),
        }
    )

    ds_bdy[f"{dom_prefix}glamb"][ubdy_mask] = nemo[gridU][f"{dom_prefix}glamu"].sel(
        i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
    )
    ds_bdy[f"{dom_prefix}glamb"][vbdy_mask] = nemo[gridV][f"{dom_prefix}glamv"].sel(
        i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
    )

    ds_bdy[f"{dom_prefix}gphib"][ubdy_mask] = nemo[gridU][f"{dom_prefix}gphiu"].sel(
        i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
    )
    ds_bdy[f"{dom_prefix}gphib"][vbdy_mask] = nemo[gridV][f"{dom_prefix}gphiv"].sel(
        i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
    )
    ds_bdy[f"{dom_prefix}depthb"][:, ubdy_mask] = nemo[gridU][f"{dom_prefix}depthu"]
    ds_bdy[f"{dom_prefix}depthb"][:, vbdy_mask] = nemo[gridV][f"{dom_prefix}depthv"]

    return ds_bdy


def get_section_indexes(
    lon_section: np.ndarray,
    lat_section: np.ndarray,
    ds_bdy: xr.Dataset,
) -> list[int]:
    """
    Get indexes along a mask boundary corresponding to the start
    and end points of a hydrographic section.

    Parameters
    ----------
    lon_section : np.ndarray
        Longitudes defining hydrographic section.
    lat_section : np.ndarray
        Latitudes defining hydrographic section.
    ds_bdy : xr.Dataset
        Dataset containing mask boundary indexes.

    Returns
    -------
    list[int]
        List of boundary indexes corresponding to the hydrographic section.
    """
    # -- Find indexes of boundary start and end points -- #
    bdy_points = np.array(list(zip(ds_bdy["gphib"].values, ds_bdy["glamb"].values, strict=True)))
    geoballtree = SklearnGeoBallTreeAdapter(points=bdy_points, options={})

    # Collect indices of nearest boundary points to start and end points of section:
    bdy_start = geoballtree.query(points=np.array([[lat_section[0], lon_section[0]]]))[1]
    bdy_end = geoballtree.query(points=np.array([[lat_section[-1], lon_section[-1]]]))[1]

    # -- Define section in terms of boundary indexes -- #
    if bdy_start.size > 1:
        if (bdy_start[0] == 0) & (bdy_start[-1] == ds_bdy["bdy"][-1]):
            bdy_start = bdy_start[0]
    if bdy_end.size > 1:
        if (bdy_end[0] == 0) & (bdy_end[-1] == ds_bdy["bdy"][-1]):
            bdy_end = bdy_end[0]

    sec_start, sec_end = int(bdy_start.item()), int(bdy_end.item())
    if sec_end > sec_start:
        sec_indexes = np.arange(sec_start, sec_end + 1).tolist()
    elif sec_end < sec_start:
        # Note: start and end indexes are duplicated to close boundary -> do not include final index:
        sec_indexes = np.concatenate(
            [np.arange(sec_start, ds_bdy["bdy"][-2] + 1), np.arange(0, sec_end + 1)]
        ).tolist()
    else:
        raise ValueError(
            "start and end point indexes of hydrographic section are identical."
        )

    return sec_indexes


def update_boundary_dataset(
    nemo: NEMODataTree,
    dom: str,
    ds_bdy: xr.Dataset,
    uv_vars: list[str],
    vars: list[str] | None = None,
    sec_indexes: list[int] | None = None,
):
    """
    Update a mask boundary dataset with geographical coordinates,
    depths, velocities normal to the boundary and (optionally)
    scalar variables along a hydrographic section.

    Parameters
    ----------
    nemo : NEMODataTree
        NEMODataTree storing NEMO model outputs.
    dom : str
        Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
    ds_bdy : xr.Dataset
        Dataset containing variables and NEMO model coordinates extracted
        along the boundary of a mask.
    uv_vars : list, optional
        Names of velocity variables to extract along the hydrographic section.
        Default is ['uo', 'vo'].
    vars : list, optional
        Names of scalar variables to extract along the hydrographic section.
        Default is None.
    sec_indexes : list[int], optional
        List of boundary indexes corresponding to the hydrographic section.
        Default is None.

    Returns
    -------
    xr.Dataset
        Updated hydrographic section dataset extracted from NEMO model domain.
    """
    # -- Get NEMO model grid properties -- #
    grid_paths = nemo._get_grid_paths(dom=dom)
    gridT, gridU, gridV = grid_paths["gridT"], grid_paths["gridU"], grid_paths["gridV"]
    ijk_names = nemo._get_ijk_names(dom=dom)
    k_name = ijk_names["k"]
    time_name = [dim for dim in nemo[gridT].dims if "time" in dim][0]

    # -- Subset boundary dataset using section indexes -- #
    if sec_indexes is not None:
        ds_bdy = ds_bdy.isel(bdy=sec_indexes)
        ds_bdy["bdy"] = np.arange(len(sec_indexes))

    ubdy_mask = ds_bdy["flux_type"] == "U"
    vbdy_mask = ds_bdy["flux_type"] == "V"
    dim_2d_sizes = [
        nemo[gridU][k_name].size,
        ds_bdy["bdy"].size,
    ]
    dim_3d_sizes = [
        nemo[gridU][time_name].size,
        nemo[gridU][k_name].size,
        ds_bdy["bdy"].size,
    ]

    # -- Add velocities (outward) normal to boundary -- #
    if uv_vars[0] not in nemo[gridU].data_vars:
        raise KeyError(f"variable '{uv_vars[0]}' not found in grid '{gridU}'.")
    if uv_vars[1] not in nemo[gridV].data_vars:
        raise KeyError(f"variable '{uv_vars[1]}' not found in grid '{gridV}'.")

    ds_bdy["velocity"] = xr.DataArray(
        data=dask.array.zeros(dim_3d_sizes), dims=[time_name, k_name, "bdy"]
    )
    ds_bdy["velocity"][:, :, ubdy_mask] = (
        nemo[f"{gridU}/{uv_vars[0]}"].sel(
            i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
        )
        * ds_bdy["flux_dir"][ubdy_mask]
    )
    ds_bdy["velocity"][:, :, vbdy_mask] = (
        nemo[f"{gridV}/{uv_vars[1]}"].sel(
            i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
        )
        * ds_bdy["flux_dir"][vbdy_mask]
    )

    # -- Add NEMO land-sea mask along-section -- #
    ds_bdy["bmask"] = xr.DataArray(
        data=dask.array.zeros(dim_2d_sizes), dims=[k_name, "bdy"]
    )
    ds_bdy["bmask"][:, ubdy_mask] = nemo[f"{gridU}/umask"].sel(
        i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
    )
    ds_bdy["bmask"][:, vbdy_mask] = nemo[f"{gridV}/vmask"].sel(
        i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
    )

    # -- Add NEMO grid cell scale factors along boundary -- #
    ds_bdy["e1b"] = xr.DataArray(
        data=dask.array.zeros(ds_bdy["bdy"].size), dims=["bdy"]
    )
    ds_bdy["e1b"][ubdy_mask] = nemo[f"{gridU}/e2u"].sel(
        i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
    )
    ds_bdy["e1b"][vbdy_mask] = nemo[f"{gridV}/e1v"].sel(
        i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
    )

    ds_bdy["e3b"] = xr.DataArray(
        data=dask.array.zeros(dim_3d_sizes), dims=[time_name, k_name, "bdy"]
    )
    ds_bdy["e3b"][:, :, ubdy_mask] = nemo[f"{gridU}/e3u"].sel(
        i=ds_bdy["i_bdy"][ubdy_mask], j=ds_bdy["j_bdy"][ubdy_mask]
    )
    ds_bdy["e3b"][:, :, vbdy_mask] = nemo[f"{gridV}/e3v"].sel(
        i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask]
    )

    # -- [Optionally] Add scalar variables along-section -- #
    if vars is not None:
        for var in vars:
            if var in nemo[gridT].data_vars:
                ds_bdy[var] = xr.DataArray(
                    data=dask.array.zeros(dim_3d_sizes), dims=[time_name, k_name, "bdy"]
                )
            else:
                raise KeyError(f"variable {var} not found in grid '{gridT}'.")

            # Linearly interpolate scalar variables onto NEMO model U/V grid points:
            ds_bdy[var][:, :, ubdy_mask] = 0.5 * (
                nemo[f"{gridT}/{var}"].sel(
                    i=ds_bdy["i_bdy"][ubdy_mask] - 0.5, j=ds_bdy["j_bdy"][ubdy_mask]
                )
                + nemo[f"{gridT}/{var}"].sel(
                    i=ds_bdy["i_bdy"][ubdy_mask] + 0.5, j=ds_bdy["j_bdy"][ubdy_mask]
                )
            )
            ds_bdy[var][:, :, vbdy_mask] = 0.5 * (
                nemo[f"{gridT}/{var}"].sel(
                    i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask] - 0.5
                )
                + nemo[f"{gridT}/{var}"].sel(
                    i=ds_bdy["i_bdy"][vbdy_mask], j=ds_bdy["j_bdy"][vbdy_mask] + 0.5
                )
            )

    return ds_bdy

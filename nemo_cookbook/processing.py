"""
processing.py

Description:
This module provides utility functions for processing NEMO
ocean general circulation model grids.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import glob
import warnings

import numpy as np
import xarray as xr

from .masks import create_dom_mask, read_dom_mask, read_dom_maskutil

_DEFAULT_NBGHOST_CHILD = 4

def _add_parent_indices(
    ds: xr.Dataset,
    grid: str,
    parent: str,
    label: str
    ) -> xr.Dataset:
    """
    Add coordinates mapping parent domain (i, j) indices
    to child domain (ip_c, jp_c) indices.

    Parameters
    ----------
    ds : xr.Dataset
        NEMO model grid dataset.
    grid : str
        Name of the NEMO model grid (e.g. 'gridT', 'gridU', etc.).
    parent : str
        Identity of the NEMO model parent domain (e.g. '/', '1', '2', etc.).
    label : str
        Label to append to grid variable names.

    Returns
    -------
    xr.Dataset
        NEMO model grid dataset including coordinates mapping
        parent domain (i, j) indices to child domain (ip_c, jp_c)
        indices.
    """
    # Define parent domain label:
    plabel = "" if parent == "/" else parent

    # Define parent domain (i, j) indices for each child domain (ip_c, jp_c) index:
    if grid in ["gridU", "gridF"]:
        i_child = np.arange(ds.attrs["imin"] + 0.5, ds.attrs["imax"] + 0.5)
    else:
        i_child = np.arange(ds.attrs["imin"], ds.attrs["imax"])
    i_ic = xr.DataArray(
        np.repeat(i_child, repeats=ds.attrs["rx"]),
        dims=[f"i{label}"],
        coords={f"i{label}": ds[f"i{label}"]},
    )
    ds[f"i{plabel}_i{label}"] = i_ic
    ds[f"i{plabel}_i{label}"] = ds[f"i{plabel}_i{label}"].assign_attrs(
        name=f"i{plabel}_i{label}",
        long_name=f"i{plabel} indices of child domain i{label} indices",
    )

    if grid in ["gridV", "gridF"]:
        j_child = np.arange(ds.attrs["jmin"] + 0.5, ds.attrs["jmax"] + 0.5)
    else:
        j_child = np.arange(ds.attrs["jmin"], ds.attrs["jmax"])
    j_jc = xr.DataArray(
        np.repeat(j_child, repeats=ds.attrs["ry"]),
        dims=[f"j{label}"],
        coords={f"j{label}": ds[f"j{label}"]},
    )
    ds[f"j{plabel}_j{label}"] = j_jc
    ds[f"j{plabel}_j{label}"] = ds[f"j{plabel}_j{label}"].assign_attrs(
        name=f"j{plabel}_j{label}",
        long_name=f"j{plabel} indices of child domain j{label} indices",
    )

    ds = ds.assign_coords(
        {f"i{plabel}_i{label}": ds[f"i{plabel}_i{label}"],
         f"j{plabel}_j{label}": ds[f"j{plabel}_j{label}"]
        }
    )

    return ds


def _get_child_indices(
    imin: int,
    imax: int,
    jmin: int,
    jmax: int,
    rx: int,
    ry: int,
    nbghost_child: int,
) -> tuple[int, int, int, int]:
    """
    Get the indices which define the child domain within the parent domain.

    Parameters
    ----------
    imin, imax, jmin, jmax : int
        Indices defining the child domain within the parent domain.
    rx, ry : int
        Horizontal refinement factors.
    nbghost_child : int
        Number of ghost cells to remove from the western/southern
        boundaries of the child domain.

    Returns
    -------
    tuple of int
        Indices defining the parent domain nest within the child domain.
    """
    nbghost_e, nbghost_n, nbghost_w, nbghost_s = (
        nbghost_child,
        nbghost_child,
        nbghost_child,
        nbghost_child,
    )

    imin_c = 1
    imax_c = (imax - imin) * rx + nbghost_w + nbghost_e
    jmin_c = 1
    jmax_c = (jmax - jmin) * ry + nbghost_s + nbghost_n

    # Determine start and end indices excluding ghost cells:
    # Note: -1 is applied due to Python zero-based indexing.
    ist1 = imin_c + nbghost_w - 1
    iend1 = imax_c - nbghost_w - 1

    jst1 = jmin_c + nbghost_s - 1
    jend1 = jmax_c - nbghost_s - 1

    return (ist1, iend1, jst1, jend1)


def _check_grid_dims(ds: xr.Dataset, grid: str) -> None:
    """
    Check grid dataset contains the required dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        NEMO model grid dataset.
    grid : str
        Name of NEMO model grid (e.g. 'gridT', 'gridU', etc.).

    Raises
    ------
    KeyError
        If one or more required dimensions are missing from the grid dataset.
    """
    # -- NEMO model domain -- #
    if grid == "domain":
        ds = ds.squeeze()
        core_dims = ["nav_lev", "y", "x"]
        if not all([True if dim in core_dims else False for dim in ds.dims]):
            raise KeyError(
                f"{tuple(ds.dims)} is missing or exceeding required dimensions {tuple(core_dims)} expected for domain dataset."
            )

    # -- NEMO model grids -- #
    else:
        # Define core NEMO grid dimensions:
        core_2d_dims = ["time_counter", "y", "x"]
        core_3d_dims = ["time_counter", f"depth{grid[-1].lower()}", "y", "x"]

        if f"depth{grid[-1].lower()}" in ds.dims:
            if not all([True if dim in ds.dims else False for dim in core_3d_dims]):
                raise KeyError(
                    f"missing one or more required dimensions {tuple(core_3d_dims)} in {grid} dataset."
                )
        else:
            if not all([True if dim in ds.dims else False for dim in core_2d_dims]):
                raise KeyError(
                    f"missing one or more required dimensions {tuple(core_2d_dims)} in {grid} dataset."
                )


def _check_grid_datasets(d: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """
    Check compatibility of NEMO model grid xarray Datasets for
    creating a NEMODataTree.

    Parameters
    ----------
    d: dict[str, xr.Dataset]
        Dictionary of xarray Datasets, including the domain and one or more
        NEMO model grids, structured as:
        {
            'domain': 'path/to/domain.nc',
            'gridT': 'path/to/gridT.nc',
            ...
        }

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of compatible xarray Datasets corresponding to the domain
        and T/U/V/W NEMO model grids.
    """
    # Check dict keys and value dtypes:
    if "domain" not in d.keys():
        raise KeyError("missing 'domain': xarray Dataset in dictionary.")

    grid_keys = ["domain", "gridT", "gridU", "gridV", "gridW", "gridF", "icemod"]
    if not all([key in grid_keys for key in d.keys()]):
        raise KeyError(f"incompatible key in {d.keys()}. Expecting {grid_keys}.")
    if not all(isinstance(val, xr.Dataset) for val in d.values()):
        raise TypeError(
            "input dictionary should contain only (str: xarray Dataset) entries."
        )

    for key in grid_keys:
        if key not in d.keys():
            # Populate missing NEMO grid with empty xarray.Dataset:
            d.update({key: xr.Dataset()})
        else:
            # Check required grid dimensions:
            _check_grid_dims(ds=d[key], grid=key)

    # Combining sea ice and scalar variables both stored on T-grid:
    if ("gridT" in d.keys()) and ("icemod" in d.keys()):
        d["gridT"] = xr.merge([d["gridT"], d["icemod"]], compat="override", join="override")

    return d


def _open_grid_datasets(
    d_in: dict[str, str], **open_kwargs: dict[str, any]
) -> dict[str, xr.Dataset]:
    """
    Create Dictionary of grid datasets defining a NEMO model domain.

    Parameters
    ----------
    d_in : dict[str, str]
        Dictionary containing paths to NEMO grid output files, structured as:
        {
            'domain': 'path/to/domain.nc',
            'gridT': 'path/to/gridT.nc',
            'gridU': 'path/to/gridU.nc',
            'gridV': 'path/to/gridV.nc',
            'gridW': 'path/to/gridW.nc'
        }

    **open_kwargs: dict[str, any]
        Additional keyword arguments to pass to xarray.open_dataset or xarray.open_mfdataset
        when opening NEMO model grid files.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary containing NEMO grid output datasets, structured as:
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset
        }
    """
    # Define dictionary to store NEMO grid datasets:
    d_data = {}

    # NEMO model domain:
    if "domain" in d_in:
        try:
            d_data["domain"] = xr.open_dataset(d_in["domain"])
            _check_grid_dims(ds=d_data["domain"], grid="domain")
        except FileNotFoundError as e:
            raise FileNotFoundError("could not open domain configuration file") from e
    else:
        raise KeyError("missing 'domain' key in paths dictionary.")

    # NEMO model grids datasets:
    for key in ["gridT", "gridU", "gridV", "gridW", "gridF", "icemod"]:
        if key in d_in:
            try:
                if len(glob.glob(d_in[key])) > 1:
                    d_data[key] = xr.open_mfdataset(d_in[key], **open_kwargs)
                else:
                    d_data[key] = xr.open_dataset(d_in[key], **open_kwargs)
                _check_grid_dims(ds=d_data[key], grid=key)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"could not open {key} file(s)") from e
        else:
            d_data[key] = xr.Dataset()

    if "icemod" in d_data.keys():
        # Combining sea ice and scalar variables both defined on T-points:
        d_data.update(
            {
                "gridT": xr.merge(
                    [d_data["gridT"], d_data["icemod"]], compat="override", join="override"
                )
            }
        )
        del d_data["icemod"]

    return d_data


def _add_scale_factors_and_coords(
    grid_type: str,
    ds_grid: xr.Dataset,
    ds_domain: xr.Dataset,
    linssh: bool = False,
    vco : str = "1d",
    vco_ref: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Append grid cell scale factors and geographical coordinates
    to a NEMO model grid dataset.

    Parameters
    ----------
    grid_type : str
        Type of NEMO model grid. Options include "t", "u", "v", "w" & "f".
    
    ds_grid : dict[str, xr.Dataset]
        Dataset of NEMO model grid variables.

    ds_domain: xr.Dataset
        Dataset of NEMO model domain ancillary variables.

    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by
        (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical coordinates are time-dependent and must be included
        in grid datasets. Default is False.

    vco : str = "1d"
        Vertical reference coordinates. Options are '1d' to use 1-dimensional vertical reference coordinates or '3d' to use 3-dimensional vertical reference coordinates (deptht, depthu, depthv, depthw, depthf). Default is '1d'.

    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files.
        Default is False.

    Returns
    -------
    xr.Dataset
        Dataset of NEMO model grid variables with grid scale factors appended.

    """
    # -- Validate Inputs -- #
    if grid_type not in ["t", "u", "v", "w", "f"]:
        raise ValueError("'grid_type' must be one of: ['t', 'u', 'v', 'w', 'f']")

    # Define horizontal and vertical grid types:
    hgrid_type = "t" if grid_type == "w" else grid_type
    vgrid_type = "w" if grid_type == "w" else "t"

    # -- Append Grid Scale Factors -- #
    try:
        # Note: W-grid is horizontally co-located with T-grid -> use (e1t, e2t).
        ds_grid[f"e1{grid_type}"] = ds_domain[f"e1{hgrid_type}"]
        ds_grid[f"e2{grid_type}"] = ds_domain[f"e2{hgrid_type}"]
        if linssh:
            ds_grid[f"e3{grid_type}"] = ds_domain[f"e3{grid_type}_0"]

        d_coords = {f"gphi{grid_type}": ds_domain[f"gphi{hgrid_type}"],
                    f"glam{grid_type}": ds_domain[f"glam{hgrid_type}"]
                    }

        if vco == "3d":
            # Use 3-dimensional vertical reference coords:
            # Note: T/U/V/F grid are vertically co-located -> use (gdep_0).
            if f"gdep{vgrid_type}_0" in ds_domain.data_vars:
                d_coords[f"depth{grid_type}"] = ds_domain[f"gdep{vgrid_type}_0"].rename({"nav_lev": f"depth{grid_type}"})
            else:
                raise KeyError(
                    f"missing required 3-dimensional vertical reference coordinate 'gdep{vgrid_type}_0' in domain dataset."
                )

        ds_grid = ds_grid.assign_coords(d_coords)

    except AttributeError as e:
        raise AttributeError(
            f"missing required {grid_type}-grid scale factor in domain dataset"
        ) from e

    # Reference vertical scale factors and water column heights:
    if vco_ref and (f"e3{grid_type}_0" in ds_domain.data_vars):
        if not linssh:
            # Add reference vertical scale factors:
            ds_grid[f"e3{grid_type}_0"] = ds_domain[f"e3{grid_type}_0"]

        # Add reference water column heights:
        ds_grid[f"h{grid_type}_0"] = (ds_domain[f"e3{grid_type}_0"]
                                      .where(cond=ds_grid[f"{grid_type}mask"])
                                      .sum(dim="nav_lev")
                                      )

    return ds_grid


def _add_land_sea_masks(
    grid_type: str,
    ds_grid: xr.Dataset,
    ds_domain: xr.Dataset,
    iperio: bool = False,
    nftype: str | None = None,
    read_mask: bool = False,
    maskcs: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Append land-sea masks to a NEMO model grid dataset.

    Parameters
    ----------
    grid_type : str
        Type of NEMO model grid. Options include "t", "u", "v", "w" & "f".
    
    ds_grid : dict[str, xr.Dataset]
        Dataset of NEMO model grid variables.

    ds_domain: xr.Dataset
        Dataset of NEMO model domain ancillary variables.

    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by
        (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical coordinates are time-dependent and must be included
        in grid datasets. Default is False.

    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files.
        Default is False.

    Returns
    -------
    xr.Dataset
        Dataset of NEMO model grid variables with grid scale factors appended.

    """
    # -- Validate Inputs -- #
    if grid_type not in ["t", "u", "v", "w", "f"]:
        raise ValueError("'grid_type' must be one of: ['t', 'u', 'v', 'w', 'f']")

    # -- Mask Closed Seas -- #
    if maskcs:
        if "mask_opensea" in ds_domain.data_vars:
            mask_opensea = ds_domain["mask_opensea"]
        else:
            raise KeyError("missing required 'mask_opensea' variable in domain dataset.")
    else:
        mask_opensea = None

    # -- Append Land-Sea Masks -- #
    # Define vertical grid indices:
    ka = xr.DataArray(np.arange(ds_domain["nav_lev"].size), dims="nav_lev")

    if read_mask & (f"{grid_type}mask" in ds_domain.data_vars):
        # Read available 3-D mask from domain Dataset:
        ds_grid[f"{grid_type}mask"] = read_dom_mask(ka=ka, ds_domain=ds_domain, cd_nat=grid_type.upper(), mask_opensea=mask_opensea)

        if f"{grid_type}maskutil" in ds_domain.data_vars:
            # Read available 2-D (unique point) mask from domain Dataset:
            # Note: W-grid is horizontally co-located with T-grid -> read tmaskutil.
            ds_grid[f"{grid_type}maskutil"] = read_dom_maskutil(ds_domain=ds_domain, cd_nat=grid_type.upper(), mask_opensea=mask_opensea)
        else:
            # Define 2-D (unique point) mask from 3-D mask:
            ds_grid[f"{grid_type}maskutil"] = ds_grid[f"{grid_type}mask"].isel(nav_lev=0).squeeze(drop=True)
    else:
        if read_mask:
            warnings.warn(
                f"{grid_type}mask not found in domain dataset. Creating {grid_type}mask from top_level and bottom_level variables.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Calculate 3-D mask from top_level and bottom_level:
        ds_grid[f"{grid_type}mask"] = create_dom_mask(
            ka=ka,
            top_level=ds_domain["top_level"],
            bottom_level=ds_domain["bottom_level"],
            cd_nat=grid_type.upper(),
            c_NFtype=nftype,
            iperio=iperio,
            mask_opensea=mask_opensea,
        )
        # Define 2-D (unique point) mask from 3-D mask:
        ds_grid[f"{grid_type}maskutil"] = ds_grid[f"{grid_type}mask"].isel(nav_lev=0).squeeze(drop=True)

    return ds_grid


def _add_domain_vars(
    d_grids: dict[str, xr.Dataset],
    linssh: bool = False,
    vco: str = "1d",
    vco_ref: bool = False,
    iperio: bool = False,
    nftype: str | None = None,
    read_mask: bool = False,
    maskcs: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Append domain & mask variables to each grid dataset
    defining a NEMO model domain.

    Parameters
    ----------
    d_grids : dict[str, xr.Dataset]
        Dictionary containing NEMO grid datasets, structured as:
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset
        }
        
    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by
        (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical coordinates are time-dependent and must be included
        in grid datasets. Default is False.

    vco : str = "1d"
        Vertical reference coordinates. Options are '1d' to use 1-dimensional vertical reference coordinates or '3d' to use 3-dimensional vertical reference coordinates (deptht, depthu, depthv, depthw, depthf). Default is '1d'.

    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files.
        Default is False.

    iperio: bool = False
        Zonal periodicity of the domain.

    nftype: str | None = None
        Type of north fold lateral boundary condition to apply to domain. Options are 'T' for T-point pivot
        or 'F' for F-point pivot. By default, no north fold lateral boundary condition is applied (None).

    read_mask : bool = False
        If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed
        from top_level and bottom_level domain variables.

    maskcs : bool = False
        If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary containing NEMO grid datasets including domain variables, structured as:
        {
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
            'gridF': xr.Dataset
        }
    """
    # -- Open Domain Dataset -- #
    if "domain" in d_grids:
        # Remove singleton dimensions from domain dataset:
        domain = d_grids["domain"].squeeze(drop=True)
        # Drop legacy domain coordinates to prevent broadcasting conflicts:
        for coord in ("nav_lon", "nav_lat", "nav_lev", "y", "x"):
            if coord in domain:
                domain = domain.drop_vars(coord)
    else:
        raise KeyError("missing 'domain' key in grid datasets dictionary.")

    # -- Add Domain Variables -- #
    for grid_type in ["t", "u", "v", "w", "f"]:
        # Add land-sea masks:
        d_grids[f"grid{grid_type.upper()}"] = _add_land_sea_masks(grid_type=grid_type,
                                                                  ds_grid=d_grids[f"grid{grid_type.upper()}"],
                                                                  ds_domain=domain,
                                                                  iperio=iperio,
                                                                  nftype=nftype,
                                                                  read_mask=read_mask,
                                                                  maskcs=maskcs
                                                                  )

        # Add grid scale factors & coordinates:
        d_grids[f"grid{grid_type.upper()}"] = _add_scale_factors_and_coords(grid_type=grid_type,
                                                                            ds_grid=d_grids[f"grid{grid_type.upper()}"],
                                                                            ds_domain=domain,
                                                                            linssh=linssh,
                                                                            vco=vco,
                                                                            vco_ref=vco_ref
                                                                            )

    # -- Assign Global Attributes -- #
    d_grids[f"grid{grid_type.upper()}"] = d_grids[f"grid{grid_type.upper()}"].assign_attrs(nftype=nftype, iperio=iperio)

    return d_grids


def _process_grid(
    d_grids: dict[str, xr.Dataset],
    grid: str,
    label: str,
    i_slice: slice,
    j_slice: slice,
    i_name: str,
    j_name: str,
    k_name: str | None = None,
) -> xr.Dataset:
    """
    Process grid of a NEMO model domain.

    Parameters
    ----------
    d_grids : dict[str, xr.Dataset]
        Dictionary of grid datasets for NEMO model domain.
    grid : str
        Name of the grid to process (e.g., 'gridT', 'gridU', 'gridV').
    label : str
        Label to prepend to grid variable names.
    i_slice : slice
        Slice defining i-coordinate domain.
    j_slice : slice
        Slice defining the j-coordinate domain.
    i_name : str
        Name of i-coordinate.
    j_name : str
        Name of j-coordinate.
    k_name : str | None
        Name of k-coordinate. Default is None, meaning the grid dataset
        does not include a k-coordinate.

    Returns
    -------
    xr.Dataset
        Processed grid dataset with renamed variables and coordinates,
        sliced to the specified i and j indices, and with NEMO grid
        coordinate offsets applied.
    """
    # Define variable names and dimension mappings:
    grid_type = grid[-1].lower()
    mask_name = f"{label}{grid_type}mask"

    # Rename horizontal dimensions of grid:
    data = d_grids[grid].rename_dims({"y": j_name, "x": i_name})

    # Rename vertical dimension & grid coordinate variables:
    d_vars = {
        f"gphi{grid_type}": f"{label}gphi{grid_type}",
        f"glam{grid_type}": f"{label}glam{grid_type}",
    }
    if f"depth{grid_type}" in data.coords:
        data = data.rename_dims({f"depth{grid_type}": k_name})
        d_vars.update({f"depth{grid_type}": f"{label}depth{grid_type}"})
    if "nav_lev" in data.dims:
        d_vars.update({"nav_lev": k_name})

    # Rename mask variable:
    if mask_name in data.data_vars:
        d_vars.update({mask_name: f"{label}{mask_name}"})

    data = data.rename(d_vars)

    # Drop legacy coordinates:
    for coord in ("nav_lat", "nav_lon", "nav_lev", "y", "x"):
        if coord in data:
            data = data.drop_vars(coord)

    # Define domain size:
    data = data.isel({i_name: i_slice, j_name: j_slice})

    # Define offsets to transform to NEMO grid coordinates:
    match grid:
        case "gridT":
            i_offset, j_offset, k_offset = 1, 1, 1
        case "gridU":
            i_offset, j_offset, k_offset = 1.5, 1, 1
        case "gridV":
            i_offset, j_offset, k_offset = 1, 1.5, 1
        case "gridW":
            i_offset, j_offset, k_offset = 1, 1, 0.5
        case "gridF":
            i_offset, j_offset, k_offset = 1.5, 1.5, 1

    # Re-define to NEMO grid coords:
    d_coords = {
        j_name: data[j_name] + j_offset,
        i_name: data[i_name] + i_offset,
        f"{label}gphi{grid_type}": data[f"{label}gphi{grid_type}"],
        f"{label}glam{grid_type}": data[f"{label}glam{grid_type}"],
    }
    if k_name in data.coords:
        d_coords.update({k_name: data[k_name] + k_offset})
    data = data.assign_coords(d_coords)

    # Assign attrs for horizontal and vertical grid indices:
    data[i_name] = data[i_name].assign_attrs(
        name=i_name, long_name=f"{i_name} indices of NEMO model {grid[-1]}-points"
    )
    data[j_name] = data[j_name].assign_attrs(
        name=j_name, long_name=f"{j_name} indices of NEMO model {grid[-1]}-points"
    )
    if k_name in data.coords:
        data[k_name] = data[k_name].assign_attrs(
            name=k_name, long_name=f"{k_name} indices of NEMO model {grid[-1]}-points"
        )

    return data


def _process_parent(
    d_parent: dict[str, str] | dict[str, xr.Dataset],
    iperio: bool = False,
    nftype: str | None = None,
    read_mask: bool = False,
    maskcs: bool = False,
    linssh: bool = False,
    vco: str = "1d",
    vco_ref: bool = False,
    open_kwargs: dict[str, any] | None = None,
) -> dict[str, xr.Dataset]:
    """
    Create Dictionary of grid datasets defining a NEMO model parent domain.

    Parameters
    ----------
    d_parent : dict[str, str] | dict[str, xr.Dataset]
        Dictionary containing paths to or xarray Datasets created from NEMO parent grid output files,
        structured as:
        {
            'domain': 'path/to/parent_domain.nc',
            'gridT': 'path/to/parent_gridT.nc',
            'gridU': 'path/to/parent_gridU.nc',
            'gridV': 'path/to/parent_gridV.nc',
            'gridW': 'path/to/parent_gridW.nc',
            'icemod': 'path/to/parent_icemod.nc',
        }
        or
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
            'icemod': xr.Dataset
        }

    iperio: bool = False
        Zonal periodicity of the parent domain.

    nftype: str | None = None
        Type of north fold lateral boundary condition to apply to parent domain. Options are 'T' for T-point
        pivot or 'F' for F-point pivot. By default, no north fold lateral boundary condition is applied (None).

    read_mask : bool = False
        If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed
        from top_level and bottom_level domain variables. Default is False.

    maskcs : bool = False
        If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.

    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by
        (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical coordinates are time-dependent and must be included
        in grid datasets. Default is False.

    vco : str = "1d"
        Vertical reference variables. Options are '1d' to use 1-dimensional vertical reference coordinates or '3d' to use 3-dimensional vertical reference coordinates (deptht, depthu, depthv, depthw, depthf). Default is '1d'.  

    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files.
        Default is False.

    open_kwargs: dict[str, any], optional
        Additional keyword arguments to pass to xarray.open_dataset or xarray.open_mfdataset when opening
        parent grid files. Default is None.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary containing processed NEMO parent grid datasets, structured as:
        {
            '/': xr.Dataset,
            '/gridT': xr.Dataset,
            '/gridU': xr.Dataset,
            '/gridV': xr.Dataset,
            '/gridW': xr.Dataset,
            '/gridF': xr.Dataset
        }
    """
    # Define default open_kwargs:
    if open_kwargs is None:
        open_kwargs = {}

    # Open NEMO domain and grid datasets:
    if isinstance(d_parent, dict) and all(
        isinstance(entry, str) for entry in d_parent.values()
    ):
        d_grids = _open_grid_datasets(d_in=d_parent, **open_kwargs)
    elif isinstance(d_parent, dict) and all(
        isinstance(entry, xr.Dataset) for entry in d_parent.values()
    ):
        d_grids = _check_grid_datasets(d_parent)
    else:
        raise TypeError(
            "d_parent must be a dictionary of only paths or xarray Datasets."
        )

    # Add domain variables to each grid dataset:
    d_grids = _add_domain_vars(
        d_grids=d_grids, linssh=linssh, iperio=iperio, nftype=nftype,
        read_mask=read_mask, maskcs=maskcs, vco=vco, vco_ref=vco_ref
    )

    # Process T / U / V / W / F grids:
    d_proc_grids = {}
    for grid in ["gridT", "gridU", "gridV", "gridW", "gridF"]:
        d_proc_grids[grid] = _process_grid(
            d_grids=d_grids,
            grid=grid,
            label="",
            i_slice=slice(None),
            j_slice=slice(None),
            i_name="i",
            j_name="j",
            k_name="k",
        )

    # Define root node inheritable coords & attrs from first non-domain grid.
    root_name = [grid for grid in d_parent.keys() if grid != "domain"][0]
    # Handle case where icemod is only non-domain grid:
    root_name = "gridT" if root_name == "icemod" else root_name
    d_root = d_proc_grids[root_name].drop_dims(["j", "i", "k"])
    d_root.attrs = {"nftype": nftype, "iperio": iperio}

    # Construct DataTree node dictionary:
    d_out = {
        "/": d_root,
        "/gridT": d_proc_grids["gridT"],
        "/gridU": d_proc_grids["gridU"],
        "/gridV": d_proc_grids["gridV"],
        "/gridW": d_proc_grids["gridW"],
        "/gridF": d_proc_grids["gridF"],
    }

    return d_out


def _process_child(
    d_child: dict[dict[str, str]] | dict[dict[str, xr.Dataset]],
    d_nests: dict[str, str],
    label: int,
    parent_label: int,
    read_mask: bool = False,
    maskcs: bool = False,
    nbghost_child: int = _DEFAULT_NBGHOST_CHILD,
    linssh: bool = False,
    vco: str = "1d",
    vco_ref: bool = False,
    open_kwargs: dict[str, any] | None = None,
) -> dict[str, xr.Dataset]:
    """
    Create Dictionary of grid datasets defining a NEMO model (grand)child domain.

    Parameters
    ----------
    d_child : dict[dict[str, str]] | dict[dict[str, xr.Dataset]]
        Dictionary containing paths to or xarray Datasets created from NEMO (grand)child grid output files,
        structured as:
        {
            'domain': 'path/to/child_domain.nc',
            'gridT': 'path/to/child_gridT.nc',
            'gridU': 'path/to/child_gridU.nc',
            'gridV': 'path/to/child_gridV.nc',
            'gridW': 'path/to/child_gridW.nc',
            'icemod': 'path/to/child_icemod.nc',
        }
        or
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
            'icemod': xr.Dataset,
        }

    d_nests : dict[str, int]
        Dictionary describing the properties of the (grand)child domain, structured as:
        {
            'rx': rx,
            'ry': ry,
            'imin': imin,
            'imax': imax,
            'jmin': jmin,
            'jmax': jmax,
            'iperio': iperio
        }

    label : int
        Label for the (grand)child grid, used to differentiate between multiple (grand)child domains.

    parent_label : int
        Label for the parent domain, used to identify the child domain to which this grandchild grid belongs.
        Default is None, meaning a child domain is specified.

    read_mask : bool = False
        If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed
        from top_level and bottom_level domain variables. Default is False.

    maskcs : bool = False
        If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.

    nbghost_child : int = _DEFAULT_NBGHOST_CHILD
        Number of ghost cells to remove from the western/southern boundaries of the (grand)child domain.
        Default is 4 (`_DEFAULT_NBGHOST_CHILD`).

    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by
        (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical coordinates are time-dependent and must be included
        in grid datasets. Default is False.

    vco : str = "1d"
        Vertical reference variables. Options are '1d' to use 1-dimensional vertical reference coordinates or '3d' to use 3-dimensional vertical reference coordinates (deptht, depthu, depthv, depthw, depthf). Default is '1d'.

    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files.
        Default is False.

    open_kwargs: dict[str, any], optional
        Additional keyword arguments to pass to xarray.open_dataset or xarray.open_mfdataset when opening
        (grand)child grid files. Default is None.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary containing NEMO (grand)child grid output datasets, structured as:
        {
            f"/gridT/{label}_gridT": xr.Dataset,
            f"/gridU/{label}_gridU": xr.Dataset,
            f"/gridV/{label}_gridV": xr.Dataset,
            f"/gridW/{label}_gridW": xr.Dataset,
            f"/gridF/{label}_gridF": xr.Dataset
        }
        or
        {
            f"/gridT/{parent_label}_gridT/{label}_gridT": xr.Dataset,
            f"/gridU/{parent_label}_gridU/{label}_gridU": xr.Dataset,
            f"/gridV/{parent_label}_gridV/{label}_gridV": xr.Dataset,
            f"/gridW/{parent_label}_gridW/{label}_gridW": xr.Dataset,
            f"/gridF/{parent_label}_gridF/{label}_gridF": xr.Dataset
        }

    """
    # Define default open_kwargs:
    if open_kwargs is None:
        open_kwargs = {}

    # Open NEMO (grand)child domain and grid datasets:
    if isinstance(d_child, dict) and all(
        isinstance(entry, str) for entry in d_child.values()
    ):
        d_grids = _open_grid_datasets(d_in=d_child, **open_kwargs)
    elif isinstance(d_child, dict) and all(
        isinstance(entry, xr.Dataset) for entry in d_child.values()
    ):
        d_grids = _check_grid_datasets(d=d_child)
    else:
        raise TypeError(
            "d_child must be a dictionary of only paths or xarray Datasets."
        )

    # Add child domain variables to each grid:
    d_grids = _add_domain_vars(
        d_grids=d_grids, linssh=linssh, iperio=d_nests["iperio"], nftype=None,
        read_mask=read_mask, maskcs=maskcs, vco=vco, vco_ref=vco_ref
    )

    # Get child domain indices excluding ghost cells:
    ind_child = _get_child_indices(
        rx=d_nests.get("rx"),
        ry=d_nests.get("ry"),
        imin=d_nests.get("imin"),
        imax=d_nests.get("imax"),
        jmin=d_nests.get("jmin"),
        jmax=d_nests.get("jmax"),
        nbghost_child=nbghost_child,
    )
    i_slice = slice(ind_child[0], ind_child[1] + 1)
    j_slice = slice(ind_child[2], ind_child[3] + 1)

    # Process T / U / V / W / F grids:
    d_proc_grids = {}
    for grid in ["gridT", "gridU", "gridV", "gridW", "gridF"]:
        d_proc_grids[grid] = _process_grid(
            d_grids=d_grids,
            grid=grid,
            label=f"{label}_",
            i_slice=i_slice,
            j_slice=j_slice,
            i_name=f"i{label}",
            j_name=f"j{label}",
            k_name=f"k{label}",
        )

        # Add nest attributes & parent indices to child grids:
        d_proc_grids[grid] = _add_parent_indices(
            ds=d_proc_grids[grid].assign_attrs(
                {
                    "rx": d_nests.get("rx"),
                    "ry": d_nests.get("ry"),
                    "imin": d_nests.get("imin"),
                    "imax": d_nests.get("imax"),
                    "jmin": d_nests.get("jmin"),
                    "jmax": d_nests.get("jmax"),
                }
            ),
            grid=grid,
            parent=d_nests.get("parent"),
            label=label,
        )

    # Construct DataTree node path dictionary:
    if parent_label is not None:
        # Grandchild Domain -> Use /parent/child/grandchild node path:
        d_out = {
            f"/gridT/{parent_label}_gridT/{label}_gridT": d_proc_grids["gridT"],
            f"/gridU/{parent_label}_gridU/{label}_gridU": d_proc_grids["gridU"],
            f"/gridV/{parent_label}_gridV/{label}_gridV": d_proc_grids["gridV"],
            f"/gridW/{parent_label}_gridW/{label}_gridW": d_proc_grids["gridW"],
            f"/gridF/{parent_label}_gridF/{label}_gridF": d_proc_grids["gridF"],
        }
    else:
        # Child Domain -> Use /parent/child node path:
        d_out = {
            f"/gridT/{label}_gridT": d_proc_grids["gridT"],
            f"/gridU/{label}_gridU": d_proc_grids["gridU"],
            f"/gridV/{label}_gridV": d_proc_grids["gridV"],
            f"/gridW/{label}_gridW": d_proc_grids["gridW"],
            f"/gridF/{label}_gridF": d_proc_grids["gridF"],
        }

    return d_out


def create_datatree_dict(
    d_parent: dict[str, xr.Dataset] | dict[str, str],
    d_child: dict[str, dict[str, xr.Dataset]] | None = None,
    d_grandchild: dict[str, dict[str, xr.Dataset]] | None = None,
    nests: dict[str, dict[str, str]] | None = None,
    iperio: bool = False,
    nftype: str | None = None,
    read_mask: bool = False,
    maskcs: bool = False,
    nbghost_child: int = _DEFAULT_NBGHOST_CHILD,
    linssh: bool = False,
    vco: str = "1d",
    vco_ref: bool = False,
    open_kwargs: dict[str, any] | None = None,
) -> dict[str, xr.Dataset]:
    """
    Create Dictionary of DataTree paths (keys) and xarray Datasets (values)
    representing a collection of NEMO model grids.

    Parameters
    ----------
    d_parent : dict[str, xr.Dataset] | dict[str, str]
        Dictionary containing paths to or xarray Datasets created from NEMO parent grid output files.
    d_child : dict[str, dict[str, xr.Dataset]] | None, optional
        Dictionary containing paths to or xarray Datasets created from NEMO child grid output files.
    d_grandchild : dict[str, dict[str, xr.Dataset]] | None, optional
        Dictionary containing paths to or xarray Datasets created from NEMO grandchild grid output files.
    nests : dict[str, dict[str, str]] | None, optional
        Dictionary describing the properties of nested domains.
    iperio: bool = False
        Zonal periodicity of the parent domain.
    nftype: str | None = None
        Type of north fold lateral boundary condition to apply to parent domain. Options are 'T' for T-point
        pivot or 'F' for F-point pivot. By default, no north fold lateral boundary condition is applied (None).
    read_mask : bool = False
        If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed from top_level and bottom_level
        domain variables. Default is False.
    maskcs: bool = False
        If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.
    nbghost_child : int = _DEFAULT_NBGHOST_CHILD
        Number of ghost cells to remove from the western/southern boundaries of the (grand)child domain. Default is 4 (`_DEFAULT_NBGHOST_CHILD`).
    linssh: bool = False
        Linear free-surface approximation. If True, vertical coordinates are time-independent and given by (e3t_0, e3u_0, e3v_0, e3w_0). If False, vertical
        coordinates are time-dependent and must be included in grid datasets. Default is False.
    vco : str = "1d"
        Vertical reference variables. Options are '1d' to use 1-dimensional vertical reference coordinates or '3d' to use 3-dimensional vertical reference coordinates (deptht, depthu, depthv, depthw, depthf). Default is '1d'.  
    vco_ref: bool = False
        If True, add reference vertical scale factors and compute reference water column heights from domain files. Default is False.
    open_kwargs : dict[str, any], optional
        Additional keyword arguments passed to `xarray.open_dataset` or `xarray.open_mfdataset` when
        opening NEMO grid files. Default is None.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of DataTree paths and processed NEMO grids defining a hierarchical DataTree.
    """
    # Define default open_kwargs:
    if open_kwargs is None:
        open_kwargs = {}

    # -- Assign the parent domain -- #
    d_tree = _process_parent(
        d_parent=d_parent,
        iperio=iperio,
        read_mask=read_mask,
        maskcs=maskcs,
        nftype=nftype,
        linssh=linssh,
        vco=vco,
        vco_ref=vco_ref,
        open_kwargs=open_kwargs,
    )

    # -- Assign all child domains -- #
    if d_child is not None:
        if not all(isinstance(d_child[key], dict) for key in d_child.keys()):
            raise ValueError(
                "invalid child domain structure. Expected a nested dict defining NEMO child domain(s)."
            )
        for key in d_child.keys():
            if key not in nests.keys():
                raise KeyError(f"child domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if "parent" not in d_nests.keys():
                raise KeyError(
                    f"child nest dict '{key}' does not specify a parent domain."
                )
            d_tree.update(
                _process_child(
                    d_child=d_child[key],
                    d_nests=d_nests,
                    label=int(key),
                    parent_label=None,
                    read_mask=read_mask,
                    maskcs=maskcs,
                    nbghost_child=nbghost_child,
                    linssh=linssh,
                    vco=vco,
                    vco_ref=vco_ref,
                    open_kwargs=open_kwargs,
                )
            )

    # -- Assign all grandchild domains -- #
    if d_grandchild is not None:
        if not all(isinstance(d_grandchild[key], dict) for key in d_grandchild.keys()):
            raise ValueError(
                "invalid grandchild domain structure. Expected a nested dict defining NEMO grandchild domain(s)."
            )
        for key in d_grandchild.keys():
            if key not in nests.keys():
                raise KeyError(f"grandchild domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if "parent" not in d_nests.keys():
                raise KeyError(
                    f"grandchild nest dict '{key}' does not specify a parent domain."
                )
            if d_nests["parent"] not in d_child.keys():
                raise KeyError(
                    f"parent domain '{d_nests['parent']}' not found in child domains."
                )
            d_tree.update(
                _process_child(
                    d_child=d_grandchild[key],
                    d_nests=d_nests,
                    label=int(key),
                    parent_label=int(d_nests["parent"]),
                    read_mask=read_mask,
                    maskcs=maskcs,
                    nbghost_child=nbghost_child,
                    linssh=linssh,
                    vco=vco,
                    vco_ref=vco_ref,
                    open_kwargs=open_kwargs,
                )
            )

    return d_tree

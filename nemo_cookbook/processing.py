"""
processing.py

Description:
This module provides utility functions for processing NEMO
ocean general circulation model grids.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import glob
import numpy as np
import xarray as xr
from .masks import create_dom_mask


def _add_parent_indices(
    ds: xr.Dataset,
    grid: str,
    label: str
) -> xr.Dataset:
    """
    Add coordinates mapping parent domain (i, j) indices
    to child domain (i_c, j_c) indices.

    Parameters
    ----------
    ds : xr.Dataset
        NEMO model grid dataset.
    grid : str
        Name of the NEMO model grid (e.g. 'gridT', 'gridU', etc.).
    label : str
        Label to append to grid variable names.

    Returns
    -------
    xr.Dataset
        NEMO model grid dataset including coordinates mapping
        parent domain (i, j) indices to child domain (i_c, j_c)
        indices.
    """
    # Define parent domain (i, j) indices for each child domain (i_c, j_c) index:
    if grid in ['gridU', 'gridF']:
        i_child = np.arange(ds.attrs['imin'] + 0.5, ds.attrs['imax'] + 0.5)
    else:
        i_child = np.arange(ds.attrs['imin'], ds.attrs['imax'])
    i_ic = xr.DataArray(np.repeat(i_child, repeats=ds.attrs['rx']), dims=[f'i{label}'], coords={f'i{label}': ds[f'i{label}']})
    ds[f'i_i{label}'] = i_ic
    ds[f'i_i{label}'] = ds[f'i_i{label}'].assign_attrs(name=f'i_i{label}', long_name=f'parent domain i indices of child domain i{label} indices')

    if grid in ['gridV', 'gridF']:
        j_child = np.arange(ds.attrs['jmin'] + 0.5, ds.attrs['jmax'] + 0.5)
    else:
        j_child = np.arange(ds.attrs['jmin'], ds.attrs['jmax'])
    j_jc = xr.DataArray(np.repeat(j_child, repeats=ds.attrs['ry']), dims=[f'j{label}'], coords={f'j{label}': ds[f'j{label}']})
    ds[f'j_j{label}'] = j_jc
    ds[f'j_j{label}'] = ds[f'j_j{label}'].assign_attrs(name=f'j_j{label}', long_name=f'parent domain j indices of child domain j{label} indices')

    ds = ds.assign_coords({f'i_i{label}': ds[f'i_i{label}'], f'j_j{label}': ds[f'j_j{label}']})

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
    nbghost_e, nbghost_n, nbghost_w, nbghost_s = nbghost_child, nbghost_child, nbghost_child, nbghost_child

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


def _check_grid_dims(
    ds: xr.Dataset,
    grid: str
    ) -> None:
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
    if grid == 'domain':
        core_dims = ['nav_lev', 'y', 'x']
        if not all([True if dim in core_dims else False for dim in ds.dims]):
            raise KeyError(f"missing one or more required dimensions {tuple(core_dims)} in domain dataset.")

    # -- NEMO model grids -- #
    else:
        # Define core NEMO grid dimensions:
        core_2d_dims = ['time_counter', 'y', 'x']
        core_3d_dims = ['time_counter', f'depth{grid[-1].lower()}', 'y', 'x']

        if f'depth{grid[-1].lower()}' in ds.dims:
            if not all([True if dim in ds.dims else False for dim in core_3d_dims]):
                raise KeyError(f"missing one or more required dimensions {tuple(core_3d_dims)} in {grid} dataset.")
        else:
            if not all([True if dim in ds.dims else False for dim in core_2d_dims]):
                raise KeyError(f"missing one or more required dimensions {tuple(core_2d_dims)} in {grid} dataset.")


def _check_grid_datasets(
    d: dict[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
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
    if 'domain' not in d.keys():
        raise KeyError("missing 'domain': xarray Dataset in dictionary.")

    grid_keys = ['domain', 'gridT', 'gridU', 'gridV', 'gridW', 'icemod']
    if not all([key in grid_keys for key in d.keys()]):
        raise KeyError(f"incompatible key in {d.keys()}. Expecting {grid_keys}.")
    if not all(isinstance(val, xr.Dataset) for val in d.values()):
        raise TypeError("input dictionary should contain only (str: xarray Dataset) entries.")

    for key in grid_keys:
        if key not in d.keys():
            # Populate missing NEMO grid with empty xarray.Dataset:
            d.update({key: xr.Dataset()})
        else:
            # Check required grid dimensions:
            _check_grid_dims(ds=d[key], grid=key)

    # Combining sea ice and scalar variables both stored on T-grid:
    if ('gridT' in d.keys()) & ('icemod' in d.keys()):
        d['gridT'] = xr.merge([d['icemod'], d['gridT']], compat='no_conflicts')

    return d

def _open_grid_datasets(
    d_in: dict[str, str],
    **open_kwargs: dict[str, any]
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
    if 'domain' in d_in:
        try:
            d_data['domain'] = xr.open_dataset(d_in['domain'])
            _check_grid_dims(ds=d_data['domain'], grid='domain')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"could not open domain configuration file: {e}")
    else:
        raise KeyError("missing 'domain' key in paths dictionary.")

    # NEMO model grids datasets:
    for key in ['gridT', 'gridU', 'gridV', 'gridW', 'icemod']:
        if key in d_in:
            try:
                if len(glob.glob(d_in[key])) > 1:
                    d_data[key] = xr.open_mfdataset(d_in[key], **open_kwargs)
                else:
                    d_data[key] = xr.open_dataset(d_in[key], **open_kwargs)
                _check_grid_dims(ds=d_data[key], grid=key)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"could not open {key} file: {e}")
        else:
            d_data[key] = xr.Dataset()

    if 'icemod' in d_data.keys():
        # Combining sea ice and scalar variables both defined on T-points:
        d_data.update({'gridT': xr.merge([d_data['icemod'], d_data['gridT']], compat='no_conflicts')})
        del d_data['icemod']

    return d_data


def _add_domain_vars(
    d_grids: dict[str, xr.Dataset],
    iperio: bool = False,
    nftype: str | None = None
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

    iperio: bool = False
        Zonal periodicity of the domain.

    nftype: str | None = None
        Type of north fold lateral boundary condition to apply to domain. Options are 'T' for T-point pivot
        or 'F' for F-point pivot. By default, no north fold lateral boundary condition is applied (None).

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
    if 'domain' in d_grids:
        domain = d_grids['domain'].squeeze()
        # Drop all empty coordinates following squeeze:
        domain = domain.drop_vars([coord for coord in domain.coords if domain[coord].size == 1])
    else:
        raise KeyError("missing 'domain' key in grid datasets dictionary.")

    # Determine if closed seas should be masked:
    if "mask_opensea" in domain.data_vars:
        mask_opensea = domain["mask_opensea"]
    else:
        mask_opensea = None

    # Define vertical grid indices:
    ka = xr.DataArray(np.arange(domain["nav_lev"].size), dims='nav_lev')

    # T-grid:
    try:
        d_grids['gridT']['e1t'] = domain["e1t"]
        d_grids['gridT']['e2t'] = domain["e2t"]
        d_grids['gridT']['gphit'] = domain["gphit"]
        d_grids['gridT']['glamt'] = domain["glamt"]
        d_grids['gridT']['top_level'] = domain["top_level"]
        d_grids['gridT']['bottom_level'] = domain["bottom_level"]
    except AttributeError as e:
        raise AttributeError(f"missing required T-grid variable in domain dataset -> {e}")

    d_grids['gridT']['tmask'] = create_dom_mask(ka=ka,
                                                top_level=domain["top_level"],
                                                bottom_level=domain["bottom_level"],
                                                cd_nat="T",
                                                c_NFtype=nftype,
                                                iperio=iperio,
                                                mask_opensea=mask_opensea
                                                )
    d_grids['gridT']['tmaskutil'] = d_grids['gridT']['tmask'][0, :, :].squeeze(drop=True)
    d_grids['gridT'] = d_grids['gridT'].assign_attrs(nftype=nftype, iperio=iperio)

    # U-grid:
    try:
        d_grids['gridU']['e1u'] = domain["e1u"]
        d_grids['gridU']['e2u'] = domain["e2u"]
        d_grids['gridU']['gphiu'] = domain["gphiu"]
        d_grids['gridU']['glamu'] = domain["glamu"]
    except AttributeError as e:
        raise AttributeError(f"missing required U-grid variable in domain dataset -> {e}")

    d_grids['gridU']['umask'] = create_dom_mask(ka=ka,
                                                top_level=domain["top_level"],
                                                bottom_level=domain["bottom_level"],
                                                cd_nat="U",
                                                c_NFtype=nftype,
                                                iperio=iperio,
                                                mask_opensea=mask_opensea
                                                )
    d_grids['gridU']['umaskutil'] = d_grids['gridU']['umask'][0, :, :].squeeze(drop=True)
    d_grids['gridU'] = d_grids['gridU'].assign_attrs(nftype=nftype, iperio=iperio)

    # V-grid:
    try:
        d_grids['gridV']['e1v'] = domain["e1v"]
        d_grids['gridV']['e2v'] = domain["e2v"]
        d_grids['gridV']['gphiv'] = domain["gphiv"]
        d_grids['gridV']['glamv'] = domain["glamv"]
    except AttributeError as e:
        raise AttributeError(f"missing required V-grid variable in domain dataset -> {e}")

    d_grids['gridV']['vmask'] = create_dom_mask(ka=ka,
                                                top_level=domain["top_level"],
                                                bottom_level=domain["bottom_level"],
                                                cd_nat="V",
                                                c_NFtype=nftype,
                                                iperio=iperio,
                                                mask_opensea=mask_opensea
                                                )
    d_grids['gridV']['vmaskutil'] = d_grids['gridV']['vmask'][0, :, :].squeeze(drop=True)
    d_grids['gridV'] = d_grids['gridV'].assign_attrs(nftype=nftype, iperio=iperio)

    # W-grid:
    try:
        d_grids['gridW']['e1t'] = domain["e1t"]
        d_grids['gridW']['e2t'] = domain["e2t"]
        d_grids['gridW']['gphit'] = domain["gphit"]
        d_grids['gridW']['glamt'] = domain["glamt"]
    except AttributeError as e:
        raise AttributeError(f"missing required W-grid variable in domain dataset -> {e}")

    d_grids['gridW']['wmask'] = create_dom_mask(ka=ka,
                                                top_level=domain["top_level"],
                                                bottom_level=domain["bottom_level"],
                                                cd_nat="W",
                                                c_NFtype=nftype,
                                                iperio=iperio,
                                                mask_opensea=mask_opensea
                                                )
    d_grids['gridW'] = d_grids['gridW'].assign_attrs(nftype=nftype, iperio=iperio)

    # F-grid:
    d_grids['gridF'] = xr.Dataset()
    try:
        d_grids['gridF']['e1f'] = domain["e1f"]
        d_grids['gridF']['e2f'] = domain["e2f"]
        d_grids['gridF']['gphif'] = domain["gphif"]
        d_grids['gridF']['glamf'] = domain["glamf"]
    except AttributeError as e:
        raise AttributeError(f"missing required F-grid variable in domain dataset -> {e}")

    d_grids['gridF']['fmask'] = create_dom_mask(ka=ka,
                                                top_level=domain["top_level"],
                                                bottom_level=domain["bottom_level"],
                                                cd_nat="F",
                                                c_NFtype=nftype,
                                                iperio=iperio,
                                                mask_opensea=mask_opensea
                                                )
    d_grids['gridF']['fmaskutil'] = d_grids['gridF']['fmask'][0, :, :].squeeze(drop=True)
    d_grids['gridF'] = d_grids['gridF'].assign_attrs(nftype=nftype, iperio=iperio)

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
    # W-grid is horizontally co-located with T-grid, so use 'glamt' and 'gphit':
    hgrid_type = grid_type if grid_type in ('t', 'u', 'v', 'f') else 't'
    mask_name = f"{label}{grid_type}mask"

    # Rename horizontal dimensions of grid:
    data = d_grids[grid].rename_dims({"y": j_name, "x": i_name})

    # Rename vertical dimension & grid coordinate variables:
    d_vars = {f"gphi{hgrid_type}": f"{label}gphi{hgrid_type}",
              f"glam{hgrid_type}": f"{label}glam{hgrid_type}"
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
    for coord in ('nav_lat', 'nav_lon', 'nav_lev', 'y', 'x'):
        if coord in data:
            data = data.drop_vars(coord)

    # Define domain size:
    data = data.isel({i_name: i_slice, j_name: j_slice})

    # Define offsets to transform to NEMO grid coordinates:
    match grid:
        case 'gridT':
            i_offset, j_offset, k_offset = 1, 1, 1
        case 'gridU':
            i_offset, j_offset, k_offset = 1.5, 1, 1
        case 'gridV':
            i_offset, j_offset, k_offset = 1, 1.5, 1
        case 'gridW':
            i_offset, j_offset, k_offset = 1, 1, 0.5
        case 'gridF':
            i_offset, j_offset, k_offset = 1.5, 1.5, 1

    # Re-define to NEMO grid coords:
    d_coords = {j_name: data[j_name] + j_offset,
                i_name: data[i_name] + i_offset,
                f"{label}gphi{hgrid_type}": data[f"{label}gphi{hgrid_type}"],
                f"{label}glam{hgrid_type}": data[f"{label}glam{hgrid_type}"]
                }
    if k_name in data.coords:
        d_coords.update({k_name: data[k_name] + k_offset})
    data = data.assign_coords(d_coords)

    # Assign attrs for horizontal and vertical grid indices:
    data[i_name] = data[i_name].assign_attrs(name=i_name,
                                             long_name=f"{i_name} indices of NEMO model {grid[-1]}-points"
                                             )
    data[j_name] = data[j_name].assign_attrs(name=j_name,
                                             long_name=f"{j_name} indices of NEMO model {grid[-1]}-points"
                                             )
    if k_name in data.coords:
        data[k_name] = data[k_name].assign_attrs(name=k_name,
                                                 long_name=f"{k_name} indices of NEMO model {grid[-1]}-points"
                                                 )

    return data


def _process_parent(
    d_parent: dict[str, str] | dict[str, xr.Dataset],
    iperio: bool = False,
    nftype: str | None = None,
    open_kwargs: dict[str, any] = {}
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

    open_kwargs: dict[str, any], optional
        Additional keyword arguments to pass to xarray.open_dataset or xarray.open_mfdataset when opening
        parent grid files.

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
    # Open NEMO domain and grid datasets:
    if isinstance(d_parent, dict) and all(isinstance(entry, str) for entry in d_parent.values()):
        d_grids = _open_grid_datasets(d_in=d_parent, **open_kwargs)
    elif isinstance(d_parent, dict) and all(isinstance(entry, xr.Dataset) for entry in d_parent.values()):
        d_grids = _check_grid_datasets(d_parent)
    else:
        raise TypeError("d_parent must be a dictionary of only paths or xarray Datasets.")

    # Add domain variables to each grid dataset:
    d_grids = _add_domain_vars(d_grids=d_grids, iperio=iperio, nftype=nftype)

    # Process T / U / V / W / F grids:
    d_proc_grids = {}
    for grid in ['gridT', 'gridU', 'gridV', 'gridW', 'gridF']:
        d_proc_grids[grid] = _process_grid(d_grids=d_grids,
                                           grid=grid,
                                           label="",
                                           i_slice=slice(None),
                                           j_slice=slice(None),
                                           i_name="i",
                                           j_name="j",
                                           k_name="k",
                                           )

    # Define root node inheritable coords & attrs from first non-domain grid.
    root_name = [grid for grid in d_parent.keys() if grid != 'domain'][0]
    d_root = d_proc_grids[root_name].drop_dims(["j", "i", "k"])
    d_root.attrs = {'nftype': nftype, 'iperio': iperio}

    # Construct DataTree node dictionary:
    d_out = {
        "/": d_root,
        "/gridT": d_proc_grids['gridT'],
        "/gridU": d_proc_grids['gridU'],
        "/gridV": d_proc_grids['gridV'],
        "/gridW": d_proc_grids['gridW'],
        "/gridF": d_proc_grids['gridF']
            }

    return d_out


def _process_child(
    d_child: dict[dict[str, str]] | dict[dict[str, xr.Dataset]],
    d_nests: dict[str, str],
    label: int,
    parent_label: int,
    nbghost_child: int = 4,
    open_kwargs: dict[str, any] = {}
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

    nbghost_child : int = 4
        Number of ghost cells to remove from the western/southern boundaries of the (grand)child domain. Default is 4.

    open_kwargs: dict[str, any], optional
        Additional keyword arguments to pass to xarray.open_dataset or xarray.open_mfdataset when opening
        (grand)child grid files.

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
    # Open NEMO (grand)child domain and grid datasets:
    if isinstance(d_child, dict) and all(isinstance(entry, str) for entry in d_child.values()):
        d_grids = _open_grid_datasets(d_in=d_child, **open_kwargs)
    elif isinstance(d_child, dict) and all(isinstance(entry, xr.Dataset) for entry in d_child.values()):
        d_grids = _check_grid_datasets(d=d_child)
    else:
        raise TypeError("d_child must be a dictionary of only paths or xarray Datasets.")

    # Add child domain variables to each grid:
    d_grids = _add_domain_vars(d_grids=d_grids, iperio=d_nests['iperio'], nftype=None)

    # Get child domain indices excluding ghost cells:
    ind_child = _get_child_indices(rx=d_nests.get('rx'),
                                   ry=d_nests.get('ry'),
                                   imin=d_nests.get('imin'),
                                   imax=d_nests.get('imax'),
                                   jmin=d_nests.get('jmin'),
                                   jmax=d_nests.get('jmax'),
                                   nbghost_child=nbghost_child
                                   )
    i_slice = slice(ind_child[0], ind_child[1] + 1)
    j_slice = slice(ind_child[2], ind_child[3] + 1)

    # Process T / U / V / W / F grids:
    d_proc_grids = {}
    for grid in ['gridT', 'gridU', 'gridV', 'gridW', 'gridF']:        
        d_proc_grids[grid] = _process_grid(d_grids=d_grids,
                                           grid=grid,
                                           label=f"{label}_",
                                           i_slice=i_slice,
                                           j_slice=j_slice,
                                           i_name=f"i{label}",
                                           j_name=f"j{label}",
                                           k_name=f"k{label}",
                                           )

        # Add nest attributes & parent indices to child grids:
        d_proc_grids[grid] = _add_parent_indices(ds=d_proc_grids[grid]
                                                 .assign_attrs({
                                                     'rx': d_nests.get('rx'),
                                                     'ry': d_nests.get('ry'),
                                                     'imin': d_nests.get('imin'),
                                                     'imax': d_nests.get('imax'),
                                                     'jmin': d_nests.get('jmin'),
                                                     'jmax': d_nests.get('jmax')
                                                     }),
                                                 grid=grid,
                                                 label=label,
                                                 )

    # Construct DataTree node path dictionary:
    if parent_label is not None:
        # Grandchild Domain -> Use /parent/child/grandchild node path:
        d_out = {
            f"/gridT/{parent_label}_gridT/{label}_gridT": d_proc_grids['gridT'],
            f"/gridU/{parent_label}_gridU/{label}_gridU": d_proc_grids['gridU'],
            f"/gridV/{parent_label}_gridV/{label}_gridV": d_proc_grids['gridV'],
            f"/gridW/{parent_label}_gridW/{label}_gridW": d_proc_grids['gridW'],
            f"/gridF/{parent_label}_gridF/{label}_gridF": d_proc_grids['gridF']
        }
    else:
        # Child Domain -> Use /parent/child node path:
        d_out = {
            f"/gridT/{label}_gridT": d_proc_grids['gridT'],
            f"/gridU/{label}_gridU": d_proc_grids['gridU'],
            f"/gridV/{label}_gridV": d_proc_grids['gridV'],
            f"/gridW/{label}_gridW": d_proc_grids['gridW'],
            f"/gridF/{label}_gridF": d_proc_grids['gridF']
                } 

    return d_out


def create_datatree_dict(
    d_parent: dict[str, xr.Dataset] | dict[str, str],
    d_child: dict[str, dict[str, xr.Dataset]] | None = None,
    d_grandchild: dict[str, dict[str, xr.Dataset]] | None = None,
    nests: dict[str, dict[str, str]] | None = None,
    iperio: bool = False,
    nftype: str | None = None,
    nbghost_child: int = 4,
    open_kwargs: dict[str, any] = {}
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
    nbghost_child : int = 4
        Number of ghost cells to remove from the western/southern boundaries of the (grand)child domain.
        Default is 4.
    open_kwargs : dict[str, any], optional
        Additional keyword arguments passed to `xarray.open_dataset` or `xarray.open_mfdataset` when
        opening NEMO grid files.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of DataTree paths and processed NEMO grids defining a hierarchical DataTree.
    """
    # -- Assign the parent domain -- #
    d_tree = _process_parent(d_parent=d_parent, iperio=iperio, nftype=nftype, open_kwargs=open_kwargs)

    # -- Assign all child domains -- #
    if d_child is not None:
        if not all(isinstance(d_child[key], dict) for key in d_child.keys()):
            raise ValueError("invalid child domain structure. Expected a nested dict defining NEMO child domain(s).")
        for key in d_child.keys():
            if key not in nests.keys():
                raise KeyError(f"child domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if 'parent' not in d_nests.keys():
                raise KeyError(f"child nest dict '{key}' does not specify a parent domain.")
            d_tree.update(_process_child(d_child=d_child[key],
                                         d_nests=d_nests,
                                         label=int(key),
                                         parent_label=None,
                                         nbghost_child=nbghost_child,
                                         open_kwargs=open_kwargs
                                         ))

    # -- Assign all grandchild domains -- #
    if d_grandchild is not None:
        if not all(isinstance(d_grandchild[key], dict) for key in d_grandchild.keys()):
            raise ValueError("invalid grandchild domain structure. Expected a nested dict defining NEMO grandchild domain(s).")
        for key in d_grandchild.keys():
            if key not in nests.keys():
                raise KeyError(f"grandchild domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if 'parent' not in d_nests.keys():
                raise KeyError(f"grandchild nest dict '{key}' does not specify a parent domain.")
            if d_nests['parent'] not in d_child.keys():
                raise KeyError(f"parent domain '{d_nests['parent']}' not found in child domains.")
            d_tree.update(_process_child(d_child=d_grandchild[key],
                                         d_nests=d_nests,
                                         label=int(key),
                                         parent_label=int(d_nests['parent']),
                                         nbghost_child=nbghost_child,
                                         open_kwargs=open_kwargs
                                         ))

    return d_tree

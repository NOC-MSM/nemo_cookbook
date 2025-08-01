"""
processing.py

Description:
This module provides utility functions for processing NEMO
ocean general circulation model grids.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import xarray as xr
from .mask import _add_dom_msk


def _get_child_indices(
    imin: int,
    imax: int,
    jmin: int,
    jmax: int,
    rx: int,
    ry: int,
) -> tuple[int, int, int, int]:
    """
    Get the indices which define the child domain within the parent domain.
    
    Parameters
    ----------
    imin, imax, jmin, jmax : int
        Indices defining the child domain within the parent domain.
    rx, ry : int
        Horizontal refinement factors.
    
    Returns
    -------
    tuple of int
        Indices defining the parent domain nest within the child domain.
    """
    nbghost_e, nbghost_n, nbghost_w, nbghost_s = 4, 4, 4, 4

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


def _check_grid_datasets(
    d: dict[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
    """
    Check compatibility of NEMO model grid xarray Datasets for
    creating a NEMODataTree.

    Parameters
    ----------
    d: dict[str, xr.Dataset]
        A dictionary of xarray Datasets, including the domain and one or more
        NEMO model grids, structured as:
        {
            'domain': 'path/to/domain.nc',
            'gridT': 'path/to/gridT.nc',
            ...
        }

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary of compatible xarray Datasets corresponding to the domain
        and T/U/V/W NEMO model grids.
    """
    # Check dict keys and value dtypes:
    if 'domain' not in d.keys():
        raise KeyError("Missing 'domain': xarray Dataset in dictionary.")    
    grid_keys = ['domain', 'gridT', 'gridU', 'gridV', 'gridW']
    if not all([key in grid_keys for key in d.keys()]):
        raise KeyError(f"Incompatible key in {d.keys()}. Expecting {grid_keys}.")
    if not all(isinstance(val, xr.Dataset) for val in d.values()):
        raise TypeError("Input dictionary should contain only (str: xarray Dataset) entries.")

    # Populate missing NEMO grid keys with empty xarray Datasets:
    for key in grid_keys:
        if key not in d.keys():
            d.update({key: xr.Dataset()})

    return d

def _open_grid_datasets(
    d_in: dict[str, str]
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of grid datasets defining a NEMO model domain.

    Parameters
    ----------
    d_in : dict[str, str]
        A dictionary containing paths to NEMO grid output files, structured as:
        {
            'domain': 'path/to/domain.nc',
            'gridT': 'path/to/gridT.nc',
            'gridU': 'path/to/gridU.nc',
            'gridV': 'path/to/gridV.nc',
            'gridW': 'path/to/gridW.nc'
        }

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO grid output datasets, structured as:
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset
        }
    """
    # Domain Variables:
    if 'domain' in d_in:
        try:
            domain_cfg = xr.open_dataset(d_in['domain'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not open domain configuration file: {e}")
    else:
        raise KeyError("Missing 'domain' key in parent dictionary.")

    # T / U / V / W Grids:
    for key in ['gridT', 'gridU', 'gridV', 'gridW']:
        if key in d_in:
            try:
                dataset = xr.open_dataset(d_in[key])
                if key == 'gridT':
                    gridT = dataset
                elif key == 'gridU':
                    gridU = dataset
                elif key == 'gridV':
                    gridV = dataset
                elif key == 'gridW':
                    gridW = dataset
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not open {key} file: {e}")
        else:
            if key == 'gridT':
                gridT = xr.Dataset()
            elif key == 'gridU':
                gridU = xr.Dataset()
            elif key == 'gridV':
                gridV = xr.Dataset()
            elif key == 'gridW':
                gridW = xr.Dataset()

    d_out = {'domain': domain_cfg,
             "gridT": gridT,
             "gridU": gridU,
             "gridV": gridV,
             "gridW": gridW
            }

    return d_out


def _add_domain_vars(
    d_grids: dict[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
    """
    Append domain variables to each grid dataset
    defining a NEMO model domain.

    Parameters
    ----------
    d_grids : dict[str, xr.Dataset]
        A dictionary containing NEMO grid datasets, structured as:
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset
        }

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO grid datasets including domain variables, structured as:
        {
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
            'gridF': xr.Dataset
        }
    """
    if 'domain' in d_grids:
        domain = d_grids['domain']
    else:
        raise KeyError("Missing 'domain' key in grid datasets dictionary.")

    # T-grid:
    try:
        d_grids['gridT']['e1t'] = domain.e1t
        d_grids['gridT']['e2t'] = domain.e2t
        d_grids['gridT']['gphit'] = domain.gphit
        d_grids['gridT']['glamt'] = domain.glamt
        d_grids['gridT']['top_level'] = domain.top_level
        d_grids['gridT']['bottom_level'] = domain.bottom_level
    except AttributeError as e:
        raise AttributeError(f"Missing required T-grid variable in domain dataset -> {e}")
    if 'tmask' in domain.data_vars:
        d_grids['gridT']['tmask'] = domain.tmask
    else:
        d_grids['gridT'].attrs['Iperio'] = domain.attrs.get('Iperio', False).astype(bool)
        d_grids['gridT'].attrs['NFold'] = domain.attrs.get('NFold', False).astype(bool)
        d_grids['gridT']['tmask'] = _add_dom_msk(ka=domain.nav_lev,
                                                 top_level=domain.top_level,
                                                 bottom_level=domain.bottom_level,
                                                 msk="T",
                                                 iperio=d_grids['gridT'].attrs['Iperio'],
                                                 nfold=d_grids['gridT'].attrs['NFold']
                                                 )

    # U-grid:
    try:
        d_grids['gridU']['e1u'] = domain.e1u
        d_grids['gridU']['e2u'] = domain.e2u
        d_grids['gridU']['gphiu'] = domain.gphiu
        d_grids['gridU']['glamu'] = domain.glamu
    except AttributeError as e:
        raise AttributeError(f"Missing required U-grid variable in domain dataset -> {e}")
    if 'umask' in domain.data_vars:
        d_grids['gridU']['umask'] = domain.umask
    else:
        d_grids['gridU'].attrs['Iperio'] = domain.attrs.get('Iperio', False).astype(bool)
        d_grids['gridU'].attrs['NFold'] = domain.attrs.get('NFold', False).astype(bool)
        d_grids['gridU']['umask'] = _add_dom_msk(ka=domain.nav_lev,
                                                 top_level=domain.top_level,
                                                 bottom_level=domain.bottom_level,
                                                 msk="U",
                                                 iperio=d_grids['gridU'].attrs['Iperio'],
                                                 nfold=d_grids['gridU'].attrs['NFold']
                                                 )
    # V-grid:
    try:
        d_grids['gridV']['e1v'] = domain.e1v
        d_grids['gridV']['e2v'] = domain.e2v
        d_grids['gridV']['gphiv'] = domain.gphiv
        d_grids['gridV']['glamv'] = domain.glamv
    except AttributeError as e:
        raise AttributeError(f"Missing required V-grid variable in domain dataset -> {e}")
    if 'vmask' in domain.data_vars:
        d_grids['gridV']['vmask'] = domain.vmask
    else:
        d_grids['gridV'].attrs['Iperio'] = domain.attrs.get('Iperio', False).astype(bool)
        d_grids['gridV'].attrs['NFold'] = domain.attrs.get('NFold', False).astype(bool)
        d_grids['gridV']['vmask'] = _add_dom_msk(ka=domain.nav_lev,
                                                 top_level=domain.top_level,
                                                 bottom_level=domain.bottom_level,
                                                 msk="V",
                                                 iperio=d_grids['gridV'].attrs['Iperio'],
                                                 nfold=d_grids['gridV'].attrs['NFold']
                                                 )

    # W-grid:
    try:
        d_grids['gridW']['e1t'] = domain.e1t
        d_grids['gridW']['e2t'] = domain.e2t
        d_grids['gridW']['gphit'] = domain.gphit
        d_grids['gridW']['glamt'] = domain.glamt
    except AttributeError as e:
        raise AttributeError(f"Missing required W-grid variable in domain dataset -> {e}")
    if 'wmask' in domain.data_vars:
        d_grids['gridW']['wmask'] = domain.wmask
    else:
        d_grids['gridW'].attrs['Iperio'] = domain.attrs.get('Iperio', False).astype(bool)
        d_grids['gridW'].attrs['NFold'] = domain.attrs.get('NFold', False).astype(bool)
        d_grids['gridW']['wmask'] = _add_dom_msk(ka=domain.nav_lev,
                                                 top_level=domain.top_level,
                                                 bottom_level=domain.bottom_level,
                                                 msk="W",
                                                 iperio=d_grids['gridW'].attrs['Iperio'],
                                                 nfold=d_grids['gridW'].attrs['NFold']
                                                 )

    # F-grid:
    d_grids['gridF'] = xr.Dataset()
    try:
        d_grids['gridF']['e1f'] = domain.e1f
        d_grids['gridF']['e2f'] = domain.e2f
        d_grids['gridF']['gphif'] = domain.gphif
        d_grids['gridF']['glamf'] = domain.glamf
    except AttributeError as e:
        raise AttributeError(f"Missing required F-grid variable in domain dataset -> {e}")
    if 'fmask' in domain.data_vars:
        d_grids['gridF']['fmask'] = domain.fmask
    else:
        d_grids['gridF'].attrs['Iperio'] = domain.attrs.get('Iperio', False).astype(bool)
        d_grids['gridF'].attrs['NFold'] = domain.attrs.get('NFold', False).astype(bool)
        d_grids['gridF']['fmask'] = _add_dom_msk(ka=domain.nav_lev,
                                                 top_level=domain.top_level,
                                                 bottom_level=domain.bottom_level,
                                                 msk="F",
                                                 iperio=d_grids['gridF'].attrs['Iperio'],
                                                 nfold=d_grids['gridF'].attrs['NFold']
                                                 )

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
    
    return data


def _process_parent(
    d_parent: dict[str, str] | dict[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of grid datasets defining a NEMO model parent domain.

    Parameters
    ----------
    d_parent : dict[str, str] | dict[str, xr.Dataset]
        A dictionary containing paths to or xarray Datasets created from NEMO parent grid output files,
        structured as:
        {
            'domain': 'path/to/parent_domain.nc',
            'gridT': 'path/to/parent_gridT.nc',
            'gridU': 'path/to/parent_gridU.nc',
            'gridV': 'path/to/parent_gridV.nc',
            'gridW': 'path/to/parent_gridW.nc',
        }
        or
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
        }

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing processed NEMO parent grid datasets, structured as:
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
        d_grids = _open_grid_datasets(d_parent)
    elif isinstance(d_parent, dict) and all(isinstance(entry, xr.Dataset) for entry in d_parent.values()):
        d_grids = _check_grid_datasets(d_parent)
    else:
        raise TypeError("d_parent must be a dictionary of only paths or xarray Datasets.")

    # Add domain variables to each grid dataset:
    d_grids = _add_domain_vars(d_grids)

    # Process T / U / V / W / F grids:
    d_proc_grids = {}
    for grid in ['gridT', 'gridU', 'gridV', 'gridW', 'gridF']:
        d_proc_grids[grid] = _process_grid(d_grids=d_grids,
                                           grid=grid,
                                           label="",
                                           i_slice=slice(None),
                                           j_slice=slice(None),
                                           i_name="i",
                                           j_name= "j",
                                           k_name="k",
                                           )

    # Construct DataTree node dictionary:
    # Define root node using only inheritable coords.
    d_out = {
        "/": d_proc_grids['gridT'].drop_dims(["j", "i", "k"]),
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
    parent_label: int
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of grid datasets defining a NEMO model (grand)child domain.

    Parameters
    ----------
    d_child : dict[dict[str, str]] | dict[dict[str, xr.Dataset]]
        A dictionary containing paths to or xarray Datasets created from NEMO (grand)child grid output files,
        structured as:
        {
            'domain': 'path/to/child_domain.nc',
            'gridT': 'path/to/child_gridT.nc',
            'gridU': 'path/to/child_gridU.nc',
            'gridV': 'path/to/child_gridV.nc',
            'gridW': 'path/to/child_gridW.nc',
        }
        or
        {
            'domain': xr.Dataset,
            'gridT': xr.Dataset,
            'gridU': xr.Dataset,
            'gridV': xr.Dataset,
            'gridW': xr.Dataset,
        }
    
    d_nests : dict[str, int]
        A dictionary describing the properties of the (grand)child domain, structured as:
        {
            'rx': rx,
            'ry': ry,
            'imin': imin,
            'imax': imax,
            'jmin': jmin,
            'jmax': jmax,
        }

    label : int
        Label for the (grand)child grid, used to differentiate between multiple (grand)child domains.

    parent_label : int
        Label for the parent domain, used to identify the child domain to which this grandchild grid belongs.
        Default is None, meaning a child domain is specified.
    
    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO (grand)child grid output datasets, structured as:
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
        d_grids = _open_grid_datasets(d_child)
    elif isinstance(d_child, dict) and all(isinstance(entry, xr.Dataset) for entry in d_child.values()):
        d_grids = _check_grid_datasets(d_child)
    else:
        raise TypeError("d_child must be a dictionary of only paths or xarray Datasets.")

    # Add child domain variables to each grid:
    d_grids = _add_domain_vars(d_grids)

    # Get child domain indices excluding ghost cells:
    ind_child = _get_child_indices(rx=d_nests['rx'],
                                   ry=d_nests['ry'],
                                   imin=d_nests['imin'],
                                   imax=d_nests['imax'],
                                   jmin=d_nests['jmin'],
                                   jmax=d_nests['jmax']
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


def _create_datatree_dict(
    d_parent: dict[str, xr.Dataset] | dict[str, str],
    d_child: dict[str, dict[str, xr.Dataset]] | None = None,
    d_grandchild: dict[str, dict[str, xr.Dataset]] | None = None,
    nests: dict[str, dict[str, str]] | None = None,
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of DataTree paths (key) and xarray Datasets (values)
    representing a collection of NEMO model grids.

    Parameters
    ----------
    d_parent : dict[str, xr.Dataset] | dict[str, str]
        A dictionary containing paths to or xarray Datasets created from NEMO parent grid output files.
    d_child : dict[str, dict[str, xr.Dataset]] | None, optional
        A dictionary containing paths to or xarray Datasets created from NEMO child grid output files.
    d_grandchild : dict[str, dict[str, xr.Dataset]] | None, optional
        A dictionary containing paths to or xarray Datasets created from NEMO grandchild grid output files.
    nests : dict[str, dict[str, str]] | None, optional
        A dictionary describing the properties of nested domains.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of DataTree paths and processed NEMO grids defining a hierarchical DataTree.
    """
    # -- Assign the parent domain -- #
    d_tree = _process_parent(d_parent)

    # -- Assign all child domains -- #
    if d_child is not None:
        if not all(isinstance(d_child[key], dict) for key in d_child.keys()):
            raise ValueError("Invalid child domain structure. Expected a nested dict defining NEMO child domain(s).")
        for key in d_child.keys():
            if key not in nests.keys():
                raise KeyError(f"Child domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if 'parent' not in d_nests.keys():
                raise KeyError(f"Child nest dict '{key}' does not specify a parent domain.")
            d_tree.update(_process_child(d_child=d_child[key],
                                         d_nests=d_nests,
                                         label=int(key),
                                         parent_label=None
                                         ))

    # -- Assign all grandchild domains -- #
    if d_grandchild is not None:
        if not all(isinstance(d_grandchild[key], dict) for key in d_grandchild.keys()):
            raise ValueError("Invalid grandchild domain structure. Expected a nested dict defining NEMO grandchild domain(s).")
        for key in d_grandchild.keys():
            if key not in nests.keys():
                raise KeyError(f"Grandchild domain '{key}' not found in nests dict.")
            d_nests = nests[key]
            if 'parent' not in d_nests.keys():
                raise KeyError(f"Grandchild nest dict '{key}' does not specify a parent domain.")
            if d_nests['parent'] not in d_child.keys():
                raise KeyError(f"Parent domain '{d_nests['parent']}' not found in child domains.")
            d_tree.update(_process_child(d_child=d_grandchild[key],
                                         d_nests=d_nests,
                                         label=int(key),
                                         parent_label=int(d_nests['parent'])
                                         ))

    return d_tree

"""
processing.py

Description:
This module provides utility functions for processing NEMO
ocean general circulation model grids.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import xarray as xr


def _get_child_indices(imin, imax, jmin, jmax, rx, ry):
    """
    Get the indices within the child domain which define
    the nest within the parent domain.
    
    Parameters
    ----------
    imin, imax, jmin, jmax : int
        Indices defining the child domain (nest) within the parent domain.
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

    ist1 = imin_c + nbghost_w - 1
    iend1 = imax_c - nbghost_w -1

    jst1 = jmin_c + nbghost_s - 1
    jend1 = jmax_c - nbghost_s - 1

    return (ist1, iend1, jst1, jend1)


def _process_grid_datasets(
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
    # -- Domain -- # 
    if 'domain' in d_in:
        try:
            domain_cfg = xr.open_dataset(d_in['domain'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not open domain configuration file: {e}")
    else:
        raise KeyError("Missing 'domain' key in parent dictionary.")

    # -- T / U / V / W Grids -- #
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
    d_grids['gridT']['e1t'] = domain.e1t
    d_grids['gridT']['e2t'] = domain.e2t
    d_grids['gridT']['gphit'] = domain.gphit
    d_grids['gridT']['glamt'] = domain.glamt
    d_grids['gridT']['top_level'] = domain.top_level
    d_grids['gridT']['bottom_level'] = domain.bottom_level
    if 'tmask' in domain.data_vars:
        d_grids['gridT']['tmask'] = domain.tmask

    # U-grid:
    d_grids['gridU']['e1u'] = domain.e1u
    d_grids['gridU']['e2u'] = domain.e2u
    d_grids['gridU']['gphiu'] = domain.gphiu
    d_grids['gridU']['glamu'] = domain.glamu
    if 'umask' in domain.data_vars:
        d_grids['gridU']['umask'] = domain.umask

    # V-grid:
    d_grids['gridV']['e1v'] = domain.e1v
    d_grids['gridV']['e2v'] = domain.e2v
    d_grids['gridV']['gphiv'] = domain.gphiv
    d_grids['gridV']['glamv'] = domain.glamv
    if 'vmask' in domain.data_vars:
        d_grids['gridV']['vmask'] = domain.vmask

    # W-grid:
    d_grids['gridW']['e1t'] = domain.e1t
    d_grids['gridW']['e2t'] = domain.e2t
    d_grids['gridW']['gphit'] = domain.gphit
    d_grids['gridW']['glamt'] = domain.glamt
    if 'wmask' in domain.data_vars:
        d_grids['gridW']['wmask'] = domain.wmask

    # F-grid:
    d_grids['gridF'] = xr.Dataset()
    d_grids['gridF']['e1f'] = domain.e1f
    d_grids['gridF']['e2f'] = domain.e2f
    d_grids['gridF']['gphif'] = domain.gphif
    d_grids['gridF']['glamf'] = domain.glamf
    if 'fmask' in domain.data_vars:
        d_grids['gridF']['fmask'] = domain.fmask
    
    return d_grids


def _process_parent(
    d_parent: dict[str, str]
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of grid datasets defining a NEMO model parent domain.

    Parameters
    ----------
    d_parent : dict[str, str]
        A dictionary containing paths to NEMO parent grid output files, structured as:
        {
            'domain': 'path/to/parent_domain.nc',
            'gridT': 'path/to/parent_gridT.nc',
            'gridU': 'path/to/parent_gridU.nc',
            'gridV': 'path/to/parent_gridV.nc',
            'gridW': 'path/to/parent_gridW.nc',
        }

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO parent grid output datasets, structured as:
        {
            '/': xr.Dataset,
            '/gridT': xr.Dataset,
            '/gridU': xr.Dataset,
            '/gridV': xr.Dataset,
            '/gridW': xr.Dataset,
            '/gridF': xr.Dataset
        }
    """
    # -- Open NEMO domain and grid datasets -- #
    d_grids = _process_grid_datasets(d_parent)

    # -- Append domain variables to each grid dataset -- #
    d_grids = _add_domain_vars(d_grids)

    d_rename = {"y": "j", "x": "i"}

    # -- Construct T / U / V / W / F grids -- #
    gridT = d_grids["gridT"].rename_dims({**d_rename , **{"deptht": "k"}})
    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridT:
            gridT = gridT.drop_vars(coord)
    gridT = gridT.assign_coords({"k": gridT["k"] + 1,
                                 "j": gridT["j"] + 1,
                                 "i": gridT["i"] + 1,
                                 "gphit": gridT.gphit,
                                 "glamt": gridT.glamt
                                 })

    gridU = d_grids["gridU"].rename_dims({**d_rename , **{"depthu": "k"}})
    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridU:
            gridU = gridU.drop_vars(coord)
    gridU = gridU.assign_coords({"k": gridU["k"] + 1,
                                 "j": gridU["j"] + 1,
                                 "i": gridU["i"] + 1.5,
                                 "gphiu": gridU.gphiu,
                                 "glamu": gridU.glamu
                                 })

    gridV = d_grids["gridV"].rename_dims({**d_rename , **{"depthv": "k"}})
    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridV:
            gridV = gridV.drop_vars(coord)
    gridV = gridV.assign_coords({"k": gridV["k"] + 1,
                                 "j": gridV["j"] + 1.5,
                                 "i": gridV["i"] + 1,
                                 "gphiv": gridV.gphiv,
                                 "glamv": gridV.glamv
                                 })

    gridW = d_grids["gridW"].rename_dims({**d_rename , **{"depthw": "k"}})
    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridW:
            gridW = gridW.drop_vars(coord)
    gridW = gridW.assign_coords({"k": gridW["k"] + 0.5,
                                 "j": gridW["j"] + 1,
                                 "i": gridW["i"] + 1,
                                 "gphit": gridT.gphit,
                                 "glamt": gridT.glamt
                                 })

    gridF = d_grids["gridF"].rename_dims(d_rename)
    gridF = gridF.assign_coords({"j": gridF["j"] + 1.5,
                                 "i": gridF["i"] + 1.5,
                                 "gphif": gridF.gphif,
                                 "glamf": gridF.glamf
                                 })

    d_out = {
        "/": gridT.drop_dims(["j", "i", "k"]),
        "/gridT": gridT,
        "/gridU": gridU,
        "/gridV": gridV,
        "/gridW": gridW,
        "/gridF": gridF
            } 

    return d_out


def _process_child(
    d_child: dict[dict[str, str]],
    d_nests: dict[str, str],
    label: int
) -> dict[str, xr.Dataset]:
    """
    Create a dictionary of grid datasets defining a NEMO model child domain.

    Parameters
    ----------
    d_child : dict[dict[str, str]]
        A dictionary containing paths to NEMO child grid output files, structured as:
        {
            'domain': 'path/to/child_domain.nc',
            'gridT': 'path/to/child_gridT.nc',
            'gridU': 'path/to/child_gridU.nc',
            'gridV': 'path/to/child_gridV.nc',
            'gridW': 'path/to/child_gridW.nc',
        }
    
    d_nests : dict[str, int]
        A dictionary describing the properties of the child domain, structured as:
        {
            'rx': rx,
            'ry': ry,
            'imin': imin,
            'imax': imax,
            'jmin': jmin,
            'jmax': jmax,
        }

    label : int
        Label for the child grid, used to differentiate between multiple child grids.
    
    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO child grid output datasets, structured as:
        {
            f"/gridT/{label}_gridT": xr.Dataset,
            f"/gridU/{label}_gridU": xr.Dataset,
            f"/gridV/{label}_gridV": xr.Dataset,
            f"/gridW/{label}_gridW": xr.Dataset,
            f"/gridF/{label}_gridF": xr.Dataset
        }

    """
    # -- Open NEMO domain and grid datasets -- #
    d_grids = _process_grid_datasets(d_child)

    # -- Append domain variables to each grid dataset -- #
    d_grids = _add_domain_vars(d_grids)

    # -- Get child domain indices -- #
    ind_child = _get_child_indices(rx=d_nests['rx'],
                                   ry=d_nests['ry'],
                                   imin=d_nests['imin'],
                                   imax=d_nests['imax'],
                                   jmin=d_nests['jmin'],
                                   jmax=d_nests['jmax']
                                   )
    print(f"Child domain indices: {ind_child}")
    i_slice = slice(ind_child[0], ind_child[1] + 1)
    j_slice = slice(ind_child[2], ind_child[3] + 1)

    i_name = f"i{label}"
    j_name = f"j{label}"
    k_name = f"k{label}"
    d_rename = {"y": j_name, "x": i_name}

    # -- Construct T / U / V / W / F grids -- #
    gridT = d_grids["gridT"].rename_dims({**d_rename , **{"deptht": k_name}})
    d_vars = {'deptht': f"{label}_deptht", 'gphit': f"{label}_gphit", 'glamt': f"{label}_glamt"}
    if 'tmask' in gridT.data_vars:
        d_vars['tmask'] = f"{label}_tmask"
    gridT = gridT.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridT:
            gridT = gridT.drop_vars(coord)
    gridT = gridT.isel({i_name: i_slice, j_name: j_slice})
    gridT = gridT.assign_coords({k_name: gridT[k_name] + 1,
                                 j_name: gridT[j_name] + 1,
                                 i_name: gridT[i_name] + 1,
                                 f"{label}_gphit": gridT[f"{label}_gphit"],
                                 f"{label}_glamt": gridT[f"{label}_glamt"]
                                 })

    gridU = d_grids["gridU"].rename_dims({**d_rename , **{"depthu": k_name}})
    d_vars = {'depthu': f"{label}_depthu", 'gphiu': f"{label}_gphiu", 'glamu': f"{label}_glamu"}
    if 'umask' in gridU.data_vars:
        d_vars['umask'] = f"{label}_umask"
    gridU = gridU.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridU:
            gridU = gridU.drop_vars(coord)
    gridU = gridU.isel({i_name: i_slice, j_name: j_slice})
    gridU = gridU.assign_coords({k_name: gridU[k_name] + 1,
                                 j_name: gridU[j_name] + 1,
                                 i_name: gridU[i_name] + 1.5,
                                 f"{label}_gphiu": gridU[f"{label}_gphiu"],
                                 f"{label}_glamu": gridU[f"{label}_glamu"]
                                 })

    gridV = d_grids["gridV"].rename_dims({**d_rename , **{"depthv": k_name}})
    d_vars = {'depthv': f"{label}_depthv", 'gphiv': f"{label}_gphiv", 'glamv': f"{label}_glamv"}
    if 'vmask' in gridV.data_vars:
        d_vars['vmask'] = f"{label}_vmask"
    gridV = gridV.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridV:
            gridV = gridV.drop_vars(coord)
    gridV = gridV.isel({i_name: i_slice, j_name: j_slice})
    gridV = gridV.assign_coords({k_name: gridV[k_name] + 1,
                                 j_name: gridV[j_name] + 1.5,
                                 i_name: gridV[i_name] + 1,
                                 f"{label}_gphiv": gridV[f"{label}_gphiv"],
                                 f"{label}_glamv": gridV[f"{label}_glamv"]
                                 })

    gridW = d_grids["gridW"].rename_dims({**d_rename , **{"depthw": k_name}})
    d_vars = {'depthw': f"{label}_depthw", 'gphit': f"{label}_gphit", 'glamt': f"{label}_glamt"}
    if 'wmask' in gridW.data_vars:
        d_vars['wmask'] = f"{label}_wmask"
    gridW = gridW.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridW:
            gridW = gridW.drop_vars(coord)
    gridW = gridW.isel({i_name: i_slice, j_name: j_slice})
    gridW = gridW.assign_coords({k_name: gridW[k_name] + 0.5,
                                 j_name: gridW[j_name] + 1,
                                 i_name: gridW[i_name] + 1,
                                 f"{label}_gphit": gridT[f"{label}_gphit"],
                                 f"{label}_glamt": gridT[f"{label}_glamt"]
                                 })

    gridF = d_grids["gridF"].rename_dims(d_rename)
    d_vars = {'gphif': f"{label}_gphif", 'glamf': f"{label}_glamf"}
    if 'fmask' in gridF.data_vars:
        d_vars['fmask'] = f"{label}_fmask"
    gridF = gridF.rename(d_vars)
    gridF = gridF.isel({i_name: i_slice, j_name: j_slice})
    gridF = gridF.assign_coords({j_name: gridF[j_name] + 1.5,
                                 i_name: gridF[i_name] + 1.5,
                                 f"{label}_gphif": gridF[f"{label}_gphif"],
                                 f"{label}_glamf": gridF[f"{label}_glamf"]
                                 })

    d_out = {
        f"/gridT/{label}_gridT": gridT,
        f"/gridU/{label}_gridU": gridU,
        f"/gridV/{label}_gridV": gridV,
        f"/gridW/{label}_gridW": gridW,
        f"/gridF/{label}_gridF": gridF
            } 

    return d_out


def _process_grandchild(
    d_grandchild: dict[dict[str, str]],
    d_nests: dict[str, int],
    label: int,
    parent_label: int
) -> xr.DataTree:
    """
    Create a dictionary of grid datasets defining a NEMO model grandchild domain.

    Parameters
    ----------
    d_grandchild : dict[dict[str, str]]
        A dictionary containing paths to NEMO grandchild grid output files,
        structured as:
        {
            'parent': '1',
            'domain': 'path/to/grandchild_domain.nc',
            'gridT': 'path/to/grandchild_gridT.nc',
            'gridU': 'path/to/grandchild_gridU.nc',
            'gridV': 'path/to/grandchild_gridV.nc',
            'gridW': 'path/to/grandchild_gridW.nc',
        }

    d_nests : dict[str, int]
        A dictionary describing the properties of the grandchild domain, structured as:
        {
            'rx': rx,
            'ry': ry,
            'imin': imin,
            'imax': imax,
            'jmin': jmin,
            'jmax': jmax,
        }
    
    label : int
        Label for the grandchild grid, used to differentiate between multiple
        grandchild grids.

    parent_label : int
        Label for the parent grid, used to identify the child grid to which
        this grandchild grid belongs.

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary containing NEMO grandchild grid output datasets, structured as:
        {
            f"/gridT/{parent_label}_gridT/{label}_gridT": xr.Dataset,
            f"/gridU/{parent_label}_gridU/{label}_gridU": xr.Dataset,
            f"/gridV/{parent_label}_gridV/{label}_gridV": xr.Dataset,
            f"/gridW/{parent_label}_gridW/{label}_gridW": xr.Dataset,
            f"/gridF/{parent_label}_gridF/{label}_gridF": xr.Dataset
        }
    """
    # -- Open NEMO domain and grid datasets -- #
    d_grids = _process_grid_datasets(d_grandchild)

    # -- Append domain variables to each grid dataset -- #
    d_grids = _add_domain_vars(d_grids)

    # -- Get child domain indices -- #
    ind_child = _get_child_indices(rx=d_nests['rx'],
                                   ry=d_nests['ry'],
                                   imin=d_nests['imin'],
                                   imax=d_nests['imax'],
                                   jmin=d_nests['jmin'],
                                   jmax=d_nests['jmax']
                                   )
    i_name = f"i{label}"
    j_name = f"j{label}"
    k_name = f"k{label}"
    d_rename = {"y": j_name, "x": i_name}

    # -- Construct T / U / V / W / F grids -- #
    gridT = d_grids["gridT"].rename_dims({**d_rename , **{"deptht": k_name}})
    d_vars = {'deptht': f"{label}_deptht", 'gphit': f"{label}_gphit", 'glamt': f"{label}_glamt"}
    if 'tmask' in gridT.data_vars:
        d_vars['tmask'] = f"{label}_tmask"
    gridT = gridT.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridT:
            gridT = gridT.drop_vars(coord)
    gridT = gridT.isel({i_name: slice(*ind_child[:2]), j_name: slice(*ind_child[2:])})
    gridT = gridT.assign_coords({k_name: gridT[k_name] + 1,
                                 j_name: gridT[j_name] + 1,
                                 i_name: gridT[i_name] + 1,
                                 f"{label}_gphit": gridT[f"{label}_gphit"],
                                 f"{label}_glamt": gridT[f"{label}_glamt"]
                                 })

    gridU = d_grids["gridU"].rename_dims({**d_rename , **{"depthu": k_name}})
    d_vars = {'depthu': f"{label}_depthu", 'gphiu': f"{label}_gphiu", 'glamu': f"{label}_glamu"}
    if 'umask' in gridU.data_vars:
        d_vars['umask'] = f"{label}_umask"
    gridU = gridU.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridU:
            gridU = gridU.drop_vars(coord)
    gridU = gridU.isel({i_name: slice(*ind_child[:2]), j_name: slice(*ind_child[2:])})
    gridU = gridU.assign_coords({k_name: gridU[k_name] + 1,
                                 j_name: gridU[j_name] + 1,
                                 i_name: gridU[i_name] + 1.5,
                                 f"{label}_gphiu": gridU[f"{label}_gphiu"],
                                 f"{label}_glamu": gridU[f"{label}_glamu"]
                                 })

    gridV = d_grids["gridV"].rename_dims({**d_rename , **{"depthv": k_name}})
    d_vars = {'depthv': f"{label}_depthv", 'gphiv': f"{label}_gphiv", 'glamv': f"{label}_glamv"}
    if 'vmask' in gridV.data_vars:
        d_vars['vmask'] = f"{label}_vmask"
    gridV = gridV.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridV:
            gridV = gridV.drop_vars(coord)
    gridV = gridV.isel({i_name: slice(*ind_child[:2]), j_name: slice(*ind_child[2:])})
    gridV = gridV.assign_coords({k_name: gridV[k_name] + 1,
                                 j_name: gridV[j_name] + 1.5,
                                 i_name: gridV[i_name] + 1,
                                 f"{label}_gphiv": gridV[f"{label}_gphiv"],
                                 f"{label}_glamv": gridV[f"{label}_glamv"]
                                 })

    gridW = d_grids["gridW"].rename_dims({**d_rename , **{"depthw": k_name}})
    d_vars = {'depthw': f"{label}_depthw", 'gphit': f"{label}_gphit", 'glamt': f"{label}_glamt"}
    if 'wmask' in gridW.data_vars:
        d_vars['wmask'] = f"{label}_wmask"
    gridW = gridW.rename(d_vars)

    for coord in ('nav_lat', 'nav_lon'):
        if coord in gridW:
            gridW = gridW.drop_vars(coord)
    gridW = gridW.isel({i_name: slice(*ind_child[:2]), j_name: slice(*ind_child[2:])})
    gridW = gridW.assign_coords({k_name: gridW[k_name] + 0.5,
                                 j_name: gridW[j_name] + 1,
                                 i_name: gridW[i_name] + 1,
                                 f"{label}_gphit": gridT[f"{label}_gphit"],
                                 f"{label}_glamt": gridT[f"{label}_glamt"]
                                 })

    gridF = d_grids["gridF"].rename_dims(d_rename)
    d_vars = {'gphif': f"{label}_gphif", 'glamf': f"{label}_glamf"}
    if 'fmask' in gridF.data_vars:
        d_vars['fmask'] = f"{label}_fmask"
    gridF = gridF.rename(d_vars)
    gridF = gridF.isel({i_name: slice(*ind_child[:2]), j_name: slice(*ind_child[2:])})
    gridF = gridF.assign_coords({j_name: gridF[j_name] + 1.5,
                                 i_name: gridF[i_name] + 1.5,
                                 f"{label}_gphif": gridF[f"{label}_gphif"],
                                 f"{label}_glamf": gridF[f"{label}_glamf"]
                                 })

    d_out = {
        f"/gridT/{parent_label}_gridT/{label}_gridT": gridT,
        f"/gridU/{parent_label}_gridU/{label}_gridU": gridU,
        f"/gridV/{parent_label}_gridV/{label}_gridV": gridV,
        f"/gridW/{parent_label}_gridW/{label}_gridW": gridW,
        f"/gridF/{parent_label}_gridF/{label}_gridF": gridF
            } 

    return d_out

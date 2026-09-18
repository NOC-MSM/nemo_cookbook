"""
virtualize.py

Description:
This module includes virtualization functions used to construct virtual
NEMODataTree data structures used in the NEMO Cookbook library.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import glob
import importlib
import warnings

import xarray as xr


def get_invalid_vars(filepath: str) -> list[str]:
    """
    Return list of invalid variables in NetCDF file.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file.

    Returns
    -------
    list[str]
        List of invalid variables in NetCDF file.
    """
    # -- Local Imports -- #
    try:
        h5py = importlib.import_module("h5py")
    except ImportError as exc:
        raise ImportError(
            "virtualize_from_paths() requires the optional 'h5py' package.\n"
            "Install with: pip install 'nemo-cookbook[virtual]'"
        ) from exc

    # -- Identify invalid variables -- #
    drop_vars_list = []
    with h5py.File(filepath, "r") as f:
        for var in f.keys():
            if f[var].shape[0] == 0:
                drop_vars_list.append(f[var].name.replace("/", ""))

    if len(drop_vars_list) > 0:
        warnings.warn(f"Dropping invalid variables {drop_vars_list} found in file: '{filepath.split('/')[-1]}'", stacklevel=2)

    return drop_vars_list

def create_virtual_dataset(
    prefix: str,
    filepaths: list[str],
    drop_vars: list[str] | None = None,
    loadable_vars: list[str] | None = None,
    decode_times: bool=True,
    parallel: str="dask",
    combine: str="by_coords",
    combine_attrs: str="drop_conflicts"
    ) -> xr.Dataset:
    """
    Create a virtual xarray.Dataset from a list of NetCDF filepaths.

    Parameters
    ----------
    prefix : str
        Prefix shared by the paths to all NetCDF files.
    filepaths : list[str]
        List of paths to the NetCDF files.
    drop_vars : list[str] | None, optional
        List of variables to drop in virtual datasets. Default is None.
    loadable_vars : list[str] | None, optional
        List of variables to load as Dask/NumPy arrays. Default is None.
    decode_times : bool, optional
        Whether to decode times in the virtual dataset. Default is True.
    parallel : str, optional
        Parallelization strategy for opening the virtual dataset. Default is "dask".
    combine : str, optional
        Method to combine multiple datasets. Default is "by_coords".
    combine_attrs : str, optional
        Method to combine attributes when combining datasets. Default is "drop_conflicts".

    Returns
    -------
    xarray.Dataset
        Virtual xarray.Dataset constructed from NetCDF filepaths.
    """
    # -- Local Imports -- #
    try:
        virtualizarr = importlib.import_module("virtualizarr")
        obspec_utils = importlib.import_module("obspec_utils")
        obstore = importlib.import_module("obstore")
    except ImportError as exc:
        raise ImportError(
            "virtualize_from_paths() requires the optional 'virtualizarr,' 'obspec_utils,' and 'obstore' packages.\n"
            "Install with: pip install 'nemo-cookbook[virtual]'"
        ) from exc

    # -- Open virtual dataset from NetCDF filepaths -- #
    # Prepare registry for virtual dataset:
    file_urls = [f"file://{fp}" for fp in filepaths]
    store = obstore.store.LocalStore(prefix=prefix)
    registry = obspec_utils.registry.ObjectStoreRegistry(stores={file_url : store for file_url in file_urls})

    # Define parser for virtual dataset:
    parser = virtualizarr.parsers.HDFParser(drop_variables=drop_vars)

    # Open virtual single / multifile dataset from paths:
    if len(filepaths) == 1:
        vds = virtualizarr.open_virtual_dataset(
            url=filepaths[0],
            registry=registry,
            parser=parser,
            decode_times=decode_times
        )
    else:
        vds = virtualizarr.open_virtual_mfdataset(
            urls=filepaths,
            registry=registry,
            parser=parser,
            loadable_variables=loadable_vars,
            decode_times=decode_times,
            parallel=parallel,
            combine=combine,
            combine_attrs=combine_attrs
        )

    return vds

def create_virtual_dataset_dict(
    prefix: str,
    paths: dict[str, str],
    nests: dict[str, str] | None = None,
    drop_vars: list[str] | None = None,
    loadable_vars: list[str] | None = None,
    ) -> dict[str, dict]:
    """
    Create a paths dictionary containing virtual xarray Datasets
    representing a collection of NEMO model grids.

    Parameters
    ----------
    prefix : str
        Prefix shared by the paths to all NEMO grid files.
    paths : dict[str, str]
        Dictionary containing paths to NEMO grid files, structured as:
        {
            'parent': {'domain': 'path/to/domain.nc',
                        'gridT': 'path/to/gridT.nc',
                        , ... ,
                        'icemod': 'path/to/icemod.nc',
                        },
            'child': {'1': {'domain': 'path/to/child_domain.nc',
                            'gridT': 'path/to/child_gridT.nc',
                            , ... ,
                            'icemod': 'path/to/child_icemod.nc',
                            },
                        },
            'grandchild': {'2': {'domain': 'path/to/grandchild_domain.nc',
                                    'gridT': 'path/to/grandchild_gridT.nc',
                                    , ...,
                                    'icemod': 'path/to/grandchild_icemod.nc',
                                    },
                            }
        }

    nests : dict[str, str], optional
        Dictionary describing the properties of nested domains, structured as:
        {
            "1": {
                "parent": "/",
                "rx": rx,
                "ry": ry,
                "imin": imin,
                "imax": imax,
                "jmin": jmin,
                "jmax": jmax,
                "iperio": iperio,
                },
        }
        where `rx` and `ry` are the horizontal refinement factors, and `imin`, `imax`, `jmin`, `jmax`
        define the indices of the child (grandchild) domain within the parent (child) domain. Zonally
        periodic nested domains should be specified with `iperio=True`.

    drop_vars : list[str] | None, optional
        List of variables to drop in virtual datasets. Default is None.

    loadable_vars : list[str] | None, optional
        List of variables to load as Dask/NumPy arrays. Default is None.

    Returns
    -------
    dict[str, dict[str, xr.Dataset]]
        Paths dictionary of virtual xarray Datasets representing a collection of NEMO model grids.
    """
    # -- Open Virtual Datasets -- #
    default_drop_vars = ["x", "y", "nav_lev"]
    drop_vars = list(dict.fromkeys((drop_vars or []) + default_drop_vars))

    default_loadable_vars = ['time_counter', 'nav_lat', 'nav_lon', 'deptht', 'depthu', 'depthv', 'depthw', 'ncatice']
    loadable_vars = list(dict.fromkeys((loadable_vars or []) + default_loadable_vars))

    if "parent" in paths.keys() and isinstance(paths["parent"], dict):
        d_vds = {key: {} for key in paths.keys()}
        for key in paths.keys():
            if key not in ("parent", "child", "grandchild"):
                raise KeyError(f"Unexpected key '{key}' in `paths` dictionary.")
            if key == "parent":
                for key_parent, path_parent in paths["parent"].items():
                    if prefix not in path_parent:
                        raise ValueError(f"prefix '{prefix}' must be found in path '{path_parent}'")
                    filepaths = sorted(glob.glob(path_parent))
                    invalid_vars = get_invalid_vars(filepaths[0])
                    # Open virtual dataset for each NEMO grid type in the parent domain:
                    d_vds["parent"][key_parent] = create_virtual_dataset(prefix=prefix,
                                                                                    filepaths=filepaths,
                                                                                    drop_vars=drop_vars + invalid_vars,
                                                                                    loadable_vars=loadable_vars
                                                                                    ).squeeze(drop=True)

                    # Optionally merge scalar T-point and sea-ice variables into gridT virtual dataset:
                    if ("gridT" in d_vds["parent"]) and ("icemod" in d_vds["parent"]):
                        d_vds["parent"]["gridT"] = xr.combine_by_coords([d_vds["parent"]["gridT"], d_vds["parent"]["icemod"]],
                                                                                    compat="override", # use gridT
                                                                                    combine_attrs="override" # use gridT
                                                                                    )

                    # Drop 1D depth variables from the parent domain -> use Dataset coords:
                    if 'gdept_1d' in d_vds['parent']['domain'].data_vars:
                        d_vds['parent']['domain'] = d_vds['parent']['domain'].drop_vars('gdept_1d')
                    if 'gdepw_1d' in d_vds['parent']['domain'].data_vars:
                        d_vds['parent']['domain'] = d_vds['parent']['domain'].drop_vars('gdepw_1d')

            elif (key == "child") or (key == "grandchild"):
                if nests is None:
                    raise ValueError(
                        "`nests` dictionary must be provided when defining NEMO child domains."
                    )
                for n in paths[key].keys():
                    d_vds[key][n] = {}
                    for key_n, path_n in paths[key][n].items():
                        if prefix not in path_n:
                            raise ValueError(f"prefix '{prefix}' must be found in path '{path_n}'")
                        filepaths = sorted(glob.glob(path_n))
                        invalid_vars = get_invalid_vars(filepaths[0])
                        # Open virtual dataset for each NEMO grid in the child / grandchild domain:
                        d_vds[key][n][key_n] = create_virtual_dataset(prefix=prefix,
                                                                                filepaths=filepaths,
                                                                                drop_vars=drop_vars + invalid_vars,
                                                                                loadable_vars=loadable_vars
                                                                                ).squeeze(drop=True)

                        # Optionally merge scalar T-point and sea-ice variables into gridT virtual dataset:
                        if ("gridT" in d_vds[key][n]) and ("icemod" in d_vds[key][n]):
                            d_vds[key][n]["gridT"] = xr.combine_by_coords([d_vds[key][n]["gridT"], d_vds[key][n]["icemod"]],
                                                                                    compat="override", # use gridT
                                                                                    combine_attrs="override" # use gridT
                                                                                    )

                        # Drop 1D depth variables from the child / grandchild domain -> use Dataset coords:
                        if 'gdept_1d' in d_vds[key][n]['domain'].data_vars:
                            d_vds[key][n]['domain'] = d_vds[key][n]['domain'].drop_vars('gdept_1d')
                        if 'gdepw_1d' in d_vds[key][n]['domain'].data_vars:
                            d_vds[key][n]['domain'] = d_vds[key][n]['domain'].drop_vars('gdepw_1d')
    else:
        raise ValueError(
            "Invalid `paths` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
        )

    return d_vds

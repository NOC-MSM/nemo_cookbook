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
from concurrent.futures import Executor
from typing import Literal, cast

import xarray as xr

from nemo_cookbook.utils import DatasetsDict, NestsDict, PathsDict


def _resolve_filepaths(
    pattern: str,
    prefix: str,
    label: str
    ) -> list[str]:
    """
    Resolve filepaths matching a pattern and ensure the prefix is present.

    Parameters
    ----------
    pattern : str
        Glob pattern to match files.
    prefix : str
        Prefix that must be present in the pattern.
    label : str
        Label for the type of files being resolved (used in error messages).

    Returns
    -------
    list[str]
        Sorted list of filepaths matching the pattern.

    Raises
    ------
    ValueError
        If the prefix is not found in the pattern or no files match the pattern.
    """
    if prefix not in pattern:
        raise ValueError(f"prefix '{prefix}' must be found in path '{pattern}'")

    filepaths = sorted(glob.glob(pattern))
    if not filepaths:
        raise ValueError(
            f"No files matched pattern '{pattern}' for {label}."
        )

    return filepaths


def _get_invalid_vars(filepath: str) -> list[str]:
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
        h5py = importlib.import_module(name="h5py")
    except ImportError as exc:
        raise ImportError(
            "virtualize_from_paths() requires optional virtualization dependencies.\n"
            "Install with: pip install 'nemo-cookbook[virtual]'"
        ) from exc

    # -- Identify invalid variables -- #
    drop_vars_list = []
    with h5py.File(name=filepath, mode="r") as f:
        for var in f.keys():
            shape = getattr(f[var], "shape", None)
            if (shape is None) or (len(shape) == 0):
                continue
            if shape[0] == 0:
                drop_vars_list.append(f[var].name.replace("/", ""))

    if drop_vars_list:
        warnings.warn(
            message=f"Dropping invalid variables {drop_vars_list} found in file: '{filepath.split(sep='/')[-1]}'",
            stacklevel=2
            )

    return drop_vars_list

def _create_virtual_dataset(
    prefix: str,
    filepaths: list[str],
    drop_vars: list[str] | None = None,
    loadable_vars: list[str] | None = None,
    decode_times: bool=True,
    parallel: Literal["dask", "lithops", False] | type[Executor] ="dask",
    combine: Literal["by_coords", "nested"] = "by_coords",
    combine_attrs: Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"]="drop_conflicts"
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
    parallel : Literal["dask", "lithops", False] | type[Executor], optional
        Parallelization strategy for opening the virtual dataset. Default is "dask".
    combine : Literal["by_coords", "nested"], optional
        Method to combine multiple datasets. Default is "by_coords".
    combine_attrs : Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"], optional
        Method to combine attributes when combining datasets. Default is "drop_conflicts".

    Returns
    -------
    xarray.Dataset
        Virtual xarray.Dataset constructed from NetCDF filepaths.
    """
    # -- Local Imports -- #
    try:
        virtualizarr = importlib.import_module(name="virtualizarr")
        virtualizarr_parsers = importlib.import_module(name="virtualizarr.parsers")
        obspec_registry = importlib.import_module(name="obspec_utils.registry")
        obstore = importlib.import_module(name="obstore")
    except ImportError as exc:
        raise ImportError(
            "virtualize_from_paths() requires optional virtualization dependencies.\n"
            "Install with: pip install 'nemo-cookbook[virtual]'"
        ) from exc

    # -- Open virtual dataset from NetCDF filepaths -- #
    # Prepare registry for virtual dataset:
    file_urls = [f"file://{fp}" for fp in filepaths]
    store = obstore.store.LocalStore(prefix=prefix)
    registry = obspec_registry.ObjectStoreRegistry(stores={file_url : store for file_url in file_urls})

    # Define parser for virtual dataset:
    parser = virtualizarr_parsers.HDFParser(drop_variables=drop_vars)

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
    paths: PathsDict,
    nests: NestsDict | None = None,
    drop_vars: list[str] | None = None,
    loadable_vars: list[str] | None = None,
    ) -> DatasetsDict:
    """
    Create a paths dictionary containing virtual xarray Datasets
    representing a collection of NEMO model grids.

    Parameters
    ----------
    prefix : str
        Prefix shared by the paths to all NEMO grid files.
    paths : PathsDict,
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

    nests : NestsDict | None, optional
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
    DatasetsDict
        Dictionary of virtual xarray Datasets representing a collection of NEMO model grids.
    """
    # -- Open Virtual Datasets -- #
    default_drop_vars = ["x", "y", "nav_lev"]
    drop_vars = list(dict.fromkeys((drop_vars or []) + default_drop_vars))

    default_loadable_vars = [
        "time_counter",
        "nav_lat",
        "nav_lon",
        "deptht",
        "depthu",
        "depthv",
        "depthw",
        "ncatice",
    ]
    loadable_vars = list(dict.fromkeys((loadable_vars or []) + default_loadable_vars))

    # -- Parent Domain -- #
    parent_paths = paths.get("parent")
    if parent_paths is None:
        raise ValueError(
            "Invalid `paths` structure. Expected a nested dictionary defining NEMO "
            "'parent', 'child' and 'grandchild' domains."
        )

    d_vds: DatasetsDict = {"parent": {}}

    for key_parent, path_parent in parent_paths.items():
        filepaths = _resolve_filepaths(
            pattern=path_parent,
            prefix=prefix,
            label=f"parent/{key_parent}",
        )
        invalid_vars = _get_invalid_vars(filepath=filepaths[0])
        d_vds["parent"][key_parent] = _create_virtual_dataset(
            prefix=prefix,
            filepaths=filepaths,
            drop_vars=drop_vars + invalid_vars,
            loadable_vars=loadable_vars,
        ).squeeze(drop=True)

    # Optionally merge scalar T-point and sea-ice variables into gridT virtual dataset:
    if ("gridT" in d_vds["parent"]) and ("icemod" in d_vds["parent"]):
        d_vds["parent"]["gridT"] = cast(
            typ=xr.Dataset,
            val=xr.combine_by_coords(data_objects=[d_vds["parent"]["gridT"], d_vds["parent"]["icemod"]],
                                     compat="override", # use gridT
                                     combine_attrs="override" # use gridT
                                     ))

    # Drop 1D depth variables from the parent domain -> use Dataset coords:
    if 'gdept_1d' in d_vds['parent']['domain'].data_vars:
        d_vds['parent']['domain'] = d_vds['parent']['domain'].drop_vars(names='gdept_1d')
    if 'gdepw_1d' in d_vds['parent']['domain'].data_vars:
        d_vds['parent']['domain'] = d_vds['parent']['domain'].drop_vars(names='gdepw_1d')

    # -- Child Domain(s) -- #
    child_paths = paths.get("child")
    if child_paths is not None:
        if nests is None:
            raise ValueError(
                "`nests` dictionary must be provided when defining NEMO child domains."
            )
        d_vds["child"] = {}
        for n, domain_paths in child_paths.items():
            d_vds["child"][n] = {}
            for key_n, path_n in domain_paths.items():
                filepaths = _resolve_filepaths(
                    pattern=path_n,
                    prefix=prefix,
                    label=f"child/{n}/{key_n}",
                )
                invalid_vars = _get_invalid_vars(filepath=filepaths[0])
                d_vds["child"][n][key_n] = _create_virtual_dataset(
                    prefix=prefix,
                    filepaths=filepaths,
                    drop_vars=drop_vars + invalid_vars,
                    loadable_vars=loadable_vars,
                ).squeeze(drop=True)

            # Optionally merge scalar T-point and sea-ice variables into gridT virtual dataset:
            if ("gridT" in d_vds["child"][n]) and ("icemod" in d_vds["child"][n]):
                d_vds["child"][n]["gridT"] = cast(
                    typ=xr.Dataset, 
                    val=xr.combine_by_coords(data_objects=[d_vds["child"][n]["gridT"], d_vds["child"][n]["icemod"]],
                                             compat="override", # use gridT
                                             combine_attrs="override" # use gridT
                                            ))

            # Drop 1D depth variables from the child / grandchild domain -> use Dataset coords:
            if 'gdept_1d' in d_vds["child"][n]['domain'].data_vars:
                d_vds["child"][n]['domain'] = d_vds["child"][n]['domain'].drop_vars(names='gdept_1d')
            if 'gdepw_1d' in d_vds["child"][n]['domain'].data_vars:
                d_vds["child"][n]['domain'] = d_vds["child"][n]['domain'].drop_vars(names='gdepw_1d')

    # -- Grandchild Domain(s) -- #
    grandchild_paths = paths.get("grandchild")
    if grandchild_paths is not None:
        if child_paths is None:
            raise ValueError(
                "`child_paths` must be defined when defining NEMO grandchild domains."
            )
        d_vds["grandchild"] = {}
        for n, domain_paths in grandchild_paths.items():
            d_vds["grandchild"][n] = {}
            for key_n, path_n in domain_paths.items():
                filepaths = _resolve_filepaths(
                    pattern=path_n,
                    prefix=prefix,
                    label=f"grandchild/{n}/{key_n}",
                )
                invalid_vars = _get_invalid_vars(filepath=filepaths[0])
                d_vds["grandchild"][n][key_n] = _create_virtual_dataset(
                    prefix=prefix,
                    filepaths=filepaths,
                    drop_vars=drop_vars + invalid_vars,
                    loadable_vars=loadable_vars,
                ).squeeze(drop=True)

            # Optionally merge scalar T-point and sea-ice variables into gridT virtual dataset:
            if ("gridT" in d_vds["grandchild"][n]) and ("icemod" in d_vds["grandchild"][n]):
                d_vds["grandchild"][n]["gridT"] = cast(
                    typ=xr.Dataset,
                    val=xr.combine_by_coords(
                        data_objects=[d_vds["grandchild"][n]["gridT"], d_vds["grandchild"][n]["icemod"]],
                        compat="override", # use gridT
                        combine_attrs="override" # use gridT
                    )
                )

            # Drop 1D depth variables from the child / grandchild domain -> use Dataset coords:
            if 'gdept_1d' in d_vds["grandchild"][n]['domain'].data_vars:
                d_vds["grandchild"][n]['domain'] = d_vds["grandchild"][n]['domain'].drop_vars(names='gdept_1d')
            if 'gdepw_1d' in d_vds["grandchild"][n]['domain'].data_vars:
                d_vds["grandchild"][n]['domain'] = d_vds["grandchild"][n]['domain'].drop_vars(names='gdepw_1d')

    return d_vds

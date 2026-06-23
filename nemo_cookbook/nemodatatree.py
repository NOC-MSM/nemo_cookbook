"""
nemodatatree.py

Description:
This module defines the NEMODataTree class, a hierarchical data structure
for analysing NEMO ocean general circulation outputs defining one or more
model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

from typing import Self

import icechunk
import numpy as np
import xarray as xr
from xarray.indexes import NDPointIndex
from xoak import SklearnGeoBallTreeAdapter

from nemo_cookbook.extract import (
    create_boundary_dataset,
    create_section_polygon,
    get_section_indexes,
    update_boundary_dataset,
)
from nemo_cookbook.masks import create_polygon_mask, get_mask_boundary
from nemo_cookbook.nemodataarray import NEMODataArray
from nemo_cookbook.processing import create_datatree_dict
from nemo_cookbook.stats import compute_binned_statistic
from nemo_cookbook.utils import deprecated
from nemo_cookbook.validation import validate_nemo_grid_node


class NEMODataTree(xr.DataTree):
    """
    Hierarchical data structure containing collections of NEMO ocean model outputs.

    This class extends `xarray.DataTree` to provide methods for processing
    and analysing NEMO output xarray objects defining one or more model domains.

    It supports NEMO discrete scalar and vector operators such as computing gradients,
    divergence, curl, weighted averages, integrals, cumulative integrals, and
    transforming variables between grids.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a single node of a NEMODataTree.

        The node may optionally contain data in the form of data
        and coordinate variables, stored in the same way as data
        is stored in an `xarray.Dataset`.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the parent class.
        **kwargs : dict
            Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)

    @classmethod
    def from_paths(
        cls,
        paths: dict[str, str],
        nests: dict[str, str] | None = None,
        name : str = "NEMO model",
        iperio: bool = False,
        nftype: str | None = None,
        read_mask: bool = False,
        maskcs: bool = False,
        key_linssh: bool = False,
        vco_ref: bool = False,
        nbghost_child: int = 4,
        **open_kwargs: dict[str, any],
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of paths to NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
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

        name: str, optional
            Name of the NEMODataTree. Default is "NEMO model".

        iperio: bool = False
            Zonal periodicity of the parent domain. Default is False.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

        read_mask: bool = False
            If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed from top_level and
            bottom_level domain variables. Default is False.

        maskcs: bool = False
            If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.

        key_linssh: bool = False
            Linear free-surface approximation. If True, vertical coordinates are time-independent and given by (e3t_0, e3u_0, e3v_0, e3w_0) in domain_cfg.
            If False, vertical coordinates are time-dependent and must be specified in NEMO model grid datasets. Default is False.

        vco_ref: bool = False
            If True, add reference vertical scale factors and compute reference water column heights from domain files. Default is False.

        nbghost_child : int = 4
            Number of ghost cells to remove from the western/southern boundaries of the (grand)child domains. Default is 4.

        **open_kwargs : dict, optional
            Additional keyword arguments to pass to `xarray.open_dataset` or `xr.open_mfdataset` when opening NEMO model output files.
        Returns
        -------
        NEMODataTree
            A hierarchical DataTree storing NEMO model outputs.

        Examples
        --------
        Create a zonally periodic `NEMODataTree` with north folding on T-points from a dictionary of paths to local netCDF files:

        >>> from nemo_cookbook import NEMODataTree
        >>> paths = {"parent": {
        ...          "domain": "/path/to/domain_cfg.nc",
        ...          "gridT": "path/to/*_gridT.nc",
        ...          "gridU": "path/to/*_gridV.nc",
        ...          "gridV": "path/to/*_gridV.nc",
        ...          "gridW": "path/to/*_gridW.nc",
        ...          "icemod": "path/to/*_icemod.nc",
        ...          }}
        >>> nemo = NEMODataTree.from_paths(paths, name="My NEMO model", iperio=True, nftype="T")

        Create a regional `NEMODataTree` using a linear free-surface approximation from a dictionary of paths to remote netCDF files:

        >>> nemo = NEMODataTree.from_paths(paths, name="My NEMO model", iperio=False, nftype=None, key_linssh=True)

        See Also
        --------
        from_datasets
        """
        # -- Validate Inputs -- #
        if not isinstance(paths, dict):
            raise TypeError("`paths` must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("`nests` must be a dictionary or None.")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity (`iperio`) of parent domain must be a boolean.")
        if nftype is not None and nftype not in ("T", "F"):
            raise ValueError(
                "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
            )
        if not isinstance(read_mask, bool):
            raise TypeError("reading land-sea masks from domain_cfg (`read_mask`) must be a boolean.")
        if not isinstance(maskcs, bool):
            raise TypeError("masking of closed seas (`maskcs`) must be a boolean.")
        if not isinstance(key_linssh, bool):
            raise TypeError("linear free-surface approximation (`key_linssh`) must be a boolean.")
        if not isinstance(vco_ref, bool):
            raise TypeError("reference vertical coordinates (`vco_ref`) must be a boolean.")
        if not isinstance(nbghost_child, int):
            raise TypeError(
                "number of ghost cells along the western/southern boundaries (`nbghost_child`) must be an integer."
            )
        if not isinstance(open_kwargs, dict):
            raise TypeError("`open_kwargs` must be a dictionary.")

        # -- Define parent, child, grandchild filepath collections -- #
        d_child, d_grandchild = None, None
        if "parent" in paths.keys() and isinstance(paths["parent"], dict):
            for key in paths.keys():
                if key not in ("parent", "child", "grandchild"):
                    raise KeyError(f"Unexpected key '{key}' in `paths` dictionary.")
                if key == "parent":
                    d_parent = paths["parent"]
                elif key == "child":
                    if nests is None:
                        raise ValueError(
                            "`nests` dictionary must be provided when defining NEMO child domains."
                        )
                    else:
                        d_child = paths["child"]
                elif key == "grandchild":
                    d_grandchild = paths["grandchild"]
        else:
            raise ValueError(
                "Invalid `paths` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
            )

        # -- Construct DataTree from parent / child / grandchild domains -- #
        d_tree = create_datatree_dict(
            d_parent=d_parent,
            d_child=d_child,
            d_grandchild=d_grandchild,
            nests=nests,
            iperio=iperio,
            nftype=nftype,
            read_mask=read_mask,
            maskcs=maskcs,
            nbghost_child=nbghost_child,
            key_linssh=key_linssh,
            vco_ref=vco_ref,
            open_kwargs=dict(**open_kwargs),
        )

        nemo = super().from_dict(d_tree)
        nemo.name = name

        return nemo

    @classmethod
    def from_datasets(
        cls,
        datasets: dict[str, dict[str, xr.Dataset]],
        nests: dict[str, dict[str, str]] | None = None,
        name : str = "NEMO model",
        iperio: bool = False,
        nftype: str | None = None,
        read_mask: bool = False,
        key_linssh: bool = False,
        vco_ref: bool = False,
        maskcs: bool = False,
        nbghost_child: int = 4,
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of `xarray.Dataset` objects created from NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        datasets : dict[str, dict[str, xr.Dataset]]
            Dictionary containing `xarray.Datasets` created from NEMO grid files, structured as:
            {
                'parent': {'domain': ds_domain, 'gridT': ds_gridT, ... , 'icemod': ds_icemod.nc},
                'child': {'1': {'domain': ds_domain_1, 'gridT': d_gridT_1, ...}},
                'grandchild': {'2': {'domain': ds_domain_2, 'gridT': ds_gridT_2, ...}}
            }

        nests : dict[str, dict[str, str]], optional
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
                    },
            }
            where `rx` and `ry` are the horizontal refinement factors, and `imin`, `imax`, `jmin`, `jmax`
            define the indices of the child (grandchild) domain within the parent (child) domain.

        name: str, optional
            Name of the NEMODataTree. Default is "NEMO model".

        iperio: bool = False
            Zonal periodicity of the parent domain.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

        read_mask: bool = False
            If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed from top_level and bottom_level
            domain variables. Default is False.

        maskcs: bool = False
            If True, all closed seas are masked using mask_opensea variables from domain files. Default is False.

        key_linssh: bool = False
            Linear free-surface approximation. If True, vertical coordinates are time-independent and given by (e3t_0, e3u_0, e3v_0, e3w_0) in domain_cfg.
            If False, vertical coordinates are time-dependent and must be specified in NEMO model grid datasets. Default is False.

        vco_ref: bool = False
            If True, add reference vertical scale factors and compute reference water column heights from domain files. Default is False.

        nbghost_child : int = 4
            Number of ghost cells to remove from the western/southern boundaries of the (grand)child domains. Default is 4.

        Returns
        -------
        NEMODataTree
            Hierarchical DataTree of NEMO model outputs.

        Examples
        --------
        Create a zonally periodic `NEMODataTree` with north folding on T-points from a dictionary of xarray.Dataset objects:

        >>> import xarray as xr
        >>> from nemo_cookbook import NEMODataTree
        >>> ds_domain = xr.open_zarr("https://some_remote_data/domain_cfg.zarr")
        >>> ds_gridT = xr.open_zarr("https://some_remote_data/my_model_gridT.zarr")
        >>> datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT}}
        >>> nemo = NEMODataTree.from_datasets(datasets=datasets, name="My NEMO Model", iperio=True, nftype="T")

        Create a regional `NEMODataTree` using a linear free-surface approximation from a dictionary of xarray.Dataset objects:

        >>> nemo = NEMODataTree.from_datasets(datasets=datasets, name="My NEMO Model", iperio=False, nftype=None, key_linssh=True)

        See Also
        --------
        from_paths
        """
        # -- Validate Inputs -- #
        if not isinstance(datasets, dict):
            raise TypeError("`datasets` must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("`nests` must be a dictionary or None.")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity (`iperio`) of parent domain must be a boolean.")
        if nftype is not None and nftype not in ("T", "F"):
            raise ValueError(
                "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
            )
        if not isinstance(read_mask, bool):
            raise TypeError("reading land-sea masks from domain_cfg (`read_mask`) must be a boolean.")
        if not isinstance(maskcs, bool):
            raise TypeError("masking of closed seas (`maskcs`) must be a boolean.")
        if not isinstance(key_linssh, bool):
            raise TypeError("linear free-surface approximation (`key_linssh`) must be a boolean.")
        if not isinstance(vco_ref, bool):
            raise TypeError("reference vertical coordinates (`vco_ref`) must be a boolean.")
        if not isinstance(nbghost_child, int):
            raise TypeError(
                "number of ghost cells along the western/southern boundaries (`nbghost_child`) must be an integer."
            )

        # -- Define parent, child, grandchild dataset collections -- #
        d_child, d_grandchild = None, None
        if "parent" in datasets.keys() and isinstance(datasets["parent"], dict):
            for key in datasets.keys():
                if key not in ("parent", "child", "grandchild"):
                    raise KeyError(f"Unexpected key '{key}' in `datasets` dictionary.")
                if key == "parent":
                    d_parent = datasets["parent"]
                elif key == "child":
                    if nests is None:
                        raise ValueError(
                            "`nests` dictionary must be provided when defining NEMO child domains."
                        )
                    else:
                        d_child = datasets["child"]
                elif key == "grandchild":
                    d_grandchild = datasets["grandchild"]
        else:
            raise ValueError(
                "Invalid `datasets` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
            )

        # -- Construct DataTree from parent / child / grandchild domains -- #
        d_tree = create_datatree_dict(
            d_parent=d_parent,
            d_child=d_child,
            d_grandchild=d_grandchild,
            nests=nests,
            iperio=iperio,
            nftype=nftype,
            read_mask=read_mask,
            maskcs=maskcs,
            key_linssh=key_linssh,
            vco_ref=vco_ref,
            nbghost_child=nbghost_child,
        )

        nemo = super().from_dict(d_tree)
        nemo.name = name

        return nemo

    @classmethod
    def from_icechunk(
        cls,
        repo: icechunk.repository.Repository,
        name: str = "NEMO model",
        iperio: bool = False,
        nftype: str | None = None,
        open_kwargs: dict[str, any] | None = None,
        **session_kwargs: dict[str, any],
    ) -> Self:
        """
        Create a NEMODataTree from an Icechunk repository storing NEMO model outputs
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        repo : icechunk.repository.Repository
            Icechunk repository containing NEMO model outputs organised into a
            hierarchy of domains.

        name : str, optional
            Name of the NEMODataTree. Default is "NEMO model".
        
        iperio: bool = False
            Zonal periodicity of the parent domain. Default is False.
        
        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point

        open_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to `xarray.open_datatree`. Default is None.

        **session_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to Icechunk `repo.readonly_session`.

        Returns
        -------
        NEMODataTree
            Hierarchical DataTree of NEMO model outputs.

        Examples
        --------
        Create a zonally periodic `NEMODataTree` with north folding on F-points from the main branch of an Icechunk repository:

        >>> from nemo_cookbook import NEMODataTree
        >>> nemo = NEMODataTree.from_icechunk(repo=repo, branch="main", iperio=True, nftype="F")

        See Also
        --------
        from_zarr
        """
        # -- Validate Inputs -- #
        if not hasattr(repo, "readonly_session"):
            raise TypeError("`repo` must implement readonly_session().")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity (`iperio`) of parent domain must be a boolean.")
        if nftype is not None and nftype not in ("T", "F"):
            raise ValueError(
                "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
            )

        # -- Create NEMODataTree from Icechunk repository -- #:
        session = repo.readonly_session(**session_kwargs)
        datatree = xr.open_datatree(session.store, engine="zarr", **(open_kwargs or {}))
        nemo = super().from_dict(datatree.to_dict())

        # -- Update NEMODataTree properties -- #
        nemo["/"].attrs.update({"nftype": nftype, "iperio": iperio})
        nemo.name = name

        # -- Validate NEMO grid node Datasets -- #
        for key in [grid for grid in nemo.groups if grid.startswith("grid")]:
            validate_nemo_grid_node(key=key, value=nemo[key])

        return nemo

    @classmethod
    def from_zarr(
        cls,
        store: str,
        name: str = "NEMO model",
        iperio: bool = False,
        nftype: str | None = None,
        **open_kwargs: dict[str, any],
    ) -> Self:
        """
        Create a NEMODataTree from Zarr store groups storing NEMO model outputs
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        store : str
            Path to the Zarr store containing NEMO model outputs in hierarchical groups.

        name : str, optional
            Name of the NEMODataTree. Default is "NEMO model".
        
        iperio: bool = False
            Zonal periodicity of the parent domain. Default is False.
        
        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point

        **open_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to `xarray.open_datatree`.

        Returns
        -------
        NEMODataTree
            Hierarchical DataTree of NEMO model outputs.

        Examples
        --------
        Create a zonally periodic `NEMODataTree` with north folding on T-points from a hierarchical Zarr store:

        >>> from nemo_cookbook import NEMODataTree
        >>> nemo = NEMODataTree.from_zarr(store="path/to/zarr/store", iperio=True, nftype="T")

        See Also
        --------
        from_icechunk
        """
        # -- Validate Inputs -- #
        if not isinstance(store, str):
            raise TypeError("`store` must be a string.")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity (`iperio`) of parent domain must be a boolean.")
        if nftype is not None and nftype not in ("T", "F"):
            raise ValueError(
                "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
            )

        # -- Create NEMODataTree from Zarr store -- #:
        datatree = xr.open_datatree(store, engine="zarr", **(open_kwargs or {}))
        nemo = super().from_dict(datatree.to_dict())

        # -- Update NEMODataTree properties -- #
        nemo["/"].attrs.update({"nftype": nftype, "iperio": iperio})
        nemo.name = name

        # -- Validate NEMO grid node Datasets -- #
        for key in [grid for grid in nemo.groups if grid.startswith("grid")]:
            validate_nemo_grid_node(key=key, value=nemo[key])

        return nemo
    
    def __setitem__(
            self,
            key: str,
            value: NEMODataArray | xr.DataArray | xr.Dataset,
            strict: bool = True
        ) -> None:
        """
        Set a child node or variable in this NEMODataTree.

        Overloads the __setitem__() method of xarray.DataTree to allow
        setting NEMODataArrays via variable paths (i.e, /grid/var).

        Optionally set strict=False to bypass validation of child grid nodes.

        Parameters
        ----------
        key : str
            Name of variable or child node, or unix-like path to variable
            within a child node.

        value : NEMODataArray | xarray.DataArray | xarray.Dataset
            Object to set at the specified key. If a NEMODataArray is provided,
            the underlying xarray.DataArray will be set at the specified key.

        strict : bool, optional
            Validate Datasets assigned to NEMO grid nodes to ensure they contain
            the required dimensions and coordinates. Default is True.

        Returns
        -------
        None
        """
        # -- Access DataArray from NEMODataArray -- #
        if isinstance(value, NEMODataArray):
            value = value.data

        if strict and isinstance(value, xr.Dataset):
            # -- Validate NEMO grid node Dataset -- #
            validate_nemo_grid_node(key=key, value=value)

        return super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Self | NEMODataArray:
        """
        Access child nodes, variables, or coordinates stored in this NEMODataTree.

        Returned object will be either a NEMODataTree or NEMODataArray object depending
        on whether the key points to a child node or variable.

        Overloads the __getitem__() method of xarray.DataTree to return NEMODataArrays
        accessed via variable paths (i.e, /grid/var).

        Parameters
        ----------
        key : str
            Name of variable / child within this node, or unix-like path to variable
            / child within another node.

        Returns
        -------
        NEMODataTree | NEMODataArray
        """
        # -- Access child node or variable -- #
        item = super().__getitem__(key=key)
        is_gridpath = key.startswith("/grid") or key.startswith("grid")

        # -- Return NEMODataArray -- #
        if isinstance(item, xr.DataArray) and is_gridpath:
            var_name = key.split("/")[-1]
            grid = key.replace(f"/{var_name}", "")
            item = NEMODataArray(da=item,
                                 tree=self,
                                 grid=grid
                                 )

        return item

    def _get_properties(
        self, dom: str | None = None, grid: str | None = None, infer_dom: bool = False
    ) -> str | tuple[str, ...]:
        """
        Get NEMO model domain and grid properties.

        The domain prefix & suffix (e.g., '1_', '1') are returned
        if only the NEMO model domain (`dom`) is specified.

        The grid suffix (e.g., 't', 'u', 'v', 'w') is returned if
        only the NEMO model grid (`grid`) is specified.

        The domain number, domain prefix & suffix, and grid suffix
        are returned if both the NEMO model grid (`grid`) and
        `infer_dom = True` are specified.

        Parameters
        ----------
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
        grid : str, optional
            Path to NEMO model grid (e.g., 'gridT').
        infer_dom : bool, optional
            Whether to infer the domain number & domain name from only the
            grid path. Default is False.

        Returns
        -------
        str | tuple[str, ...]
            NEMO model domain and grid properties.
        """
        if (grid is None) and (dom is not None):
            dom_prefix = "" if dom == "." else f"{dom}_"
            dom_suffix = "" if dom == "." else f"{dom}"
            return dom_prefix, dom_suffix
        else:
            grid_keys = list(dict(self.subtree_with_keys).keys())
            if grid not in grid_keys:
                raise KeyError(
                    f"grid '{grid}' not found in available NEMODataTree grids {grid_keys}."
                )
            grid_suffix = f"{grid.lower()[-1]}"

            if infer_dom:
                dom_inds = [char for char in grid if char.isdigit()]
                dom_prefix = f"{dom_inds[-1]}_" if len(dom_inds) != 0 else ""
                dom = dom_prefix[:-1] if dom_prefix != "" else "."
                dom_suffix = dom if dom != "." else ""
                return dom, dom_prefix, dom_suffix, grid_suffix
            else:
                return grid_suffix

    def _get_grid_paths(self, dom: str) -> dict[str, str]:
        """
        Get paths to NEMO model grids in a given domain.

        Parameters
        ----------
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).

        Returns
        -------
        dict[str, str]
            Dictionary of NEMO model grid paths.
        """
        # Collect paths to all NEMO model grids:
        grid_paths = list(dict(self.subtree_with_keys).keys())

        if dom == ".":
            grid_paths = [
                path for path in grid_paths if ("_" not in path) and ("grid" in path)
            ]
        else:
            grid_paths = [path for path in grid_paths if dom in path]

        d_paths = {path.split("/")[0]: path for path in grid_paths}

        return d_paths

    def _get_ijk_names(self, dom: str | None = None, grid: str | None = None) -> dict[str, str]:
        """
        Get (i, j, k) grid index names for given NEMO model domain.

        If path to NEMO model grid is provided, domain is inferred.

        Parameters
        ----------
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
        grid : str, optional
            Path to NEMO model grid (e.g., 'gridT').

        Returns
        -------
        dict[str, str]
            NEMO model grid index names.
        """
        if grid is not None:
            dom, _, dom_suffix, _ = self._get_properties(grid=grid, infer_dom=True)
        else:
            _, dom_suffix = self._get_properties(dom=dom)

        indexes = ["i", "j", "k"]
        if dom == ".":
            d_ijk = {index: index for index in indexes}
        else:
            d_ijk = {index: f"{index}{dom_suffix}" for index in indexes}

        return d_ijk

    def _get_weights(self, grid: str, dims: list, fillna: bool = True) -> xr.DataArray:
        """
        Get the weights (scale factors) for specified dimensions of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where weights are stored (e.g., 'gridT').
        dims : list
            Dimensions to collect weights for.
        fillna : bool, optional
            Fill NaN values in weights with zeros. Default is True.

        Returns
        -------
        xr.DataArray
            Weights (scale factors) for the specified dimensions of the NEMO model grid.
        """
        if any(dim not in ["i", "j", "k"] for dim in dims):
            raise ValueError(
                "dims must be a list containing one or more of the following dimensions: ['i', 'j', 'k']."
            )

        grid_suffix = self._get_properties(grid=grid)

        weights_dict = {
            "i": f"e1{grid_suffix}",
            "j": f"e2{grid_suffix}",
            "k": f"e3{grid_suffix}",
        }
        try:
            weights_list = [self[f"{grid}"][weights_dict[dim]] for dim in dims]
        except KeyError as e:
            raise KeyError(
                f"weights missing for dimensions {dims} of NEMO model grid {grid}"
            ) from e

        if len(weights_list) == 1:
            weights = weights_list[0]
        elif len(weights_list) == 2:
            weights = weights_list[0] * weights_list[1]
        elif len(weights_list) == 3:
            weights = weights_list[0] * weights_list[1] * weights_list[2]

        if fillna:
            weights = weights.fillna(value=0)

        return weights

    def add_geoindex(
        self,
        grid: str,
    ) -> Self:
        """
        Add geographical index variables to a given NEMO model grid.

        This enables users to index grid variables using geographical
        coordinates (e.g., glamt, gphit) in addition to their (i, j, k)
        dimensions.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid to add geographical indexes (e.g., 'gridT').

        Returns
        -------
        NEMODataTree
            NEMO DataTree with geographical indexes added to specified model grid.

        Examples
        --------
        Add glamt, gphit as geographical indexes to the T-grid of the NEMO parent domain:

        >>> nemo.add_geoindex(grid="gridT")

        """
        # -- Set geographical indexes -- #
        _, dom_prefix, _, grid_suffix = self._get_properties(grid=grid, infer_dom=True)
        lon_name = f"{dom_prefix}glam{grid_suffix}"
        lat_name = f"{dom_prefix}gphi{grid_suffix}"
        self_copy = self.copy()
        self_copy[grid] = (
            self_copy[grid]
            .dataset.assign_coords(
                {lat_name: self_copy[grid][lat_name], lon_name: self_copy[grid][lon_name]}
            )
            .set_xindex(
                (lat_name, lon_name),
                NDPointIndex,
                tree_adapter_cls=SklearnGeoBallTreeAdapter,
            )
        )

        return self_copy

    def cell_area(
        self,
        grid: str,
        dim: str,
    ) -> xr.DataArray:
        """
        Calculate grid cell areas orthogonal to a given dimension of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid from which to calculate grid cell areas
            (e.g., 'gridT').
        dim : str
            Dimension orthogonal to grid cell area to
            calculate (e.g., 'k' returns e1 * e2).

        Returns
        -------
        xr.DataArray
            Grid cell areas (m^2) for the specified NEMO model grid.

        Examples
        --------
        Compute the horizontal area of each grid cell centered on a V-grid point
        in the NEMO parent domain:

        >>> nemo.cell_area(grid="gridT", dim="k")

        Note, `dim` represents the dimension orthogonal to the grid cell
        area to be computed.

        See Also
        --------
        cell_volume
        """
        grid_suffix = self._get_properties(grid=grid)

        if dim not in ["i", "j", "k"]:
            raise ValueError(f"dim {dim} must be one of ['i', 'j', 'k'].")

        match dim:
            case "i":
                cell_area = (
                    self[f"{grid}/e3{grid_suffix}"].masked.data * self[f"{grid}/e2{grid_suffix}"].masked.data
                )
            case "j":
                cell_area = (
                    self[f"{grid}/e3{grid_suffix}"].masked.data * self[f"{grid}/e1{grid_suffix}"].masked.data
                )
            case "k":
                cell_area = (
                    self[f"{grid}/e1{grid_suffix}"].masked.data * self[f"{grid}/e2{grid_suffix}"].masked.data
                )
        cell_area.name = "areacello"

        return cell_area

    def cell_volume(self, grid: str) -> xr.DataArray:
        """
        Calculate grid cell volumes for a given NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid to calculate grid cell volumes
            (e.g., 'gridT').

        Returns
        -------
        xr.DataArray
            Grid cell volumes for the specified NEMO model grid.

        Examples
        --------
        Compute the volume of each grid cell centered on a V-grid point
        in the NEMO parent domain:

        >>> nemo.cell_volumes(grid="gridV")

        See Also
        --------
        cell_area
        """
        grid_suffix = self._get_properties(grid=grid)

        cell_volume = (
            self[f"{grid}/e3{grid_suffix}"].masked.data
            * self[f"{grid}/e1{grid_suffix}"].masked.data
            * self[f"{grid}/e2{grid_suffix}"].masked.data
        )
        cell_volume.name = "volcello"

        return cell_volume

    @deprecated(version_since="2026.03.b1",
                version_removed="2026.07",
                alternative="NEMODataArray.derivative or NEMOVectorField.gradient from v2026.07 onwards"
                )
    def gradient(
        self,
        var: str,
        dim: str,
        dom: str = ".",
    ) -> xr.DataArray:
        """
        Calculate the gradient of a scalar variable along one dimension (e.g., 'i', 'j', 'k') of a NEMO model grid.

        Parameters
        ----------
        var : str
            Name of the scalar variable.
        dim : str
            Dimension along which to calculate gradient (e.g., 'i', 'j', 'k').
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Gradient of scalar variable defined on a NEMO model grid.

        Examples
        --------
        Compute the 'meridional' gradient of sea surface temperature `tos_con`
        along the NEMO parent domain `j` dimension:

        >>> nemo.gradient(dom='.', var="tos_con", dim="j")

        Compute the vertical gradient of absolute salinity in the first NEMO
        nested child domain:

        >>> nemo.gradient(dom="1", var="so_abs", dim="k")

        See Also
        --------
        integral
        """
        # -- Validate input -- #
        if not isinstance(var, str):
            raise ValueError(
                "var must be a string specifying name of the scalar variable."
            )
        if not isinstance(dim, str):
            raise ValueError(
                "dim must be a string specifying dimension along which to calculate the gradient (e.g., 'i', 'j', 'k')."
            )
        if not isinstance(dom, str):
            raise ValueError(
                "dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )

        # -- Get NEMO model grid properties -- #
        dom_prefix, dom_suffix = self._get_properties(dom=dom)
        grid_paths = self._get_grid_paths(dom=dom)
        gridT, gridU, gridV, gridW = (
            grid_paths["gridT"],
            grid_paths["gridU"],
            grid_paths["gridV"],
            grid_paths["gridW"],
        )

        if var not in self[gridT].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{gridT}'.")

        da = self[f"{gridT}/{var}"].masked.data
        dim_name = f"{dim}{dom_suffix}"
        if dim_name not in da.dims:
            raise KeyError(
                f"dimension '{dim_name}' not found in variable '{var}'. Dimensions available: {da.dims}."
            )

        match dim:
            case "i":
                if f"{dom_prefix}deptht" in da.coords:
                    # 3-dimensional umask:
                    umask = self[gridU]["umask"]
                else:
                    # 2-dimensional umask:
                    umask = self[gridU]["umaskutil"]

                # Zonally Periodic Domain:
                if self[gridT].attrs.get("iperio", False):
                    da_end = da.isel({dim_name: 0})
                    da_end[dim_name] = da[dim_name].max() + 1
                    da = xr.concat([da, da_end], dim=dim_name)
                    dvar = da.diff(dim=dim_name, label="lower")
                else:
                    # Non-Periodic: pad with NaN values after differencing:
                    dvar = da.diff(dim=dim_name, label="lower").pad({dim_name: (0, 1)})
                # Apply u-mask & transform coords -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                gradient = dvar.where(umask) / self[f"{gridU}/e1u"].masked.data

                # Remove redundant depth coordinates:
                if f"{dom_prefix}deptht" in gradient.coords:
                    gradient = gradient.drop_vars(
                        [f"{dom_prefix}deptht"]
                    ).assign_coords(
                        {f"{dom_prefix}depthu": self[gridU][f"{dom_prefix}depthu"]}
                    )
            case "j":
                # 3-dimensional vmask:
                if f"{dom_prefix}deptht" in da.coords:
                    vmask = self[gridV]["vmask"]
                else:
                    # 2-dimensional vmask (unique points):
                    vmask = self[gridV]["vmaskutil"]

                # Pad with zeros after differencing (zero gradient at jmaxdom):
                dvar = da.diff(dim=dim_name, label="lower").pad(
                    {dim_name: (0, 1)}, constant_values=0
                )
                # Apply vmask & transform coords -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                gradient = dvar.where(vmask) / self[f"{gridV}/e2v"].masked.data

                if f"{dom_prefix}deptht" in gradient.coords:
                    gradient = gradient.drop_vars(
                        [f"{dom_prefix}deptht"]
                    ).assign_coords(
                        {f"{dom_prefix}depthv": self[gridV][f"{dom_prefix}depthv"]}
                    )

            case "k":
                dvar = da.diff(dim=dim_name, label="lower")
                # Transform coords & apply w-mask -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                dvar = dvar.where(self[gridW]["wmask"].isel({dim_name: slice(1, None)}))
                try:
                    gradient = -dvar / self[f"{gridW}/e3w"].masked.data.isel(
                        {dim_name: slice(1, None)}
                    )
                    gradient = gradient.drop_vars([f"{dom_prefix}deptht"])
                except KeyError as e:
                    raise KeyError(
                        f"NEMO model grid: '{gridW}' does not contain vertical scale factor 'e3w' required to calculate gradients along the k-dimension."
                    ) from e

        # Update DataArray properties:
        gradient.name = f"grad_{dim_name}({var})"
        gradient = gradient.drop_vars([f"{dom_prefix}glamt", f"{dom_prefix}gphit"])

        return gradient

    @deprecated(version_since="2026.03.b1",
                version_removed="2026.07",
                alternative="NEMOVectorField.divergence from v2026.07 onwards"
                )
    def divergence(
        self,
        vars: list[str],
        dom: str = ".",
    ) -> xr.DataArray:
        """
        Calculate the horizontal divergence of a vector field defined on a NEMO model grid.

        Parameters
        ----------
        vars : list[str]
            Name of vector variables, structured as: ['u', 'v'], where
            'u' and 'v' are the i and j components of the vector field,
            respectively.
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Horizontal divergence of vector field defined on a NEMO model grid.

        Examples
        --------
        Compute the horizontal divergence of the seawater velocity field in the
        NEMO parent domain:

        >>> nemo.divergence(dom=".", vars=["uo", "vo"])

        Note, `vars` expects a list of the `i` and `j` components of the vector
        field, respectively.

        See Also
        --------
        divergence
        """
        # -- Validate input -- #
        if not isinstance(vars, list) or len(vars) != 2:
            raise ValueError(
                "vars must be a list of two elements structured as ['u', 'v']."
            )
        if not isinstance(dom, str):
            raise ValueError(
                "dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )

        # -- Get NEMO model grid properties -- #
        dom_prefix, _ = self._get_properties(dom=dom)
        grid_paths = self._get_grid_paths(dom=dom)
        gridT, gridU, gridV = (
            grid_paths["gridT"],
            grid_paths["gridU"],
            grid_paths["gridV"],
        )
        ijk_names = self._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names["i"], ijk_names["j"]

        # -- Define i,j vector components -- #
        var_i, var_j = vars[0], vars[1]
        if var_i not in self[gridU].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{gridU}'.")
        if var_j not in self[gridV].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{gridV}'.")

        da_i = self[f"{gridU}/{var_i}"].masked.data
        da_j = self[f"{gridV}/{var_j}"].masked.data

        # -- Collect mask -- #
        if (f"{dom_prefix}depthu" in da_i.coords) and (
            f"{dom_prefix}depthv" in da_j.coords
        ):
            # 3-dimensional tmask:
            tmask = self[gridT]["tmask"]
        else:
            # 2-dimensional tmask (unique points):
            tmask = self[gridT]["tmaskutil"]

        # -- Neglecting the first T-grid points along i, j dimensions -- #
        e1t = self[f"{gridT}/e1t"].masked.data.isel({i_name: slice(1, None), j_name: slice(1, None)})
        e2t = self[f"{gridT}/e2t"].masked.data.isel({i_name: slice(1, None), j_name: slice(1, None)})
        e3t = self[f"{gridT}/e3t"].masked.data.isel({i_name: slice(1, None), j_name: slice(1, None)})

        e2u, e3u = self[f"{gridU}/e2u"].masked.data, self[f"{gridU}/e3u"].masked.data
        e1v, e3v = self[f"{gridV}/e1v"].masked.data, self[f"{gridV}/e3v"].masked.data  

        # -- Calculate divergence on T-points -- #
        dvar_i = (e2u * e3u * da_i).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1v * e3v * da_j).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        divergence = (1 / (e1t * e2t * e3t)) * (dvar_i + dvar_j).where(tmask)

        # -- Update DataArray properties -- #
        divergence.name = f"div({var_i}, {var_j})"
        divergence = divergence.drop_vars(
            [
                f"{dom_prefix}glamu",
                f"{dom_prefix}gphiu",
                f"{dom_prefix}glamv",
                f"{dom_prefix}gphiv",
                f"{dom_prefix}depthu",
                f"{dom_prefix}depthv",
            ]
        )

        return divergence

    @deprecated(version_since="2026.03.b1",
                version_removed="2026.07",
                alternative="NEMOVectorField.curl from v2026.07 onwards"
                )
    def curl(
        self,
        vars: list[str],
        dom: str = ".",
    ) -> xr.DataArray:
        """
        Calculate the vertical (k) curl component of a vector field on a NEMO model grid.

        Parameters
        ----------
        vars : list[str]
            Name of the vector variables, structured as: ['u', 'v'], where 'u' and 'v' are
            the i and j components of the vector field, respectively.
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Vertical curl component of vector field defined on a NEMO model grid.

        Examples
        --------
        Compute the vertical component of the curl of the seawater velocity field in
        the second NEMO nested child domain:

        >>> nemo.curl(dom="2", vars=["uo", "vo"])

        Note, `vars` expects a list of the `i` and `j` components of the vector field,
        respectively.

        See Also
        --------
        divergence
        """
        # -- Validate input -- #
        if not isinstance(vars, list) or len(vars) != 2:
            raise ValueError(
                "vars must be a list of two elements structured as ['u', 'v']."
            )
        if not isinstance(dom, str):
            raise ValueError(
                "dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )

        # -- Get NEMO model grid properties -- #
        dom_prefix, _ = self._get_properties(dom=dom)
        grid_paths = self._get_grid_paths(dom=dom)
        gridU, gridV, gridF = (
            grid_paths["gridU"],
            grid_paths["gridV"],
            grid_paths["gridF"],
        )
        ijk_names = self._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names["i"], ijk_names["j"]

        # -- Define i,j vector components -- #
        var_i, var_j = vars[0], vars[1]
        if var_i not in self[gridU].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{gridU}'.")
        if var_j not in self[gridV].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{gridV}'.")

        da_i = self[f"{gridU}/{var_i}"].masked.data
        da_j = self[f"{gridV}/{var_j}"].masked.data

        # -- Collect mask -- #
        if (f"{dom_prefix}depthu" in da_i.coords) and (
            f"{dom_prefix}depthv" in da_j.coords
        ):
            # 3-dimensional fmask
            fmask = self[gridF]["fmask"]
        else:
            # 2-dimensional fmask (unique points):
            fmask = self[gridF]["fmaskutil"]

        # -- Neglecting the final F-grid points along i, j dimensions -- #
        e1f = self[f"{gridF}/e1f"].masked.data.isel(
            {i_name: slice(None, -1), j_name: slice(None, -1)}
        )
        e2f = self[f"{gridF}/e2f"].masked.data.isel(
            {i_name: slice(None, -1), j_name: slice(None, -1)}
        )

        e1u = self[f"{gridU}/e1u"].masked.data
        e2v = self[f"{gridV}/e2v"].masked.data
        # -- Calculate vertical curl component on F-points -- #
        dvar_i = (e2v * da_j).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1u * da_i).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        curl = (1 / (e1f * e2f)) * (dvar_i - dvar_j).where(fmask)

        # -- Update DataArray properties -- #
        curl.name = f"curl({var_i}, {var_j})"
        curl = curl.drop_vars(
            [
                f"{dom_prefix}glamu",
                f"{dom_prefix}gphiu",
                f"{dom_prefix}glamv",
                f"{dom_prefix}gphiv",
            ]
        )

        return curl

    def clip_grid(
        self,
        grid: str,
        bbox: tuple,
    ) -> Self:
        """
        Clip a NEMO model grid to specified longitude and latitude range.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid to clip (e.g., 'gridT').
        bbox : tuple
            Bounding box to clip to (lon_min, lon_max, lat_min, lat_max).

        Returns
        -------
        NEMODataTree
            NEMO DataTree with specified model grid clipped to bounding box.

        Examples
        --------
        Clip T-grid in a NEMO parent domain in the bounding box (-80°E, 0°E, 40°N, 80°N):

        >>> bbox = (-80, 0, 40, 80)

        >>> nemo.clip_grid(grid="gridT", bbox=bbox)

        See Also
        --------
        clip_domain
        """
        # -- Validate input -- #
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError(
                "bounding box must be a tuple (lon_min, lon_max, lat_min, lat_max)."
            )

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, _, grid_suffix = self._get_properties(grid=grid, infer_dom=True)

        # -- Clip the grid to given bounding box -- #
        # Indexing with a mask requires loading coords into memory:
        glam = self[grid][f"{dom_prefix}glam{grid_suffix}"].load()
        gphi = self[grid][f"{dom_prefix}gphi{grid_suffix}"].load()

        grid_clipped = self[grid].dataset.where(
            (glam >= bbox[0])
            & (glam <= bbox[1])
            & (gphi >= bbox[2])
            & (gphi <= bbox[3]),
            drop=True,
        )

        d_dtypes = {var: self[grid][var].dtype for var in self[grid].dataset.data_vars}
        for var, dtype in d_dtypes.items():
            if dtype in [np.int32, np.int64, bool]:
                grid_clipped[var] = grid_clipped[var].fillna(0).astype(dtype)

        if bbox != (-180, 180, -90, 90):
            # Update zonal periodicity of grid node:
            grid_clipped = grid_clipped.assign_attrs({"iperio": False})

        # Update shallow copy of NEMODataTree:
        self_copy = self.copy()
        self_copy[grid].dataset = grid_clipped

        return self_copy

    def clip_domain(
        self,
        dom: str,
        bbox: tuple,
    ) -> Self:
        """
        Clip a NEMO model domain to specified longitude and latitude range.

        Parameters
        ----------
        dom : str
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.
        bbox : tuple
            Bounding box to clip to (lon_min, lon_max, lat_min, lat_max).

        Returns
        -------
        NEMODataTree
            NEMO DataTree with specified model domain clipped to bounding box.

        Examples
        --------
        Clip all model grids in a NEMO parent domain in the bounding box
        (-80°E, 0°E, 40°N, 80°N):

        >>> bbox = (-80, 0, 40, 80)

        >>> nemo.clip_domain(dom=".", bbox=bbox)

        See Also
        --------
        clip_grid
        """
        # -- Validate input -- #
        if not isinstance(dom, str):
            raise ValueError(
                "dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError(
                "bounding box must be a tuple: (lon_min, lon_max, lat_min, lat_max)."
            )

        # -- Get NEMO model grid properties -- #
        grid_paths = self._get_grid_paths(dom=dom)
        ijk_names = self._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names["i"], ijk_names["j"]

        # -- Clip grids to given bounding box -- #
        if not grid_paths:
            raise ValueError(f"NEMO model domain '{dom}' not found in the DataTree.")
        else:
            for grid in grid_paths.values():
                # Identify grid type:
                grid_suffix = self._get_properties(grid=grid)

                if grid_suffix == "t":
                    # Clip shallow copy of NEMODataTree T-grid using lon/lat bbox:
                    self_copy = self.copy().clip_grid(grid=grid, bbox=bbox)
                    # Store (i, j) coords of bbox on T-grid:
                    i_bbox = self_copy[grid][i_name]
                    j_bbox = self_copy[grid][j_name]

                else:
                    # Clip adjacent horizontal grid using (i, j) coords of clipped T-grid:
                    match grid_suffix:
                        case "u":
                            grid_clipped = self[grid].dataset.sel(i=i_bbox + 0.5, j=j_bbox)
                        case "v":
                            grid_clipped = self[grid].dataset.sel(i=i_bbox, j=j_bbox + 0.5)
                        case "w":
                            grid_clipped = self[grid].dataset.sel(i=i_bbox, j=j_bbox)
                        case "f":
                            grid_clipped = self[grid].dataset.sel(i=i_bbox + 0.5, j=j_bbox + 0.5)

                    if bbox != (-180, 180, -90, 90):
                        # Update zonal periodicity of NEMO model grid:
                        grid_clipped = grid_clipped.assign_attrs({"iperio": False})
                    # Update shallow copy of NEMODataTree:
                    self_copy[grid].dataset = grid_clipped

        return self_copy

    def mask_with_polygon(
        self,
        grid: str,
        lon_poly: list | np.ndarray,
        lat_poly: list | np.ndarray,
    ):
        """
        Create mask of NEMO model grid points contained within a polygon.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where longitude and latitude coordinates
            are stored (e.g., 'gridT').
        lon_poly : list | ndarray
            Longitudes of closed polygon.
        lat_poly : list | ndarray
            Latitudes of closed polygon.

        Returns
        -------
        xr.DataArray
            Boolean mask identifying NEMO model grid points which are inside
            the polygon.

        Examples
        --------
        Create a regional boolean mask using the geographical coordinates of a closed
        polygon `lon_poly` and `lat_poly` in a NEMO parent domain:


        >>> nemo.mask_with_polygon(grid="gridT",
        ...                        lon_poly=lon_poly,
        ...                        lat_poly=lat_poly,
        ...                        )

        See Also
        --------
        masked_statistic
        """
        # -- Validate input -- #
        if not isinstance(lon_poly, (np.ndarray, list)) or not isinstance(
            lat_poly, (np.ndarray, list)
        ):
            raise TypeError(
                "longitude & latitude coordinates of polygon must be numpy arrays or lists."
            )
        if (lon_poly[0] != lon_poly[-1]) or (lat_poly[0] != lat_poly[-1]):
            raise ValueError(
                "longitude & latitude coordinates must form a closed polygon."
            )

        # -- Get NEMO model grid properties -- #
        dom, dom_prefix, _, grid_suffix = self._get_properties(grid=grid, infer_dom=True)
        ijk_names = self._get_ijk_names(grid=grid)
        i_name, j_name = ijk_names["i"], ijk_names["j"]

        if dom == ".":
            lon_name = f"glam{grid_suffix}"
            lat_name = f"gphi{grid_suffix}"
        else:
            lon_name = f"{dom_prefix}glam{grid_suffix}"
            lat_name = f"{dom_prefix}gphi{grid_suffix}"

        # -- Create mask using polygon coordinates -- #
        mask = create_polygon_mask(
            lon_grid=self[grid][lon_name],
            lat_grid=self[grid][lat_name],
            lon_poly=lon_poly,
            lat_poly=lat_poly,
            dims=(j_name, i_name),
        )

        return mask

    def extract_mask_boundary(
        self,
        mask: xr.DataArray,
        uv_vars: list | None = None,
        vars: list | None = None,
        dom: str = ".",
    ) -> xr.Dataset:
        """
        Extract the piecewise boundary of a masked region defined on a NEMO model grid.

        The boundary of the region represents the staggered set of NEMO U- and V-grid
        cell faces which form a continous section upon which normal velocities are
        defined. 

        Scalar variables defined on the NEMO model T-grid are linearly interpolated
        onto the appropriate U/V-grid cell face.

        Parameters
        ----------
        mask : xr.DataArray
            Boolean mask identifying NEMO model grid points which
            are inside the region of interest.
        uv_vars : list, optional
            Names of velocity variables to extract along the boundary.
            Default is ['uo', 'vo'].
        vars : list, optional
            Names of scalar variables to extract along the boundary.
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns
        -------
        xr.Dataset
            Dataset containing variables and NEMO model coordinates
            extracted along the boundary of the mask.

        Examples
        --------
        Extract normal velocities and absolute salinity along the boundary of
        a masked region in the NEMO parent domain:

        >>> nemo.extract_mask_boundary(mask=mask_osnap,
        ...                            uv_vars=["uo", "vo"],
        ...                            vars=["so_abs"],
        ...                            dom=".",
        ...                            )

        See Also
        --------
        extract_section
        """
        # -- Validate Input -- #
        if not isinstance(mask, xr.DataArray):
            raise TypeError("mask must be an xarray DataArray")
        if not isinstance(dom, str):
            raise TypeError(
                "dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )
        if uv_vars is None:
            uv_vars = ["uo", "vo"]
        else:
            if not isinstance(uv_vars, list) or len(uv_vars) != 2:
                raise TypeError(
                    "uv_vars must be a list of velocity variables to extract (e.g., ['uo', 'vo'])."
                )

        # -- Get NEMO model grid properties -- #
        _, dom_suffix = self._get_properties(dom=dom)

        # -- Extract mask boundary -- #
        if f"i{dom_suffix}" not in mask.dims or f"j{dom_suffix}" not in mask.dims:
            raise ValueError(
                f"mask must have dimensions 'i{dom_suffix}' and 'j{dom_suffix}'"
            )
        i_bdy, j_bdy, flux_type, flux_dir = get_mask_boundary(mask)

        # -- Construct boundary dataset -- #
        # Neglecting final indices -> duplicate of the first indices:
        ds_bdy = create_boundary_dataset(
            nemo=self,
            dom=dom,
            i_bdy=i_bdy[:-1],
            j_bdy=j_bdy[:-1],
            flux_type=flux_type[:-1],
            flux_dir=flux_dir[:-1],
        )

        # -- Update boundary dataset with extracted section data -- #
        ds_bdy = update_boundary_dataset(
            ds_bdy=ds_bdy,
            nemo=self,
            dom=dom,
            sec_indexes=None,
            uv_vars=uv_vars,
            vars=vars,
        )

        return ds_bdy

    def extract_section(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        uv_vars: list | None = None,
        vars: list | None = None,
        dom: str = ".",
    ) -> xr.Dataset:
        """
        Extract a piecewise hydrographic section from a NEMODataTree.

        Hydrographic section represents the staggered set of NEMO U-
        and V-grid cell faces which form a continous section upon which
        normal velocities are defined. 

        Scalar variables defined on the NEMO model T-grid are linearly
        interpolated onto the appropriate U/V-grid cell face.

        Parameters
        ----------
        lon : np.ndarray
            Longitudes defining the section polygon.
        lat : np.ndarray
            Latitudes defining the section polygon.
        uv_vars : list, optional
            Names of velocity variables to extract along the boundary.
            Default is ['uo', 'vo'].
        vars : list, optional
            Names of scalar variables to extract along the boundary.
        dom : str
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns
        -------
        xr.Dataset
            Dataset containing hydrographic section extracted from a NEMODataTree.

        Examples
        --------
        Extract normal velocities and potential density along the Overturning in the Subpolar
        North Atlantic (OSNAP) array defined by `lon_osnap` and `lat_osnap` coordinates in the
        NEMO parent domain:

        >>> nemo.extract_section(lon=lon_osnap,
        ...                      lat=lat_osnap,
        ...                      uv_vars=["uo", "vo"],
        ...                      vars=["sigma0"],
        ...                      dom=".",
        ...                      )

        See Also
        --------
        extract_mask_boundary
        """
        # -- Validate Input -- #
        if not isinstance(lon, np.ndarray):
            raise TypeError("lon must be a numpy array.")
        if not isinstance(lat, np.ndarray):
            raise TypeError("lat must be a numpy array.")
        if not isinstance(dom, str):
            raise TypeError(
                "dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )
        if uv_vars is None:
            uv_vars = ["uo", "vo"]
        else:
            if not isinstance(uv_vars, list) or len(uv_vars) != 2:
                raise TypeError(
                    "uv_vars must be a list of velocity variables to extract (e.g., ['uo', 'vo'])."
                )

        # -- Get NEMO model grid properties -- #
        grid_paths = self._get_grid_paths(dom=dom)

        # -- Define hydrographic section using polygon -- #
        lon_poly, lat_poly = create_section_polygon(
            lon_sec=lon,
            lat_sec=lat,
        )

        mask = self.mask_with_polygon(
            grid=grid_paths["gridT"], lon_poly=lon_poly, lat_poly=lat_poly
        )

        i_bdy, j_bdy, flux_type, flux_dir = get_mask_boundary(mask)

        # -- Create mask boundary dataset -- #
        ds_bdy = create_boundary_dataset(
            nemo=self,
            dom=dom,
            i_bdy=i_bdy,
            j_bdy=j_bdy,
            flux_type=flux_type,
            flux_dir=flux_dir,
        )

        # -- Get indexes of hydrographic section along mask boundary -- #
        dom_prefix, _ = self._get_properties(dom=dom)
        sec_indexes = get_section_indexes(
            lon_section=lon,
            lat_section=lat,
            gphib=ds_bdy[f"{dom_prefix}gphib"].values,
            glamb=ds_bdy[f"{dom_prefix}glamb"].values,
            bdy=ds_bdy["bdy"].values,
        )

        # -- Update boundary dataset with extracted section data -- #
        ds_bdy = update_boundary_dataset(
            ds_bdy=ds_bdy,
            nemo=self,
            dom=dom,
            sec_indexes=sec_indexes,
            uv_vars=uv_vars,
            vars=vars,
        )

        return ds_bdy

    def extract_zonal_section(
        self,
        lat: int | float,
        lon_min: int | float,
        lon_max: int | float,
        u_vars: list[str] | None = None,
        scalar_vars: list[str] | None = None,
        dom: str = '.',
    ) -> xr.Dataset:
        """
        Extract an approximately zonal section at a chosen latitude from a NEMODataTree.

        Hydrographic section will be located at the constant j-coordinate whose average
        latitude (following selection between lon_min and lon_max) is closest to the given
        latitude.

        Hydrographic section will be defined on the NEMO model V-grid
        
        Scalar variables defined on the NEMO model T-grid and vector variables defined on
        the NEMO model U-grid are linearly interpolated onto the V-grid.

        Parameters:
        -----------
        lat : int | float
            Latitude of zonal section.
        lon_min : int | float
            Minimum longitude of zonal section.
        lon_max : int | float
            Maximum longitude of zonal section.
        u_vars : list[str], optional
            List of NEMO U-grid variables to extract along
            zonal section.
        scalar_vars : list[str], optional
            List of scalar variable names to extract along
            zonal section.
        dom : str
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Default is '.' for the parent domain.

        Returns:
        --------
        xr.Dataset
            Dataset containing zonal hydrographic section extracted from a NEMODataTree.

        Examples
        --------
        Extract meridional velocities, conservative temperature and zonal wind stress along the RAPID-MOCHA array located
        at 26.5N in the NEMO parent domain:

        >>> nemo.extract_zonal_section(lat=26.5,
        ...                            lon_min=-81,
        ...                            lon_max=10,
        ...                            u_vars=["tauuo"],
        ...                            scalar_vars=["thetao_con"],
        ...                            dom=".",
        ...                            )
        See Also
        --------
        extract_section
        """
        # -- Validate Inputs -- #
        if not isinstance(lat, (int, float)):
            raise TypeError("Latitude must be a single numeric value.")
        if not isinstance(lon_min, (int, float)):
            raise TypeError("Minimum longitude must be a single numeric value.")
        if not isinstance(lon_max, (int, float)):
            raise TypeError("Maximum longitude must be a single numeric value.")
        if u_vars is not None and not isinstance(u_vars, list):
            raise TypeError("u_vars must be a list of variable names.")
        if scalar_vars is not None and not isinstance(scalar_vars, list):
            raise TypeError("scalar_vars must be a list of variable names.")
        if not isinstance(dom, str):
            raise TypeError(
                "dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.)."
            )

        # -- Get NEMO model grid properties -- #
        grid_paths = self._get_grid_paths(dom=dom)
        dom_prefix, _ = self._get_properties(dom=dom)
        ijk_names = self._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names["i"], ijk_names["j"]

        # -- Validate latitude within latitude bounds of NEMODataTree -- #
        lat_min_grid = self[grid_paths["gridV"]][f"{dom_prefix}gphiv"].min().values.item()
        lat_max_grid = self[grid_paths["gridV"]][f"{dom_prefix}gphiv"].max().values.item()
        if (lat < lat_min_grid) or (lat > lat_max_grid):
            raise ValueError(f"Latitude of zonal section is out of bounds of the grid latitude range ({lat_min_grid}, {lat_max_grid}).")

        # -- Add geographical indexing to V-grid -- #
        nemo_geo = self.add_geoindex(grid=grid_paths["gridV"])

        # -- Determine (i, j) indices of zonal section endpoints -- #
        nemo_start = nemo_geo[grid_paths["gridV"]].dataset.sel(gphiv=lat, glamv=lon_min, method='nearest')
        i_start = nemo_start[i_name].values.item()
        j_start = nemo_start[j_name].values.item()
        nemo_end = nemo_geo[grid_paths["gridV"]].dataset.sel(gphiv=lat, glamv=lon_max, method='nearest')
        i_end = nemo_end[i_name].values.item()
        j_end = nemo_end[j_name].values.item()

        # Define j-index of zonal section:
        if j_start == j_end:
            j_sec = j_start
        else:
            j_list = np.arange(min([j_start, j_end]), max([j_start, j_end]) + 1)
            j_lats = []
            for j in j_list:
                j_lats.append(nemo_geo[grid_paths['gridV']]["gphiv"]
                              .sel({i_name: slice(i_start, i_end), j_name: j})
                              .mean(dim=i_name).values.item()
                              )
            # Select j-index of zonal section closest to specified latitude:
            j_sec = j_list[np.argmin(np.abs(np.array(j_lats) - lat))]

        # Define i-indices of zonal section:
        i_sec = [i_start, i_end]

        # -- Extract zonal section at specified latitude -- #
        nemo_geo = nemo_geo.isel({i_name: slice(i_sec[0]-1, i_sec[1]+1),
                                  j_name: slice(int(j_sec)-1, int(j_sec)+1)
                                  })
        # Transform scalar T-grid variables to V-point grid:
        if scalar_vars is not None:
            for var in scalar_vars:
                nemo_geo[grid_paths["gridV"]][var] = nemo_geo[f"gridT/{var}"].interp_to(to='V')
        # Transform U-grid variables to V-point grid:
        if u_vars is not None:
            for var in u_vars:
                nemo_geo[grid_paths["gridV"]][var] = nemo_geo[f"gridU/{var}"].interp_to(to='V')

        # Extract zonal section & update dimensions & coordinate variables:
        ds_bdy = nemo_geo[grid_paths["gridV"]].dataset.sel({i_name: slice(i_sec[0], i_sec[1]),
                                                            j_name: j_sec
                                                            })
        ds_bdy = (ds_bdy
                  .rename_vars({"glamv": f"{dom_prefix}glamb",
                                "gphiv": f"{dom_prefix}gphib",
                                "depthv": f"{dom_prefix}depthb",
                                "vmask": "bmask",
                                "vmaskutil": "bmaskutil",
                                "e1v": "e1b",
                                "e2v": "e2b",
                                "e3v": "e3b",
                                })
                  .rename_dims({i_name: "bdy"})
                  )

        if ("e3v_0" in ds_bdy.data_vars) and ("hv_0" in ds_bdy.data_vars):
            ds_bdy = ds_bdy.rename_vars({"e3v_0": "e3b_0",
                                         "hv_0": "hb_0"
                                         })

        # Add (i_bdy, j_bdy) indices of zonal section:
        ds_bdy["i_bdy"] = xr.DataArray(data=np.arange(i_sec[0], i_sec[1]+1), dims=["bdy"])
        ds_bdy["j_bdy"] = xr.DataArray(data=np.repeat(j_sec, ds_bdy["bdy"].size), dims=["bdy"])

        # Apply land-sea masks to all variables:
        for var in ds_bdy.data_vars:
            if ds_bdy[var].dims == ("time_counter", "k", "bdy"):
                ds_bdy[var] = ds_bdy[var].where(ds_bdy["bmask"])
            elif ds_bdy[var].dims == ("time_counter", "bdy"):
                ds_bdy[var] = ds_bdy[var].where(ds_bdy["bmaskutil"])

        # Drop invalid global attributes from dataset:
        ds_bdy.attrs.pop("iperio", None)
        ds_bdy.attrs.pop("nftype", None)

        return ds_bdy

    def binned_statistic(
        self,
        grid: str,
        vars: list[str],
        values: str,
        keep_dims: list[str] | None,
        bins: list[list | np.ndarray],
        statistic: str,
        mask: xr.DataArray | None,
    ) -> xr.DataArray:
        """
        Calculate binned statistic of a variable defined on a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variables and values are stored
            (e.g., 'gridT').
        vars : list[str]
            Names of variable(s) to be grouped in discrete bins.
        values : str
            Name of the values with which to calculate binned statistic.
        keep_dims : list[str] | None
            Names of dimensions in values to keep as labels in binned statistic.
        bins : list[list | np.ndarray]
            Bin edges used to group each of the variables in `vars`.
        statistic : str
            Statistic to calculate (e.g., 'count', 'sum', 'nansum', 'mean', 'nanmean',
            'max', 'nanmax', 'min', 'nanmin'). See flox.xarray.xarray_reduce for a
            complete list of aggregation statistics.
        mask : xr.DataArray | None
            Boolean mask identifying NEMO model grid points to be included (1)
            or neglected (0) from calculation.

        Returns
        -------
        xr.DataArray
            Values of the selected statistic in each bin.

        Examples
        --------
        Compute the mean depth associated with each isopycnal in discrete potential
        density `sigma0` coordinates:

        >>> sigma0_bins = np.arange(22, 29.05, 0.05)

        >>> nemo.binned_statistic(grid="gridT",
        ...                       vars=["sigma0"],
        ...                       values="deptht",
        ...                       keep_dims=["time_counter"],
        ...                       bins=[sigma0_bins],
        ...                       statistic="nanmean",
        ...                       )

        See Also
        --------
        masked_statistic
        """
        # -- Validate input -- #
        grid_keys = list(dict(self.subtree_with_keys).keys())
        if grid not in grid_keys:
            raise KeyError(
                f"grid '{grid}' not found in available NEMODataTree grids {grid_keys}."
            )
        if any(var not in self[grid].data_vars for var in vars):
            raise KeyError(f"one or more variables {vars} not found in grid '{grid}'.")
        if values not in self[grid].data_vars:
            raise KeyError(f"values '{values}' not found in grid '{grid}'.")
        if keep_dims is not None:
            if any(dim not in self[grid][values].dims for dim in keep_dims):
                raise KeyError(
                    f"one or more dimensions {keep_dims} not found in values '{values}'."
                )
        if not all(isinstance(bin, (list, np.ndarray)) for bin in bins):
            raise ValueError("bins must be a list of lists or numpy arrays.")
        if statistic not in [
            "all",
            "any",
            "count",
            "sum",
            "nansum",
            "mean",
            "nanmean",
            "max",
            "nanmax",
            "min",
            "nanmin",
            "argmax",
            "nanargmax",
            "argmin",
            "nanargmin",
            "quantile",
            "nanquantile",
            "median",
            "nanmedian",
            "mode",
            "nanmode",
            "first",
            "nanfirst",
            "last",
            "nanlast",
        ]:
            raise ValueError(f"statistic '{statistic}' is not supported.")
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise ValueError("mask must be an xarray.DataArray.")
            if mask.dtype != bool:
                raise TypeError("mask dtype must be boolean.")
            if any(dim not in self[grid].dims for dim in mask.dims):
                raise ValueError(
                    f"mask must have dimensions subset from {self[grid].dims}."
                )

        # -- Calculate binned statistics -- #
        values_data = self[f"{grid}/{values}"].masked.data
        var_data = [self[f"{grid}/{var}"].masked.data for var in vars]

        result = compute_binned_statistic(
            vars=var_data,
            values=values_data,
            keep_dims=keep_dims,
            bins=bins,
            statistic=statistic,
            mask=mask,
        )

        return result


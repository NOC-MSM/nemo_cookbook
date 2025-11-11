"""
nemodatatree.py

Description:
This module defines the NEMODataTree class, a hierarchical data structure
for analysing NEMO ocean general circulation outputs defining one or more
model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import dask
import numpy as np
import xarray as xr
from xarray.indexes import NDPointIndex
from nemo_cookbook.utils import SklearnGeoBallTreeAdapter
from typing import Self

from .interpolate import interpolate_grid
from .masks import create_polygon_mask, get_mask_boundary
from .processing import create_datatree_dict
from .stats import compute_binned_statistic
from .transform import transform_vertical_coords
from .extract import (
    create_section_polygon,
    get_section_indexes,
    update_boundary_dataset,
    create_boundary_dataset
    )


class NEMODataTree(xr.DataTree):
    """
    A hierarchical data structure containing collections of NEMO ocean model outputs.

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
        iperio: bool = False,
        nftype: str | None = None,
        read_mask: bool = False,
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

        iperio: bool = False
            Zonal periodicity of the parent domain. Default is False.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

        read_mask: bool = False
            If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed from top_level and bottom_level domain variables.

        nbghost_child : int = 4
            Number of ghost cells to remove from the western/southern boundaries of the (grand)child domains. Default is 4.

        **open_kwargs : dict, optional
            Additional keyword arguments to pass to `xarray.open_dataset` or `xr.open_mfdataset` when opening NEMO model output files.
        Returns
        -------
        NEMODataTree
            A hierarchical DataTree storing NEMO model outputs.
        """
        if not isinstance(paths, dict):
            raise TypeError("paths must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity of parent domain must be a boolean.")
        if nftype is not None and nftype not in ('T', 'F'):
            raise ValueError("north fold type of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None.")
        if not isinstance(read_mask, bool):
            raise TypeError("read_mask must be a boolean.")
        if not isinstance(nbghost_child, int):
            raise TypeError("number of ghost cells along the western/southern boundaries must be an integer.")
        if not isinstance(open_kwargs, dict):
            raise TypeError("open_kwargs must be a dictionary.")

        # Define parent, child, grandchild filepath collections:
        d_child, d_grandchild = None, None
        if 'parent' in paths.keys() and isinstance(paths['parent'], dict):
            for key in paths.keys():
                if key not in ('parent', 'child', 'grandchild'):
                    raise KeyError(f"Unexpected key '{key}' in paths dictionary.")
                if key == 'parent':
                    d_parent = paths['parent']
                elif key == 'child':
                    d_child = paths['child']
                elif key == 'grandchild':
                    d_grandchild = paths['grandchild']
        else:
            raise ValueError("Invalid paths structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains.")

        # Construct DataTree from parent / child / grandchild domains:
        d_tree = create_datatree_dict(d_parent=d_parent,
                                      d_child=d_child,
                                      d_grandchild=d_grandchild,
                                      nests=nests,
                                      iperio=iperio,
                                      nftype=nftype,
                                      read_mask=read_mask,
                                      nbghost_child=nbghost_child,
                                      open_kwargs=dict(**open_kwargs)
                                      )

        datatree = super().from_dict(d_tree)

        return datatree


    @classmethod
    def from_datasets(
        cls,
        datasets: dict[str, xr.Dataset],
        nests: dict[str, str] | None = None,
        iperio: bool = False,
        nftype: str | None = None,
        read_mask: bool = False,
        nbghost_child: int = 4
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

        nests : dict[str, dict[st, str]], optional
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

        iperio: bool = False
            Zonal periodicity of the parent domain.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

        read_mask: bool = False
            If True, read NEMO model land/sea mask from domain files. Default is False, meaning masks are computed from top_level and bottom_level domain variables.

        nbghost_child : int = 4
            Number of ghost cells to remove from the western/southern boundaries of the (grand)child domains. Default is 4.

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(datasets, dict):
            raise TypeError("datasets must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity of parent domain must be a boolean.")
        if nftype is not None and nftype not in ('T', 'F'):
            raise ValueError("north fold type of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None.")
        if not isinstance(read_mask, bool):
            raise TypeError("read_mask must be a boolean.")
        if not isinstance(nbghost_child, int):
            raise TypeError("number of ghost cells along the western/southern boundaries must be an integer.")

        # Define parent, child, grandchild dataset collections:
        d_child, d_grandchild = None, None
        if 'parent' in datasets.keys() and isinstance(datasets['parent'], dict):
            for key in datasets.keys():
                if key not in ('parent', 'child', 'grandchild'):
                    raise KeyError(f"Unexpected key '{key}' in datasets dictionary.")
                if key == 'parent':
                    d_parent = datasets['parent']
                elif key == 'child':
                    d_child = datasets['child']
                elif key == 'grandchild':
                    d_grandchild = datasets['grandchild']
        else:
            raise ValueError("Invalid datasets structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains.")

        # Construct DataTree from parent / child / grandchild domains:
        d_tree = create_datatree_dict(d_parent=d_parent,
                                      d_child=d_child,
                                      d_grandchild=d_grandchild,
                                      nests=nests,
                                      iperio=iperio,
                                      nftype=nftype,
                                      read_mask=read_mask,
                                      nbghost_child=nbghost_child
                                      )
        datatree = super().from_dict(d_tree)

        return datatree


    def _get_properties(
        cls,
        dom: str | None = None,
        grid: str | None = None,
        infer_dom: bool = False
        ) -> str:
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
            Path to NEMO model grid (e.g., '/gridT').
        infer_dom : bool, optional
            Whether to infer the domain number & domain name from only the
            grid path. Default is False.

        Returns
        -------
        tuple[str]
            NEMO model domain and grid properties.
        """
        if (grid is None) & (dom is not None):
            dom_prefix = "" if dom == "." else f"{dom}_"
            dom_suffix = "" if dom == "." else f"{dom}"
            return dom_prefix, dom_suffix
        else:
            if grid not in list(cls.subtree):
                raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
            grid_suffix = f"{grid.lower()[-1]}"

            if infer_dom:
                dom_inds = [char for char in grid if char.isdigit()]
                dom_prefix = f"{dom_inds[-1]}_" if len(dom_inds) != 0 else ""
                dom = dom_prefix[:-1] if dom_prefix != "" else "."
                dom_suffix = dom if dom != "." else ""
                return dom, dom_prefix, dom_suffix, grid_suffix
            else:
                return grid_suffix


    def _get_grid_paths(
        cls,
        dom: str
        ) -> str:
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
        grid_paths = [path[0] for path in list(cls.subtree_with_keys)]

        if dom == '.':
            grid_paths = [path for path in grid_paths if ("_" not in path) & ("grid" in path)]
        else:
            grid_paths = [path for path in grid_paths if dom in path]

        d_paths = {path.split("/")[0]: path for path in grid_paths}
        
        return d_paths


    def _get_ijk_names(
        cls,
        dom: str | None = None,
        grid: str | None = None
        ) -> str:
        """
        Get (i, j, k) grid index names for given NEMO model domain.

        If path to NEMO model grid is provided, domain is inferred.

        Parameters
        ----------
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
        grid : str, optional
            Path to NEMO model grid (e.g., '/gridT').

        Returns
        -------
        dict[str, str]
            NEMO model grid index names.
        """
        if grid is not None:
            dom, _, dom_suffix, _ = cls._get_properties(grid=grid, infer_dom=True)
        else:
            _, dom_suffix = cls._get_properties(dom=dom)

        indexes = ["i", "j", "k"]
        if dom == ".":
            d_ijk = {index: index for index in indexes}
        else:
            d_ijk = {index: f"{index}{dom_suffix}" for index in indexes}


        return d_ijk


    def _get_weights(
        cls,
        grid: str,
        dims: list,
        fillna : bool = True
        ) -> xr.DataArray:
        """
        Get the weights (scale factors) for specified dimensions of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where weights are stored (e.g., '/gridT').
        dims : list
            Dimensions to collect weights for.
        fillna : bool, optional
            Fill NaN values in weights with zeros. Default is True.

        Returns
        -------
        xr.DataArray
            Weights (scale factors) for the specified dimensions of the NEMO model grid.
        """
        if any(dim not in ['i', 'j', 'k'] for dim in dims):
            raise ValueError("dims must be a list containing one or more of the following dimensions: ['i', 'j', 'k'].")

        grid_suffix = cls._get_properties(grid=grid)

        weights_dict = {"i": f"e1{grid_suffix}",
                        "j": f"e2{grid_suffix}",
                        "k": f"e3{grid_suffix}",
                        }
        try:
            weights_list = [cls[grid][weights_dict[dim]] for dim in dims]
        except KeyError as e:
            raise KeyError(f"weights missing for dimensions {dims} of NEMO model grid {grid}: {e}")

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
        cls,
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
            Path to NEMO model grid to add geographical indexes.

        Returns
        -------
        NEMODataTree
            NEMO DataTree with geographical indexes added to specified model grid.
        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")

        # -- Set geographical indexes -- #
        grid_suffix = cls._get_properties(grid=grid)
        lon_name = f"glam{grid_suffix}"
        lat_name = f"gphi{grid_suffix}"
        cls_copy = cls.copy()
        cls_copy[grid] = (cls_copy[grid]
                            .dataset
                            .assign_coords({
                                lat_name: cls_copy[grid][lat_name],
                                lon_name: cls_copy[grid][lon_name]
                                })
                            .set_xindex(
                                (lat_name, lon_name),
                                NDPointIndex,
                                tree_adapter_cls=SklearnGeoBallTreeAdapter
                                )
                         )

        return cls_copy


    def cell_area(
        cls,
        grid: str,
        dim: str,
    ) -> xr.DataArray:
        """
        Calculate grid cell areas orthogonal to a given dimension of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid from which to calculate
            grid cell areas (e.g., '/gridT').
        dim : str
            Dimension orthogonal to grid cell area to
            calculate (e.g., 'k' returns e1 * e2).

        Returns
        -------
        xr.DataArray
            Grid cell areas (m^2) for the specified NEMO model grid.
        """
        grid_suffix = cls._get_properties(grid=grid)

        if dim not in ['i', 'j', 'k']:
            raise ValueError(f"dim {dim} must be one of ['i', 'j', 'k'].")

        match dim:
            case 'i':
                cell_area = cls[grid][f'e3{grid_suffix}'] * cls[grid][f'e2{grid_suffix}']
            case 'j':
                cell_area = cls[grid][f'e3{grid_suffix}'] * cls[grid][f'e1{grid_suffix}']
            case 'k':
                cell_area = cls[grid][f'e1{grid_suffix}'] * cls[grid][f'e2{grid_suffix}']
        cell_area.name = "areacello"

        return cell_area


    def cell_volume(
        cls,
        grid: str
    ) -> xr.DataArray:
        """
        Calculate grid cell volumes for a given NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid from which to calculate
            grid cell volumes (e.g., '/gridT').

        Returns
        -------
        xr.DataArray
            Grid cell volumes for the specified NEMO model grid.
        """
        grid_suffix = cls._get_properties(grid=grid)

        mask = cls[grid][f"{grid_suffix}mask"]

        cell_volume = cls[grid][f"e3{grid_suffix}"].where(mask) * cls[grid][f"e1{grid_suffix}"] * cls[grid][f"e2{grid_suffix}"]
        cell_volume.name = "volcello"

        return cell_volume


    def gradient(
        cls,
        var: str,
        dim: str,
        dom: str = '.',
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
        """
        # -- Validate input -- #
        if not isinstance(var, str):
            raise ValueError("var must be a string specifying name of the scalar variable.")
        if not isinstance(dim, str):
            raise ValueError("dim must be a string specifying dimension along which to calculate the gradient (e.g., 'i', 'j', 'k').")
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")

        # -- Get NEMO model grid properties -- #
        dom_prefix, dom_suffix = cls._get_properties(dom=dom)
        grid_paths = cls._get_grid_paths(dom=dom)
        gridT, gridU, gridV, gridW = grid_paths['gridT'], grid_paths['gridU'], grid_paths['gridV'], grid_paths['gridW']

        if var not in cls[gridT].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{gridT}'.")

        da = cls[gridT][var]
        dim_name = f"{dim}{dom_suffix}"
        if dim_name not in da.dims:
            raise KeyError(f"dimension '{dim_name}' not found in variable '{var}'. Dimensions available: {da.dims}.")

        match dim:
            case "i":
                if f"{dom_prefix}deptht" in da.coords:
                    # 3-dimensional umask:
                    umask = cls[gridU]["umask"]
                else:
                    # 2-dimensional umask:
                    umask = cls[gridU]["umaskutil"]

                # Zonally Periodic Domain:
                if cls[gridT].attrs.get("iperio", False):
                    da_end = da.isel(dim_name=0)
                    da_end[dim_name] = da[dim_name].max() + 1
                    da = xr.concat([da, da_end], dim=dim_name)
                    dvar = da.diff(dim=dim_name, label="lower")
                else:
                    # Non-Periodic: pad with NaN values after differencing:
                    dvar = (da
                            .diff(dim=dim_name, label="lower")
                            .pad({dim_name: (0, 1)})
                            )
                # Apply u-mask & transform coords -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                gradient = dvar.where(umask) / cls[gridU]["e1u"]

                # Remove redundant depth coordinates:
                if f"{dom_prefix}deptht" in gradient.coords:
                    gradient = (gradient
                                .drop_vars([f"{dom_prefix}deptht"])
                                .assign_coords({f"{dom_prefix}depthu": cls[gridU][f"{dom_prefix}depthu"]})
                                )
            case "j":
                # 3-dimensional vmask:
                if f"{dom_prefix}deptht" in da.coords:
                    vmask = cls[gridV]["vmask"]
                else:
                    # 2-dimensional vmask (unique points):
                    vmask = cls[gridV]["vmaskutil"]

                # Pad with zeros after differencing (zero gradient at jmaxdom):
                dvar = (da
                        .diff(dim=dim_name, label="lower")
                        .pad({dim_name: (0, 1)}, constant_values=0)
                        )
                # Apply vmask & transform coords -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                gradient = dvar.where(vmask) / cls[grid_paths['gridV']]["e2v"]

                if f"{dom_prefix}deptht" in gradient.coords:
                    gradient = (gradient
                                .drop_vars([f"{dom_prefix}deptht"])
                                .assign_coords({f"{dom_prefix}depthv": cls[gridV][f"{dom_prefix}depthv"]})
                                )

            case "k":
                dvar = da.diff(dim=dim_name, label="lower")
                # Transform coords & apply w-mask -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                dvar = dvar.where(cls[gridW]["wmask"].isel({dim_name: slice(1, None)}))
                try:
                    gradient = - dvar / cls[gridW]["e3w"].isel({dim_name: slice(1, None)})
                    gradient = gradient.drop_vars([f"{dom_prefix}deptht"])
                except KeyError:
                    raise KeyError(f"NEMO model grid: '{gridW}' does not contain vertical scale factor 'e3w' required to calculate gradients along the k-dimension.")

        # Update DataArray properties:
        gradient.name = f"grad_{var}_{dim_name}"
        gradient = gradient.drop_vars([f"{dom_prefix}glamt", f"{dom_prefix}gphit"])

        return gradient
    

    def divergence(
        cls,
        vars : list[str],
        dom: str = '.',
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
        """
        # -- Validate input -- #
        if not isinstance(vars, list) or len(vars) != 2:
            raise ValueError("vars must be a list of two elements structured as ['u', 'v'].")
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")

        # -- Get NEMO model grid properties -- #
        dom_prefix, _ = cls._get_properties(dom=dom)
        grid_paths = cls._get_grid_paths(dom=dom)
        gridT, gridU, gridV = grid_paths['gridT'], grid_paths['gridU'], grid_paths['gridV']
        ijk_names = cls._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names['i'], ijk_names['j']
        
        # -- Define i,j vector components -- #
        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[gridU].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{gridU}'.")
        if var_j not in cls[gridV].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{gridV}'.")

        da_i = cls[gridU][var_i]
        da_j = cls[gridV][var_j]

        # -- Collect mask -- #
        if (f"{dom_prefix}depthu" in da_i.coords) & (f"{dom_prefix}depthv" in da_j.coords):
            # 3-dimensional tmask:
            tmask = cls[gridT]["tmask"]
        else:
            # 2-dimensional tmask (unique points):
            tmask = cls[gridT]["tmaskutil"]

        # -- Neglecting the first T-grid points along i, j dimensions -- #
        e1t = cls[gridT]["e1t"].isel({i_name: slice(1, None), j_name: slice(1, None)})
        e2t = cls[gridT]["e2t"].isel({i_name: slice(1, None), j_name: slice(1, None)})
        e3t = cls[gridT]["e3t"].isel({i_name: slice(1, None), j_name: slice(1, None)})

        e2u, e3u = cls[gridU]["e2u"], cls[gridU]["e3u"]
        e1v, e3v = cls[gridV]["e1v"], cls[gridV]["e3v"]

        # -- Calculate divergence on T-points -- #
        dvar_i = (e2u * e3u * da_i).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1v * e3v * da_j).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        divergence = (1 / (e1t * e2t * e3t)) * (dvar_i + dvar_j).where(tmask)

        # -- Update DataArray properties -- #
        divergence.name = f"div_{var_i}_{var_j}"
        divergence = divergence.drop_vars([f"{dom_prefix}glamu", f"{dom_prefix}gphiu",
                                           f"{dom_prefix}glamv", f"{dom_prefix}gphiv",
                                           f"{dom_prefix}depthu", f"{dom_prefix}depthv"
                                           ])

        return divergence


    def curl(
        cls,
        vars : list[str],
        dom: str = '.',
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
        """
        # -- Validate input -- #
        if not isinstance(vars, list) or len(vars) != 2:
            raise ValueError("vars must be a list of two elements structured as ['u', 'v'].")
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")

        # -- Get NEMO model grid properties -- #
        dom_prefix, _ = cls._get_properties(dom=dom)
        grid_paths = cls._get_grid_paths(dom=dom)
        gridU, gridV, gridF = grid_paths['gridU'], grid_paths['gridV'], grid_paths['gridF']
        ijk_names = cls._get_ijk_names(dom=dom)
        i_name, j_name = ijk_names['i'], ijk_names['j']

        # -- Define i,j vector components -- #
        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[gridU].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{gridU}'.")
        if var_j not in cls[gridV].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{gridV}'.")

        da_i = cls[gridU][var_i]
        da_j = cls[gridV][var_j]

        # -- Collect mask -- #
        if (f"{dom_prefix}depthu" in da_i.coords) & (f"{dom_prefix}depthv" in da_j.coords):
            # 3-dimensional fmask
            fmask = cls[gridF]["fmask"]
        else:
            # 2-dimensional fmask (unique points):
            fmask = cls[gridF]["fmaskutil"]

        # -- Neglecting the final F-grid points along i, j dimensions -- #
        e1f = cls[gridF]["e1f"].isel({i_name: slice(None, -1), j_name: slice(None, -1)})
        e2f = cls[gridF]["e2f"].isel({i_name: slice(None, -1), j_name: slice(None, -1)})

        e1u = cls[gridU]["e1u"]
        e2v = cls[gridV]["e2v"]

        # -- Calculate vertical curl component on F-points -- #
        dvar_i = (e2v * da_j).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1u * da_i).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        curl = (1 / (e1f * e2f)) * (dvar_i - dvar_j).where(fmask)

        # -- Update DataArray properties -- #
        curl.name = f"curl_{var_i}_{var_j}"
        curl = curl.drop_vars([f"{dom_prefix}glamu", f"{dom_prefix}gphiu",
                               f"{dom_prefix}glamv", f"{dom_prefix}gphiv",
                               ])

        return curl


    def integral(
        cls,
        grid : str,
        var : str,
        dims : list,
        cum_dims : list | None = None,
        dir : str | None = None,
        mask : xr.DataArray | None = None
    ) -> xr.DataArray:
        """
        Integrate a variable along one or more dimensions of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variable is stored
            (e.g., '/gridT').
        var : str
            Name of variable to integrate.
        dims : list
            Dimensions over which to integrate (e.g., ['i', 'k']).
        cum_dims : list, optional
            Dimensions over which to cumulatively integrate (e.g., ['k']).
            Specified dimensions must also be included in `dims`.
        dir : str, optional
            Direction of cumulative integration. Options are '+1' (along
            increasing cum_dims) or '-1' (along decreasing cum_dims).
        mask: xr.DataArray, optional
            Boolean mask identifying NEMO model grid points to be included (1)
            or neglected (0) from integration.

        Returns
        -------
        xr.DataArray
            Variable integrated along specified dimensions of the NEMO model grid.

        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{grid}'.")
        if cum_dims is not None:
            for dim in cum_dims:
                if dim not in dims:
                    raise ValueError(f"cumulative integration dimension '{dim}' not included in `dims`.")
            if dir not in ['+1', '-1']:
                raise ValueError(f"invalid direction of cumulative integration '{dir}'. Expected '+1' or '-1'.")
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise ValueError("mask must be an xarray.DataArray.")
            if any(dim not in cls[grid].dims for dim in mask.dims):
                raise ValueError(f"mask must have dimensions subset from {cls[grid].dims}.")

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)

        # -- Collect variable, weights & mask -- #
        da = cls[grid][var].where(mask) if mask is not None else cls[grid][var]
        weights = cls._get_weights(grid=grid, dims=dims)

        if f"{dom_prefix}depth{grid_suffix}" in da.coords:
            # Apply 3-dimensional t/u/v/f/w mask:
            dom_mask = cls[grid][f"{grid_suffix}mask"]
        else:
            # Apply 2-dimensional t/u/v/f mask (unique points):
            hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'
            dom_mask = cls[grid][f"{hgrid_type}maskutil"]

        # -- Perform integration -- #
        if cum_dims is not None:
            sum_dims = [dim for dim in dims if dim not in cum_dims]
            if dir == '+1':
                # Cumulative integration along ordered dimension:
                result = da.where(dom_mask).weighted(weights).sum(dim=sum_dims, skipna=True).cumsum(dim=cum_dims, skipna=True)
            elif dir == '-1':
                # Cumulative integration along reversed dimension:
                result = (da
                            .where(dom_mask)
                            .weighted(weights)
                            .sum(dim=sum_dims, skipna=True)
                            .reindex({dim: cls[grid][dim][::-1] for dim in cum_dims})
                            .cumsum(dim=cum_dims, skipna=True)
                            )
        else:
            # Integration only:
            result = da.weighted(weights).sum(dim=dims, skipna=True)

        return result
   

    def clip_grid(
        cls,
        grid: str,
        bbox: tuple,
    ) -> Self:
        """
        Clip a NEMO model grid to specified longitude and latitude range.

        Parameters
        ----------
        Path to NEMO model grid to clip (e.g., '/gridT').
        bbox : tuple
            Bounding box to clip to (lon_min, lon_max, lat_min, lat_max).

        Returns
        -------
        NEMODataTree
            NEMO DataTree with specified model grid clipped to bounding box.
        """
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("bounding box must be a tuple (lon_min, lon_max, lat_min, lat_max).")

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)
        hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'

        # -- Clip the grid to given bounding box -- #
        # Indexing with a mask requires loading coords into memory:
        glam = cls[grid][f"{dom_prefix}glam{hgrid_type}"].load()
        gphi = cls[grid][f"{dom_prefix}gphi{hgrid_type}"].load()

        grid_clipped = cls[grid].dataset.where(
            (glam >= bbox[0]) &
            (glam <= bbox[1]) &
            (gphi >= bbox[2]) &
            (gphi <= bbox[3]),
            drop=True
            )

        d_dtypes = {var: cls[grid][var].dtype for var in cls[grid].dataset.data_vars}
        for var, dtype in d_dtypes.items():
            if dtype in [np.int32, np.int64, bool]:
                grid_clipped[var] = grid_clipped[var].fillna(0).astype(dtype)

        if bbox != (-180, 180, -90, 90):
            grid_clipped = grid_clipped.assign_attrs({"iperio": False})
        
        # Update shallow copy of NEMODataTree:
        cls_copy = cls.copy()
        cls_copy[grid] = grid_clipped

        return cls_copy


    def clip_domain(
        cls,
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
        """
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying the prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("bounding box must be a tuple: (lon_min, lon_max, lat_min, lat_max).")

        # -- Get NEMO model grid properties -- #
        dom_prefix, _ = cls._get_properties(dom=dom)
        grid_paths = cls._get_grid_paths(dom=dom)

        # -- Clip grids to given bounding box -- #
        if not grid_paths:
            raise ValueError(f"NEMO model domain '{dom}' not found in the DataTree.")
        else:
            # Update shallow copy of NEMODataTree:
            cls_copy = cls.copy()
            for grid in grid_paths.values():
                # Use (glamt, gphit) coords for W-grids:
                grid_suffix = cls._get_properties(grid=grid)
                hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'
                # Indexing with a mask requires eager loading:
                glam = cls[grid][f"{dom_prefix}glam{hgrid_type}"].load()
                gphi = cls[grid][f"{dom_prefix}gphi{hgrid_type}"].load()

                grid_clipped = cls[grid].dataset.where(
                    (glam >= bbox[0]) &
                    (glam <= bbox[1]) &
                    (gphi >= bbox[2]) &
                    (gphi <= bbox[3]),
                    drop=True
                    )

                d_dtypes = {var: cls[grid][var].dtype for var in cls[grid].dataset.data_vars}
                for var, dtype in d_dtypes.items():
                    if dtype in [np.int32, np.int64, bool]:
                        grid_clipped[var] = grid_clipped[var].fillna(0).astype(dtype)

                if bbox != (-180, 180, -90, 90):
                    grid_clipped = grid_clipped.assign_attrs({"iperio": False})
                cls_copy[grid] = grid_clipped

        return cls_copy


    def mask_with_polygon(
        cls,
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
            are stored (e.g., '/gridT').
        lon_poly : list | ndarray
            Longitudes of closed polygon.
        lat_poly : list | ndarray
            Latitudes of closed polygon.

        Returns
        -------
        xr.DataArray
            Boolean mask identifying NEMO model grid points which are inside
            the polygon.
        """
        # -- Validate input -- #
        if not isinstance(lon_poly, (np.ndarray, list)) or not isinstance(lat_poly, (np.ndarray, list)):
            raise TypeError("longitude & latitude coordinates of polygon must be numpy arrays or lists.")
        if (lon_poly[0] != lon_poly[-1]) or (lat_poly[0] != lat_poly[-1]):
            raise ValueError("longitude & latitude coordinates must form a closed polygon.")
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")

        # -- Get NEMO model grid properties -- #
        dom, dom_prefix, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)
        hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'
        ijk_names = cls._get_ijk_names(grid=grid)
        i_name, j_name = ijk_names['i'], ijk_names['j']

        if dom == ".":
            lon_name = f"glam{hgrid_type}"
            lat_name = f"gphi{hgrid_type}"
        else:
            lon_name = f"{dom_prefix}glam{hgrid_type}"
            lat_name = f"{dom_prefix}gphi{hgrid_type}"

        # -- Create mask using polygon coordinates -- #
        mask = create_polygon_mask(lon_grid=cls[grid][lon_name],
                                   lat_grid=cls[grid][lat_name],
                                   lon_poly=lon_poly,
                                   lat_poly=lat_poly,
                                   dims=(j_name, i_name)
                                   )

        return mask


    def masked_statistic(
        cls, 
        grid : str,
        var : str,
        lon_poly : list | np.ndarray,
        lat_poly : list | np.ndarray,
        statistic : str,
        dims : list
        ) -> xr.DataArray:
        """
        Masked statistic of a variable defined on a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variable is stored
            (e.g., '/gridT').
        var : str
            Name of the variable to compute statistic.
        lon_poly : list | np.ndarray
            Longitudes of closed polygon.
        lat_poly : list | np.ndarray
            Latitudes of closed polygon.
        statistic : str
            Name of the statistic to calculate (e.g., 'mean', 'weighted_mean' 'sum').
        dims : list
            Dimensions over which to apply statistic (e.g., ['i', 'j']).

        Returns
        -------
        xr.DataArray
            Masked statistic of specified variable.
        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{grid}'.")

        # -- Create polygon mask using coordinates -- #
        mask_poly = cls.mask_with_polygon(lon_poly=lon_poly,
                                          lat_poly=lat_poly,
                                          grid=grid
                                          )

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, dom_suffix, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)

        # -- Apply masks & calculate statistic -- #
        if f"{dom_prefix}depth{grid_suffix}" in cls[grid][var].coords:
            # Apply 3-dimensional t/u/v/f/w mask:
            dom_mask = cls[grid][f"{grid_suffix}mask"]
        else:
            # Apply 2-dimensional t/u/v/f mask (unique points):
            hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'
            dom_mask = cls[grid][f"{hgrid_type}maskutil"]

        da = cls[grid][var].where(dom_mask & mask_poly)

        match statistic:
            case "mean":
                result = da.mean(dim=dims, skipna=True)

            case "weighted_mean":
                weight_dims = [dim.replace(dom_suffix, "") for dim in dims]
                weights = cls._get_weights(grid=grid, dims=weight_dims)
                result = da.weighted(weights).mean(dim=dims, skipna=True)

            case "min":
                result = da.min(dim=dims, skipna=True)

            case "max":
                result = da.max(dim=dims, skipna=True)

            case "sum":
                result = da.sum(dim=dims, skipna=True)

        return result


    def extract_mask_boundary(
        cls,
        mask: xr.DataArray,
        uv_vars: list = ['uo', 'vo'],
        vars: list | None = None,
        dom: str = '.',
        ) -> xr.Dataset:
        """
        Extract the boundary of a masked region defined on a NEMO model grid.

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
        """
        if not isinstance(mask, xr.DataArray):
            raise ValueError("mask must be an xarray DataArray")
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")

        # -- Get NEMO model grid properties -- #
        dom_prefix, dom_suffix = cls._get_properties(dom=dom)
        grid_paths = cls._get_grid_paths(dom=dom)
        gridT, gridU, gridV = grid_paths['gridT'], grid_paths['gridU'], grid_paths['gridV']
        ijk_names = cls._get_ijk_names(dom=dom)
        k_name = ijk_names['k']
        
        # -- Extract mask boundary -- #
        if f'i{dom_suffix}' not in mask.dims or f'j{dom_suffix}' not in mask.dims:
            raise ValueError(f"mask must have dimensions f'i{dom_suffix}' and 'j{dom_suffix}'")
        i_bdy, j_bdy, flux_type, flux_dir = get_mask_boundary(mask)

        # -- Construct boundary dataset -- #
        time_name = [dim for dim in cls[gridU].dims if 'time' in dim][0]

        ds = xr.Dataset(
            data_vars={
            'i_bdy': (['bdy'], i_bdy[::-1]),
            'j_bdy': (['bdy'], j_bdy[::-1]),
            'flux_type': (['bdy'], flux_type[::-1]),
            'flux_dir': (['bdy'], flux_dir[::-1])
            },
            coords={
            time_name: cls[gridU][time_name].values,
            k_name: cls[gridU][k_name].values,
            'bdy': np.arange(len(i_bdy)),
            })

        # Add velocities normal to boundary:
        if uv_vars[0] not in cls[gridU].data_vars:
            raise KeyError(f"variable '{uv_vars[0]}' not found in grid '{gridU}'.")
        if uv_vars[1] not in cls[gridV].data_vars:
            raise KeyError(f"variable '{uv_vars[1]}' not found in grid '{gridV}'.")

        ubdy_mask = ds['flux_type'] == 'U'
        vbdy_mask = ds['flux_type'] == 'V'

        dim_sizes = [cls[gridU][time_name].size, cls[gridU][k_name].size, ds["bdy"].size]

        ds['velocity'] = xr.DataArray(data=dask.array.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
        ds['velocity'][:, :, ubdy_mask] = cls[gridU]['uo'].where(cls[gridU]['umask']).sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask]) * ds['flux_dir'][ubdy_mask]
        ds['velocity'][:, :, vbdy_mask] = cls[gridV]['vo'].where(cls[gridV]['vmask']).sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask]) * ds['flux_dir'][vbdy_mask]

        ds = ds.assign_coords({f"{dom_prefix}glamb": (['bdy'], np.zeros(ds["bdy"].size)),
                               f"{dom_prefix}gphib": (['bdy'], np.zeros(ds["bdy"].size)),
                               f"{dom_prefix}depthb": ((k_name, 'bdy'), np.zeros(dim_sizes[1:])),
                               })

        ds[f"{dom_prefix}glamb"][ubdy_mask] = cls[gridU]['glamu'].sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask])
        ds[f"{dom_prefix}glamb"][vbdy_mask] = cls[gridV]['glamv'].sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask])

        ds[f"{dom_prefix}gphib"][ubdy_mask] = cls[gridU]['gphiu'].sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask])
        ds[f"{dom_prefix}gphib"][vbdy_mask] = cls[gridV]['gphiv'].sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask])

        ds[f"{dom_prefix}depthb"][:, ubdy_mask] = cls[gridU]['depthu']
        ds[f"{dom_prefix}depthb"][:, vbdy_mask] = cls[gridV]['depthv']

        if vars is not None:
            # Add scalar variables along the boundary:
            for var in vars:
                if var in cls[gridT].data_vars:
                    ds[var] = xr.DataArray(data=dask.array.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
                else:
                    raise KeyError(f"variable {var} not found in grid '{gridT}'.")
        
                # Linearly interpolate scalar variables onto NEMO model U/V grid points:
                ds[var][:, :, ubdy_mask] = 0.5 * (
                    cls[gridT][var].where(cls[gridT]['tmask']).sel(i=ds['i_bdy'][ubdy_mask] - 0.5, j=ds['j_bdy'][ubdy_mask]) +
                    cls[gridT][var].where(cls[gridT]['tmask']).sel(i=ds['i_bdy'][ubdy_mask] + 0.5, j=ds['j_bdy'][ubdy_mask])
                    )
                ds[var][:, :, vbdy_mask] = 0.5 * (
                    cls[gridT][var].where(cls[gridT]['tmask']).sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask] - 0.5) +
                    cls[gridT][var].where(cls[gridT]['tmask']).sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask] + 0.5)
                    )

        return ds


    def extract_section(
            cls,
            lon_section: np.ndarray,
            lat_section: np.ndarray,
            uv_vars: list = ['uo', 'vo'],
            vars: list | None = None,
            dom: str = '.',
            ) -> xr.Dataset:
            """
            Extract hydrographic section from a NEMO model domain.

            Parameters
            ----------
            lon_section : np.ndarray
                Longitudes defining the section polygon.
            lat_section : np.ndarray
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
                Dataset containing hydrographic section extracted from NEMO model grid.
            """
            # -- Get NEMO model grid properties -- #
            grid_paths = cls._get_grid_paths(dom=dom)

            # -- Define hydrographic section using polygon -- #
            lon_poly, lat_poly = create_section_polygon(lon_sec=lon_section,
                                                        lat_sec=lat_section,
                                                        )

            mask = cls.mask_with_polygon(grid=grid_paths['gridT'],
                                        lon_poly=lon_poly,
                                        lat_poly=lat_poly
                                        )

            i_bdy, j_bdy, flux_type, flux_dir = get_mask_boundary(mask)

            # -- Create mask boundary dataset -- #
            ds_bdy = create_boundary_dataset(nemo=cls,
                                            dom=dom,
                                            i_bdy=i_bdy,
                                            j_bdy=j_bdy,
                                            flux_type=flux_type,
                                            flux_dir=flux_dir
                                            )

            # -- Get indexes of hydrographic section along mask boundary -- #
            sec_indexes = get_section_indexes(ds_bdy=ds_bdy,
                                            nemo=cls,
                                            dom=dom,
                                            mask_section=mask,
                                            lon_section=lon_section,
                                            lat_section=lat_section,
                                            )

            # -- Update boundary dataset with extracted section data -- #
            ds_bdy = update_boundary_dataset(ds_bdy=ds_bdy,
                                            nemo=cls,
                                            dom=dom,
                                            sec_indexes=sec_indexes,
                                            uv_vars=uv_vars,
                                            vars=vars,
                                            )

            return ds_bdy


    def binned_statistic(
        cls,
        grid : str,
        vars : list[str],
        values : str,
        keep_dims : list[str] | None,
        bins : list[list | np.ndarray],
        statistic : str,
        mask : xr.DataArray | None
        ) -> xr.DataArray:
        """
        Calculate binned statistic of a variable defined on a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variables and values are stored
            (e.g., '/gridT').
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
        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if any(var not in cls[grid].data_vars for var in vars):
            raise KeyError(f"one or more variables {vars} not found in grid '{grid}'.")
        if values not in cls[grid].data_vars:
            raise KeyError(f"values '{values}' not found in grid '{grid}'.")
        if keep_dims is not None:
            if any(dim not in cls[grid][values].dims for dim in keep_dims):
                raise KeyError(f"one or more dimensions {keep_dims} not found in values '{values}'.")
        if not all(isinstance(bin, (list, np.ndarray)) for bin in bins):
            raise ValueError("bins must be a list of lists or numpy arrays.")
        if statistic not in ["all", "any", "count", "sum", "nansum", "mean", "nanmean", "max",
                            "nanmax", "min", "nanmin", "argmax", "nanargmax", "argmin",
                            "nanargmin", "quantile", "nanquantile", "median", "nanmedian",
                            "mode", "nanmode", "first", "nanfirst", "last", "nanlast"]:
            raise ValueError(f"statistic '{statistic}' is not supported.")
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise ValueError("mask must be an xarray.DataArray.")
            if mask.dtype != bool:
                raise TypeError("mask dtype must be boolean.")
            if any(dim not in cls[grid].dims for dim in mask.dims):
                raise ValueError(f"mask must have dimensions subset from {cls[grid].dims}.")

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)

        # -- Define input variables & apply grid mask -- #
        if f"{dom_prefix}depth{grid_suffix}" in cls[grid][values].coords:
            # Apply 3-dimensional t/u/v/f/w mask:
            dom_mask = cls[grid][f"{grid_suffix}mask"]
        else:
            # Apply 2-dimensional t/u/v/f mask (unique points):
            hgrid_type = grid_suffix if 'w' not in grid_suffix else 't'
            dom_mask = cls[grid][f"{hgrid_type}maskutil"]

        # -- Calculate binned statistics -- #
        values_data = cls[grid][values]
        var_data = [cls[grid][var] for var in vars]

        if mask is not None:
            mask = mask & dom_mask
        else:
            mask = dom_mask
    
        result = compute_binned_statistic(
            vars=var_data,
            values=values_data,
            keep_dims=keep_dims,
            bins=bins,
            statistic=statistic,
            mask=mask
        )

        return result


    def transform_vertical_grid(
        cls,
        grid: str,
        var: str,
        e3_new: xr.DataArray
    ) -> xr.Dataset:
        """
        Transform variable defined on a NEMO model grid to a new vertical grid using conservative interpolation.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variable is stored
            (e.g., '/gridT').
        var : str
            Name of the variable to transform.
        e3_new : xarray.DataArray
            Grid cell thicknesses of the new vertical grid.
            Must be a 1-dimensional xarray.DataArray with
            dimension 'k_new'.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Values of variable defined at the centre of each vertical
            grid cell on the new grid, and vertical grid cell
            thicknesses adjusted for model bathymetry.
        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"Grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"Variable '{var}' not found in grid '{grid}'.")
        if e3_new.dims != ('k_new',) or (e3_new.ndim != 1):
            raise ValueError("e3_new must be a 1-dimensional xarray.DataArray with dimension 'k_new'.")

        # -- Get NEMO model grid properties -- #
        dom, _, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)
        ijk_names = cls._get_ijk_names(dom=dom)
        i_name, j_name, k_name = ijk_names['i'], ijk_names['j'], ijk_names['k']

        # -- Define input variables -- #
        mask = cls[grid][f"{grid_suffix}mask"]

        var_in = cls[grid][var].where(mask)
        e3_in = cls[grid][f"e3{grid_suffix}"].where(mask)
        if e3_new.sum(dim="k_new") < cls[grid][f"depth{grid_suffix}"].max(dim=k_name):
            raise ValueError(f"e3_new must sum to at least the maximum depth ({cls[grid][f'depth{grid_suffix}'].max(dim=k_name).item()} m) of the original vertical grid.")

        # -- Transform variable to target vertical grid -- #
        var_out, e3_out = xr.apply_ufunc(transform_vertical_coords,
                                         e3_in,
                                         var_in,
                                         e3_new.astype(e3_in.dtype),
                                         input_core_dims=[[k_name], [k_name], ["k_new"]],
                                         output_core_dims=[["k_new"], ["k_new"]],
                                         dask="allowed"
                                         )

        # -- Create transformed variable Dataset -- #
        t_name = var_in.dims[0]
        var_out = var_out.transpose(t_name, "k_new", j_name, i_name)

        ds_out = xr.Dataset(
            data_vars={var: var_out, f"e3{grid_suffix}_new": e3_out},
            coords={f"depth{grid_suffix}_new": ("k_new", e3_new.cumsum(dim="k_new").data)}
            )

        return ds_out


    def transform_to(
        cls,
        grid: str,
        var: str,
        to: str,
        ) -> xr.DataArray:
        """
        Transform variable defined on a NEMO model grid to a neighbouring
        horizontal grid using linear interpolation.

        For flux variables defined at U- or V-points, the specified variable
        is first weighted by grid cell face areas prior to linear interpolation,
        and is then normalised by the target grid cell face areas following
        interpolation.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where variable is stored (e.g., '/gridT').
        var : str
            Name of the variable to transform.
        to : str
            Suffix of the neighbouring horizontal NEMO model grid to
            transform variable to. Options are 'T', 'U', 'V', 'F'.

        Returns
        -------
        xr.DataArray
            Values of variable linearly interpolated onto a neighbouring
            horizontal grid.
        """
        # -- Validate input -- #
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{grid}'.")
        if not isinstance(to, str):
            raise TypeError(f"'to' must be a string, got {type(to)}.")
        if to not in ['T', 'U', 'V', 'F']:
            raise ValueError(f"'to' must be one of ['T', 'U', 'V', 'F'], got {to}.")

        # -- Get NEMO model grid properties -- #
        _, dom_prefix, _, grid_suffix = cls._get_properties(grid=grid, infer_dom=True)
        ijk_names = cls._get_ijk_names(grid=grid)
        i_name, j_name, k_name = ijk_names['i'], ijk_names['j'], ijk_names['k']
        iperio = cls[grid].attrs.get('iperio', False)
        target_grid = f"{grid.replace(grid[-1], to)}"

        # -- Prepare variable for linear interpolation -- #
        if grid_suffix.upper() in ["U", "V"]:
            weight_dims = [k_name, j_name] if grid_suffix.upper() == "U" else [k_name, i_name]
            if f"{dom_prefix}depth{grid_suffix}" in cls[grid][var].coords:
                # 3-D variables - weight by grid cell face area:
                weights = cls._get_weights(grid=grid, dims=weight_dims, fillna=False)
                target_weights = cls._get_weights(grid=target_grid, dims=weight_dims, fillna=False)
            else:
                # 2-D variables - weight by grid cell width:
                weights = cls._get_weights(grid=grid, dims=weight_dims[1], fillna=False)
                target_weights = cls._get_weights(grid=target_grid, dims=weight_dims[1], fillna=False)
            da = cls[grid][var] * weights
        else:
            # Scalar variables:
            da = cls[grid][var]

        # -- Linearly interpolate variable -- #
        result = interpolate_grid(da=da,
                                  source_grid=grid_suffix.upper(),
                                  target_grid=to,
                                  iperio=iperio,
                                  ijk_names=ijk_names
                                  )

        # -- Update interpolated variable -- #
        # Update NEMO grid coords:
        result[i_name] = cls[target_grid][i_name]
        result[j_name] = cls[target_grid][j_name]
        if k_name in result.dims:
            result[k_name] = cls[target_grid][k_name]

        # Drop NEMO source grid coords:
        drop_vars = [f"{dom_prefix}glam{grid_suffix}", f"{dom_prefix}gphi{grid_suffix}"]
        if f"{dom_prefix}depth{grid_suffix}" in da.coords:
            drop_vars.append(f"{dom_prefix}depth{grid_suffix}")
        result = result.drop_vars(drop_vars)

        # Normalise by target grid cell weights for flux variables:
        if grid_suffix.upper() in ["U", "V"]:
            result = result / target_weights

        # Apply target grid mask:
        if f"{dom_prefix}depth{grid_suffix}" in da.coords:
            target_mask = f"{to.lower()}mask"
        else:
            target_mask = f"{to.lower()}maskutil"
        result = result.where(cls[target_grid][target_mask])

        return result

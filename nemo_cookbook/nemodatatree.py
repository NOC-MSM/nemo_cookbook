"""
nemodatatree.py

Description:
This module defines the NEMODataTree class, a hierarchical data structure
for analysing NEMO ocean general circulation outputs defining one or more
model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
# from xml import dom
import numpy as np
import xarray as xr
from typing import Self
from flox.xarray import xarray_reduce

from .masks import add_polygon_msk, get_mask_boundary
from .processing import create_datatree_dict
from .transform import transform_vertical_coords


class NEMODataTree(xr.DataTree):
    """
    A hierarchical data structure containing collections of NEMO ocean model outputs.

    This class extends xarray.DataTree to provide methods for processing
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
        is stored in an xarray.Dataset.

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
        nftype: str | None = None
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of paths to NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        paths : dict[str, str]
            Dictionary containing paths to NEMO grid files, structured as:
            {
                'parent': {'domain': 'path/to/domain.nc', 'gridT': 'path/to/gridT.nc', ...},
                'child': {'1': {'domain': 'path/to/child_domain.nc', 'gridT': 'path/to/child_gridT.nc', ...},
                          },
                'grandchild': {'2': {'domain': 'path/to/grandchild_domain.nc', 'gridT': 'path/to/grandchild_gridT.nc', ...},
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
            Zonal periodicity of the parent domain.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(paths, dict):
            raise TypeError("paths must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")
        if not isinstance(iperio, bool):
            raise TypeError("zonal periodicity of parent domain must be a boolean.")
        if nftype is not None and nftype not in ('T', 'F'):
            raise ValueError("north fold type of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None.")

        # Define parent, child, grandchild filepath collections:
        d_child, d_grandchild = None, None
        if 'parent' in paths.keys() and isinstance(paths['parent'], dict):
            for key in paths.keys():
                if key not in ('parent', 'child', 'grandchild'):
                    raise ValueError(f"Unexpected key '{key}' in paths dictionary.")
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
                                      nftype=nftype
                                      )

        datatree = super().from_dict(d_tree)

        return datatree


    @classmethod
    def from_datasets(
        cls,
        datasets: dict[str, xr.Dataset],
        nests: dict[str, str] | None = None,
        iperio: bool = False,
        nftype: str | None = None
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of xarray.Dataset objects created from NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Dictionary containing xarray.Datasets created from NEMO grid files, structured as:
            {
                'parent': {'domain': ds_domain, 'gridT': ds_gridT, ...},
                'child': {'1': {'domain': ds_domain_1, 'gridT': d_gridT_1, ...},
                          },
                'grandchild': {'2': {'domain': ds_domain_2, 'gridT': ds_gridT_2, ...},
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
                    },
            }
            where `rx` and `ry` are the horizontal refinement factors, and `imin`, `imax`, `jmin`, `jmax`
            define the indices of the child (grandchild) domain within the parent (child) domain.

        iperio: bool = False
            Zonal periodicity of the parent domain.

        nftype: str, optional
            Type of north fold lateral boundary condition to apply. Options are 'T' for T-point pivot or 'F' for F-point
            pivot. By default, no north fold lateral boundary condition is applied (None).

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

        # Define parent, child, grandchild dataset collections:
        d_child, d_grandchild = None, None
        if 'parent' in datasets.keys() and isinstance(datasets['parent'], dict):
            for key in datasets.keys():
                if key not in ('parent', 'child', 'grandchild'):
                    raise ValueError(f"unexpected key '{key}' in datasets dictionary.")
                if key == 'parent':
                    d_parent = datasets['parent']
                elif key == 'child':
                    d_child = datasets['child']
                elif key == 'grandchild':
                    d_grandchild = datasets['grandchild']
        else:
            raise ValueError("invalid dataset structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains.")

        # Construct DataTree from parent / child / grandchild domains:
        d_tree = create_datatree_dict(d_parent=d_parent,
                                      d_child=d_child,
                                      d_grandchild=d_grandchild,
                                      nests=nests,
                                      iperio=iperio,
                                      nftype=nftype
                                      )
        datatree = super().from_dict(d_tree)

        return datatree


    def _get_weights(
        cls,
        grid: str,
        dims: list
        ) -> xr.DataArray:
        """
        Get the weights (scale factors) for specified dimensions
        of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to NEMO model grid where weights are stored (e.g., '/gridT').
        dims : list
            Dimensions to collect weights for.

        Returns
        -------
        xr.DataArray
            Weights (scale factors) for the specified dimensions of the NEMO model grid.
        """
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if any(dim not in ['i', 'j', 'k'] for dim in dims):
            raise ValueError("dims must be a list containing one or more of the following dimensions: ['i', 'j', 'k'].")

        grid_str = f"{grid.lower()[-1]}"

        weights_dict = {"i": f"e1{grid_str}",
                        "j": f"e2{grid_str}",
                        "k": f"e3{grid_str}",
                        }
        weights_list = [cls[grid][weights_dict[dim]] for dim in dims]

        if len(weights_list) == 1:
            weights = weights_list[0]
        elif len(weights_list) == 2:
            weights = weights_list[0] * weights_list[1]
        elif len(weights_list) == 3:
            weights = weights_list[0] * weights_list[1] * weights_list[2]
        else:
            raise RuntimeError(f"weights missing for dimensions {dims} of NEMO model grid {grid}.")

        if np.isnan(weights).any():
            weights = weights.fillna(value=0)

        return weights


    def cell_area(
        cls,
        grid: str,
        dim: str,
    ) -> xr.DataArray:
        """
        Calculate grid cell areas orthogonal to a given dimension
        of a NEMO model grid.

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
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if dim not in ['i', 'j', 'k']:
            raise ValueError(f"dim {dim} must be one of ['i', 'j', 'k'].")

        grid_str = f"{grid.lower()[-1]}"
        match dim:
            case 'i':
                cell_area = cls[grid][f'e3{grid_str}'] * cls[grid][f'e2{grid_str}']
            case 'j':
                cell_area = cls[grid][f'e3{grid_str}'] * cls[grid][f'e1{grid_str}']
            case 'k':
                cell_area = cls[grid][f'e1{grid_str}'] * cls[grid][f'e2{grid_str}']
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
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")

        grid_str = f"{grid.lower()[-1]}"
        mask = cls[grid][f"{grid_str}mask"]

        cell_volume = cls[grid][f"e3{grid_str}"].where(mask) * cls[grid][f"e1{grid_str}"] * cls[grid][f"e2{grid_str}"]
        cell_volume.name = "volcello"

        return cell_volume


    def gradient(
        cls,
        var: str,
        dim: str,
        dom: str = '.',
    ) -> xr.DataArray:
        """
        Calculate the gradient of a scalar variable along one dimension 
        (e.g., 'i', 'j', 'k') of a NEMO model grid.

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

        # -- Define path to T-grid -- #
        if dom == ".":
            grid = "/gridT"
            dom_str = ""
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid = [n for n in nodes if "gridT" in n][0]
            dom_str = f"{dom}_"
    
        if grid not in list(cls.subtree):
            raise KeyError(f"grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"variable '{var}' not found in grid '{grid}'.")

        da = cls[grid][var]
        dim_name = f"{dim}{dom}" if dom != '.' else dim
        if dim_name not in da.dims:
            raise KeyError(f"dimension '{dim_name}' not found in variable '{var}'. Dimensions available: {da.dims}.")

        match dim:
            case "i":
                gridU = grid.replace("T", "U")
                if f"{dom_str}deptht" in da.coords:
                    # 3-dimensional umask:
                    umask = cls[gridU]["umask"]
                else:
                    # 2-dimensional umask:
                    umask = cls[gridU]["umask"][0, :, :]
                    umask = umask.drop_vars([f"{dom_str}depthu"])

                # Zonally Periodic Domain:
                if cls[grid].attrs.get("iperio", False):
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
                if f"{dom_str}deptht" in gradient.coords:
                    gradient = (gradient
                                .drop_vars([f"{dom_str}deptht"])
                                .assign_coords({f"{dom_str}depthu": cls[gridU][f"{dom_str}depthu"]})
                                )
            case "j":
                gridV = grid.replace("T", "V")
                # 3-dimensional vmask:
                if f"{dom_str}deptht" in da.coords:
                    vmask = cls[gridV]["vmask"]
                else:
                    # 2-dimensional vmask:
                    vmask = cls[gridV]["vmask"][0, :, :]
                    vmask = vmask.drop_vars([f"{dom_str}depthv"])

                # Pad with zeros after differencing (zero gradient at jmaxdom):
                dvar = (da
                        .diff(dim=dim_name, label="lower")
                        .pad({dim_name: (0, 1)}, constant_values=0)
                        )
                # Apply vmask & transform coords -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                gradient = dvar.where(vmask) / cls[gridV]["e2v"]

                if f"{dom_str}deptht" in gradient.coords:
                    gradient = (gradient
                                .drop_vars([f"{dom_str}deptht"])
                                .assign_coords({f"{dom_str}depthv": cls[gridV][f"{dom_str}depthv"]})
                                )

            case "k":
                gridW = grid.replace("T", "W")
                dvar = da.diff(dim=dim_name, label="lower")
                # Transform coords & apply w-mask -> calculate gradient:
                dvar.coords[dim_name] = dvar.coords[dim_name] + 0.5
                dvar = dvar.where(cls[gridW]["wmask"].isel({dim_name: slice(1, None)}))
                try:
                    gradient = - dvar / cls[gridW]["e3w"].isel({dim_name: slice(1, None)})
                    gradient = gradient.drop_vars([f"{dom_str}deptht"])
                except KeyError:
                    raise KeyError(f"NEMO model grid: '{gridW}' does not contain vertical scale factor 'e3w' required to calculate gradients along the k-dimension.")

        # Update DataArray properties:
        gradient.name = f"grad_{var}_{dim_name}"
        gradient = gradient.drop_vars([f"{dom_str}glamt", f"{dom_str}gphit"])

        return gradient
    

    def divergence(
        cls,
        vars : list[str],
        dom: str = '.',
    ) -> xr.DataArray:
        """
        Calculate the horizontal divergence of a vector field defined
        on a NEMO model grid.

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

        # -- Define path to U/V-grids -- #
        if dom == ".":
            i_name, j_name = "i", "j"
            grid_i = "/gridU"
            grid_j = "/gridV"
            dom_str = ""
        else:
            i_name, j_name = f"i{dom}", f"j{dom}"
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid_i = [n for n in nodes if "gridU" in n][0]
            grid_j = [n for n in nodes if "gridV" in n][0]
            dom_str = f"{dom}_"

        if (grid_i not in cls.subtree) or (grid_j not in cls.subtree):
            raise KeyError(f"path '{grid_i}' or '{grid_j}' not found in the NEMODataTree.")

        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[grid_i].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{grid_i}'.")
        if var_j not in cls[grid_j].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{grid_j}'.")
        
        # -- Define i,j vector components -- #
        da_i = cls[grid_i][var_i]
        da_j = cls[grid_j][var_j]

        # -- Collect mask -- #
        if (f"{dom_str}depthu" in da_i.coords) & (f"{dom_str}depthv" in da_j.coords):
            # 3-dimensional tmask:
            tmask = cls[grid_i.replace("U", "T")]["tmask"]
        else:
            # 2-dimensional tmask:
            tmask = cls[grid_i.replace("U", "T")]["tmask"][0, :, :]
            tmask = tmask.drop_vars([f"{dom_str}deptht"])

        # -- Neglecting the first T-grid points along i, j dimensions -- #
        gridT = cls[grid_i.replace("U", "T")]
        e1t = gridT["e1t"].isel({i_name: slice(1, None), j_name: slice(1, None)})
        e2t = gridT["e2t"].isel({i_name: slice(1, None) , j_name: slice(1, None)})
        e3t = gridT["e3t"].isel({i_name: slice(1, None) , j_name: slice(1, None)})

        e2u, e3u = cls[grid_i]["e2u"], cls[grid_i]["e3u"]
        e1v, e3v = cls[grid_j]["e1v"], cls[grid_j]["e3v"]

        # -- Calculate divergence on T-points -- #
        dvar_i = (e2u * e3u * da_i).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1v * e3v * da_j).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        divergence = (1 / (e1t * e2t * e3t)) * (dvar_i + dvar_j).where(tmask)

        # -- Update DataArray properties -- #
        divergence.name = f"div_{var_i}_{var_j}"
        divergence = divergence.drop_vars([f"{dom_str}glamu", f"{dom_str}gphiu",
                                           f"{dom_str}glamv", f"{dom_str}gphiv",
                                           f"{dom_str}depthu", f"{dom_str}depthv"
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

        # -- Define path to U/V-grids -- #
        if dom == ".":
            i_name, j_name = "i", "j"
            grid_i = "/gridU"
            grid_j = "/gridV"
            dom_str = ""
        else:
            i_name, j_name = f"i{dom}", f"j{dom}"
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid_i = [n for n in nodes if "gridU" in n][0]
            grid_j = [n for n in nodes if "gridV" in n][0]
            dom_str = f"{dom}_"

        if (grid_i not in cls.subtree) or (grid_j not in cls.subtree):
            raise KeyError(f"path '{grid_i}' or '{grid_j}' not found in the NEMODataTree.")

        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[grid_i].data_vars:
            raise KeyError(f"variable '{var_i}' not found in grid '{grid_i}'.")
        if var_j not in cls[grid_j].data_vars:
            raise KeyError(f"variable '{var_j}' not found in grid '{grid_j}'.")
        
        # -- Define i,j vector components -- #
        da_i = cls[grid_i][var_i]
        da_j = cls[grid_j][var_j]

        # -- Collect mask -- #
        if (f"{dom_str}depthu" in da_i.coords) & (f"{dom_str}depthv" in da_j.coords):
            # 3-dimensional fmask
            fmask = cls[grid_i.replace("U", "F")]["fmask"]
        else:
            # 2-dimensional fmask:
            fmask = cls[grid_i.replace("U", "F")]["fmask"][0, :, :]

        # -- Neglecting the final F-grid points along i, j dimensions -- #
        gridF = cls[grid_i.replace("U", "F")]
        e1f = gridF["e1f"].isel({i_name: slice(None, -1), j_name: slice(None, -1)})
        e2f = gridF["e2f"].isel({i_name: slice(None, -1) , j_name: slice(None, -1)})

        e1u = cls[grid_i]["e1u"]
        e2v = cls[grid_j]["e2v"]

        # -- Calculate vertical curl component on F-points -- #
        dvar_i = (e2v * da_j).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1u * da_i).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        curl = (1 / (e1f * e2f)) * (dvar_i - dvar_j).where(fmask)

        # -- Update DataArray properties -- #
        curl.name = f"curl_{var_i}_{var_j}"
        curl = curl.drop_vars([f"{dom_str}glamu", f"{dom_str}gphiu",
                               f"{dom_str}glamv", f"{dom_str}gphiv",
                               ])

        return curl


    def integral(
        cls,
        grid : str,
        var : str,
        dims : list,
        cum_dims : list | None = None,
        dir : str | None = None,
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
            
        # -- Prepare variable & weights -- #
        dom_inds = [char for char in grid if char.isdigit()]
        dom_str = f"{dom_inds[-1]}_" if len(dom_inds) != 0 else ""
        grid_str = f"{grid.lower()[-1]}"

        da = cls[grid][var]
        weights = cls._get_weights(grid=grid, dims=dims)

        if f"{dom_str}depth{grid_str}" in da.coords:
            # Apply 3-dimensional mask:
            mask = cls[grid][f"{grid_str}mask"]
        else:
            # Apply 2-dimensional mask:
            mask = cls[grid][f"{grid_str}mask"][0, :, :]
            mask = mask.drop_vars([f"{dom_str}depth{grid_str}"])

        # -- Perform integration -- #
        if cum_dims is not None:
            sum_dims = [dim for dim in dims if dim not in cum_dims]
            if dir == '+1':
                # Cumulative integration along ordered dimension:
                result = cls[grid][var].where(mask).weighted(weights).sum(dim=sum_dims, skipna=True).cumsum(dim=cum_dims, skipna=True)
            elif dir == '-1':
                # Cumulative integration along reversed dimension:
                result = (cls[grid][var]
                            .where(mask)
                            .weighted(weights)
                            .sum(dim=sum_dims, skipna=True)
                            .reindex({dim: cls[grid][dim][::-1] for dim in cum_dims})
                            .cumsum(dim=cum_dims, skipna=True)
                            )
        else:
            # Integration only:
            result = cls[grid][var].weighted(weights).sum(dim=dims, skipna=True)

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

        # -- Clip the grid to given bounding box -- #
        grid_str = f"{grid.lower()[-1]}"

        grid_clipped = cls[grid].dataset.where(
            (cls[grid][f"glam{grid_str}"] >= bbox[0]) &
            (cls[grid][f"glam{grid_str}"] <= bbox[1]) &
            (cls[grid][f"gphi{grid_str}"] >= bbox[2]) &
            (cls[grid][f"gphi{grid_str}"] <= bbox[3]),
            drop=True
            )

        if bbox != (-180, 180, -90, 90):
            grid_clipped = grid_clipped.assign_attrs({"iperio": False})
        cls[grid] = grid_clipped

        return cls


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

        # -- Define paths to domain grids -- #
        grid_paths = [path[0] for path in list(cls.subtree_with_keys)]

        if dom == '.':
            grid_paths = [path for path in grid_paths if ("_" not in path) & ("grid" in path)]
        else:
            grid_paths = [path for path in grid_paths if dom in path]

        # -- Clip grids to given bounding box -- #
        if not grid_paths:
            raise ValueError(f"NEMO model domain '{dom}' not found in the DataTree.")
        else:
            for grid in grid_paths:
                # Use (glamt, gphit) coords for W-grids:
                grid_str = f"{grid.lower()[-1]}" if 'W' not in grid else 't'

                grid_clipped = cls[grid].dataset.where(
                    (cls[grid][f"glam{grid_str}"] >= bbox[0]) &
                    (cls[grid][f"glam{grid_str}"] <= bbox[1]) &
                    (cls[grid][f"gphi{grid_str}"] >= bbox[2]) &
                    (cls[grid][f"gphi{grid_str}"] <= bbox[3]),
                    drop=True
                    )

                if bbox != (-180, 180, -90, 90):
                    grid_clipped = grid_clipped.assign_attrs({"iperio": False})
                cls[grid] = grid_clipped

        return cls


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

        # -- Define path to NEMO grid -- #
        # Identify domain:
        dom_inds = [char for char in grid if char.isdigit()]
        dom_str = dom_inds[-1] if len(dom_inds) != 0 else "."

        if dom_str == ".":
            i_name, j_name = "i", "j"
        else:
            i_name, j_name = f"i{dom_str}", f"j{dom_str}"

        # -- Create mask using polygon coordinates -- #
        lon_name = f"glam{grid.lower()[-1]}"
        lat_name = f"gphi{grid.lower()[-1]}"

        mask = add_polygon_msk(lon_grid=cls[grid][lon_name],
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

        # -- Apply masks & calculate statistic -- #
        grid_str = f"{grid.lower()[-1]}"
        mask = cls[grid][f"{grid_str}mask"]
        da = cls[grid][var].where(mask & mask_poly)

        match statistic:
            case "mean":
                result = da.mean(dim=dims, skipna=True)

            case "weighted_mean":
                weights = cls._get_weights(grid=grid, dims=dims)
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
        Extract the boundary of a masked region defined on a
        NEMO model grid.

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
        if 'i' not in mask.dims or 'j' not in mask.dims:
            raise ValueError("mask must have dimensions 'i' and 'j'")
        if not isinstance(dom, str):
            raise ValueError("dom must be a string specifying prefix of a NEMO domain (e.g., '.', '1', '2', etc.).")

        # -- Define grid paths -- #
        if dom == ".":
            gridT = "/gridT"
            gridU = "/gridU"
            gridV = "/gridV"
            dom_str = ""
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            gridT = [n for n in nodes if "gridT" in n][0]
            gridU = gridT.replace("T", "U")
            gridV = gridT.replace("T", "V")
            dom_str = f"{dom}_"
        
        # -- Extract mask boundary -- #
        i_bdy, j_bdy, flux_type, flux_dir = get_mask_boundary(mask)

        # -- Construct boundary dataset -- #
        k_name = f"k{dom}" if dom != '.' else "k"
        time_name = [dim for dim in cls[gridU].dims if 'time' in dim][0]

        ds = xr.Dataset(
            data_vars={
            'i_bdy': (['bdy'], i_bdy),
            'j_bdy': (['bdy'], j_bdy),
            'flux_type': (['bdy'], flux_type),
            'flux_dir': (['bdy'], flux_dir)
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

        ds['velocity'] = xr.DataArray(data=np.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
        ds['velocity'][:, :, ubdy_mask] = cls[gridU]['uo'].where(cls[gridU]['umask']).sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask]) * ds['flux_dir'][ubdy_mask]
        ds['velocity'][:, :, vbdy_mask] = cls[gridV]['vo'].where(cls[gridV]['vmask']).sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask]) * ds['flux_dir'][vbdy_mask]

        ds[f"{dom_str}glamb"] = xr.DataArray(data=np.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
        ds[f"{dom_str}glamb"][:, :, ubdy_mask] = cls[gridU]['glamu'].sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask])
        ds[f"{dom_str}glamb"][:, :, vbdy_mask] = cls[gridV]['glamv'].sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask])

        ds[f"{dom_str}gphib"] = xr.DataArray(data=np.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
        ds[f"{dom_str}gphib"][:, :, ubdy_mask] = cls[gridU]['gphiu'].sel(i=ds['i_bdy'][ubdy_mask], j=ds['j_bdy'][ubdy_mask])
        ds[f"{dom_str}gphib"][:, :, vbdy_mask] = cls[gridV]['gphiv'].sel(i=ds['i_bdy'][vbdy_mask], j=ds['j_bdy'][vbdy_mask])

        ds[f"{dom_str}depthb"] = xr.DataArray(data=np.zeros(dim_sizes[1:]), dims=[k_name, 'bdy'])
        ds[f"{dom_str}depthb"][:, ubdy_mask] = cls[gridU]['depthu']
        ds[f"{dom_str}depthb"][:, vbdy_mask] = cls[gridV]['depthv']
        ds = ds.assign_coords({f"{dom_str}depthb": ((k_name, "bdy"), ds[f"{dom_str}depthb"].values)})

        if vars is not None:
            # Add scalar variables along the boundary:
            for var in vars:
                if var in cls[gridT].data_vars:
                    ds[var] = xr.DataArray(data=np.zeros(dim_sizes), dims=[time_name, k_name, 'bdy'])
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


    def binned_statistic(
        cls,
        grid : str,
        vars : list[str],
        values : str,
        keep_dims : list[str] | None,
        bins : list[list | np.ndarray],
        statistic : str,
        ) -> xr.DataArray:
        """
        Calculate binned statistics for a given xarray DataArray.

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

        Returns
        -------
        xr.DataArray
            Values of the selected statistic in each bin.
        """
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

        # -- Define input variables & apply grid mask -- #
        grid_str = f"{grid.lower()[-1]}"
        mask = cls[grid][f"{grid_str}mask"]

        values_data = cls[grid][values].where(mask)
        var_data = [cls[grid][var].where(mask) for var in vars]
        keep_vars_data = [cls[grid][dim] for dim in keep_dims]

        expected_groups = [None for _ in keep_dims]
        expected_groups.extend(bin for bin in bins)

        isbin = [False for _ in keep_dims]
        isbin.extend(True for _ in bins)

        # -- Calculate binned statistics -- #
        da = xarray_reduce(
            *[values_data, *keep_vars_data, *var_data],
            func=statistic,
            expected_groups=tuple(expected_groups),
            isbin=tuple(isbin),
            method="map-reduce",
            fill_value=np.nan, # Fill missing values with NaN.
            reindex=False, # Do not reindex during block aggregations to reduce memory at cost of performance.
            engine='numbagg' # Use numbagg grouped aggregations.
            )
        
        # -- Update binned dimensions -- #
        # Transform coords from pd.IntervalIndex to interval mid-points:
        coord_dict = {f'{var}_bins': np.array([interval.mid for interval in da[f'{var}_bins'].values]) for var in vars}
        result = da.assign_coords(coord_dict)

        return result


    def transform_vertical_grid(
        cls,
        grid: str,
        var: str,
        e3_new: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Transform variable defined on a NEMO model grid to a
        new vertical grid using conservative interpolation.

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

        # -- Define input variables -- #
        dom_inds = [char for char in grid if char.isdigit()]
        dom = dom_inds[-1] if len(dom_inds) != 0 else "."
        if dom == ".":
            i_name, j_name, k_name = "i", "j", "k"
        else:
            i_name, j_name, k_name = f"i{dom}", f"j{dom}", f"k{dom}"

        grid_str = f"{grid.lower()[-1]}"
        mask = cls[grid][f"{grid_str}mask"]

        var_in = cls[grid][var].where(mask)
        e3_in = cls[grid][f"e3{grid_str}"].where(mask)
        if e3_new.sum(dim="k_new") < cls[grid][f"depth{grid_str}"].max(dim=k_name):
            raise ValueError(f"e3_new must sum to at least the maximum depth ({cls[grid][f"depth{grid_str}"].max(dim=k_name).item()} m) of the original vertical grid.")

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
            data_vars={var: var_out, f"e3{grid_str}_new": e3_out},
            coords={f"depth{grid_str}_new": ("k_new", e3_new.cumsum(dim="k_new").data)}
            )

        return ds_out

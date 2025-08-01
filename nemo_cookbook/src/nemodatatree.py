"""
nemodatatree.py

Description:
This module defines the NEMODataTree class, a hierarchical data structure
for analysing NEMO ocean general circulation outputs defining one or more
model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

from unittest import case
import xarray as xr
from typing import Self

from .processing import _create_datatree_dict


class NEMODataTree(xr.DataTree):
    """
    A hierarchical data structure for collections of NEMO ocean model outputs.

    This class extends xarray.DataTree to provide methods for processing
    and analysing NEMO output xarray objects defining one or more model domains.
    
    It supports NEMO discrete operators such as gradient, Laplacian, divergence,
    curl, vertical averages, integrals, cumulative integrals, and transforming
    variables between grids.
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
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of paths to NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        paths : dict[str, str]
            A dictionary containing paths to NEMO grid files, structured as:
            {
                'parent': {'domain': 'path/to/domain.nc', 'gridT': 'path/to/gridT.nc', ...},
                'child': {'1': {'domain': 'path/to/child_domain.nc', 'gridT': 'path/to/child_gridT.nc', ...},
                          },
                'grandchild': {'2': {'domain': 'path/to/grandchild_domain.nc', 'gridT': 'path/to/grandchild_gridT.nc', ...},
                               }
            }

        nests : dict[str, str], optional
            A dictionary describing the properties of nested domains, structured as:
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

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(paths, dict):
            raise TypeError("paths must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")

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
        d_tree = _create_datatree_dict(d_parent=d_parent,
                                       d_child=d_child,
                                       d_grandchild=d_grandchild,
                                       nests=nests,
                                       )

        datatree = super().from_dict(d_tree)

        return datatree


    @classmethod
    def from_datasets(
        cls,
        datasets: dict[str, xr.Dataset],
        nests: dict[str, str] | None = None,
    ) -> Self:
        """
        Create a NEMODataTree from a dictionary of xarray.Dataset objects created from NEMO model output files,
        organised into a hierarchy of domains (i.e., 'parent', 'child', 'grandchild').

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            A dictionary containing xarray.Datasets created from NEMO grid files, structured as:
            {
                'parent': {'domain': ds_domain, 'gridT': ds_gridT, ...},
                'child': {'1': {'domain': ds_domain_1, 'gridT': d_gridT_1, ...},
                          },
                'grandchild': {'2': {'domain': ds_domain_2, 'gridT': ds_gridT_2, ...},
                               }
            }

        nests : dict[str, str], optional
            A dictionary describing the properties of nested domains, structured as:
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

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(datasets, dict):
            raise TypeError("datasets must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")

        # Define parent, child, grandchild dataset collections:
        d_child, d_grandchild = None, None
        if 'parent' in datasets.keys() and isinstance(datasets['parent'], dict):
            for key in datasets.keys():
                if key not in ('parent', 'child', 'grandchild'):
                    raise ValueError(f"Unexpected key '{key}' in datasets dictionary.")
                if key == 'parent':
                    d_parent = datasets['parent']
                elif key == 'child':
                    d_child = datasets['child']
                elif key == 'grandchild':
                    d_grandchild = datasets['grandchild']
        else:
            raise ValueError("Invalid datasets structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains.")

        # Construct DataTree from parent / child / grandchild domains:
        d_tree = _create_datatree_dict(d_parent=d_parent,
                                       d_child=d_child,
                                       d_grandchild=d_grandchild,
                                       nests=nests,
                                       )
        datatree = super().from_dict(d_tree)

        return datatree


    def gradient(
        cls,
        var: str,
        dim: str,
        dom: str = '.',
    ) -> xr.DataArray:
        """
        Calculate the gradient of a scalar variable along one dimension 
        (e.g., i, j, k) of a NEMO model grid.

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
        # -- Define path to T-grid -- #
        if dom == ".":
            grid = "/gridT"
            dom_str = ""
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid = [n for n in nodes if "gridT" in n][0]
            dom_str = f"{dom}_"
    
        if grid not in list(cls.subtree):
            raise KeyError(f"Grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"Variable '{var}' not found in grid '{grid}'.")

        da = cls[grid][var]
        if dim not in da.dims:
            raise KeyError(f"Dimension '{dim}' not found in variable '{var}'. Dimensions available: {da.dims}.")

        if dom == ".":
            iperio = cls[grid].attrs.get("Iperio", False)
            nfold = cls[grid].attrs.get("Nfold", False)
        else:
            iperio = False
            nfold = False

        if "i" in dim:
            gridU = grid.replace("T", "U")
            if iperio:
                # Zonally Periodic: add initial T-grid point
                # values to the end of array before differencing:
                da_end = da.isel(dim=0)
                da_end[dim] = da[dim].max() + 1
                da = xr.concat([da, da_end], dim=dim)
                dvar = da.diff(dim=dim, label="lower")
            else:
                # Non-Periodic: pad with NaN values after differencing:
                dvar = (da
                        .diff(dim=dim, label="lower")
                        .pad({dim: (0, 1)})
                        )
            # Transform coords & apply u-mask -> calculate gradient:
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            if f"{dom_str}deptht" in dvar.coords:
                gradient = dvar.where(cls[gridU]["umask"]) / cls[gridU]["e1u"]
            else:
                gradient = dvar.where(cls[gridU]["umask"].isel({f"{dim.replace("i", "k")}": 0})) / cls[gridU]["e1u"]

            if f"{dom_str}deptht" in gradient.coords:
                gradient = (gradient
                            .drop_vars([f"{dom_str}deptht"])
                            .assign_coords({f"{dom_str}depthu": cls[gridU][f"{dom_str}depthu"]})
                            )
        elif "j" in dim:
            gridV = grid.replace("T", "V")
            # TODO: Handle North Folding (NFold) Lateral Boundary Conditions:
            dvar = (da
                    .diff(dim=dim, label="lower")
                    .pad({dim: (0, 1)})
                    )
            # Transform coords & apply v-mask -> calculate gradient:
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            if f"{dom_str}deptht" in dvar.coords:
                gradient = dvar.where(cls[gridV]["vmask"]) / cls[gridV]["e2v"]
            else:
                gradient = dvar.where(cls[gridV]["vmask"].isel({f"{dim.replace("j", "k")}": 0})) / cls[gridV]["e2v"]

            if f"{dom_str}deptht" in gradient.coords:
                gradient = (gradient
                            .drop_vars([f"{dom_str}deptht"])
                            .assign_coords({f"{dom_str}depthv": cls[gridV][f"{dom_str}depthv"]})
                            )

        elif "k" in dim:
            gridW = grid.replace("T", "W")
            dvar = da.diff(dim=dim, label="lower")
            # Transform coords & apply u-mask -> calculate gradient:
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            dvar = dvar.where(cls[gridW]["wmask"].isel({dim: slice(1, None)}))
            try:
                gradient = - dvar / cls[gridW]["e3w"].isel({dim: slice(1, None)})
                gradient = gradient.drop_vars([f"{dom_str}deptht"])
            except KeyError:
                raise KeyError(f"NEMO model grid: '{gridW}' does not contain vertical scale factor 'e3w' required to calculate gradients along the k-dimension.")

        # Update DataArray properties:
        gradient.name = f"grad_{var}_{dim}"
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
            Name of the vector variables, structured as: ['u', 'v'], where
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
        # -- Define path to U/V-grids -- #
        if dom == ".":
            grid_i = "/gridU"
            grid_j = "/gridV"
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid_i = [n for n in nodes if "gridU" in n][0]
            grid_j = [n for n in nodes if "gridV" in n][0]

        # -- Define i,j coord names -- #
        if dom == ".":
            i_name, j_name = "i", "j"
        else:
            i_name, j_name = f"i{dom}", f"j{dom}"

        if (grid_i not in cls.subtree) or (grid_j not in cls.subtree):
            raise KeyError(f"Path '{grid_i}' or '{grid_j}' not found in the NEMODataTree.")

        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[grid_i].data_vars:
            raise KeyError(f"Variable '{var_i}' not found in grid '{grid_i}'.")
        if var_j not in cls[grid_j].data_vars:
            raise KeyError(f"Variable '{var_j}' not found in grid '{grid_j}'.")
        
        # -- Define i,j vector components -- #
        da_i = cls[grid_i][var_i]
        da_j = cls[grid_j][var_j]

        # -- Neglecting the first T-grid points along i, j dimensions -- #
        gridT = cls[grid_i.replace("U", "T")]
        e1t = gridT["e1t"].isel({i_name: slice(1, None), j_name: slice(1, None)})
        e2t = gridT["e2t"].isel({i_name: slice(1, None) , j_name: slice(1, None)})
        e3t = gridT["e3t"].isel({i_name: slice(1, None) , j_name: slice(1, None)})

        e2u, e3u = cls[grid_i]["e2u"], cls[grid_i]["e3u"]
        e1v, e3v = cls[grid_j]["e1v"], cls[grid_j]["e3v"]

        # -- Calculate divergence on T-points -- #
        # TODO: Add mask to avoid NaN values in divergence calculation.
        dvar_i = (e2u * e3u * da_i).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1v * e3v * da_j).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        divergence = (1 / (e1t * e2t * e3t)) * (dvar_i + dvar_j)

        # -- Update DataArray properties -- #
        if dom != '.':
            dom_str = f"{dom}_"
        else:
            dom_str = ""
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
        # -- Define path to U/V-grids -- #
        if dom == ".":
            grid_i = "/gridU"
            grid_j = "/gridV"
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            grid_i = [n for n in nodes if "gridU" in n][0]
            grid_j = [n for n in nodes if "gridV" in n][0]

        # -- Define i,j coord names -- #
        if dom == ".":
            i_name, j_name = "i", "j"
        else:
            i_name, j_name = f"i{dom}", f"j{dom}"

        if (grid_i not in cls.subtree) or (grid_j not in cls.subtree):
            raise KeyError(f"Path '{grid_i}' or '{grid_j}' not found in the NEMODataTree.")

        var_i, var_j = vars[0], vars[1]
        if var_i not in cls[grid_i].data_vars:
            raise KeyError(f"Variable '{var_i}' not found in grid '{grid_i}'.")
        if var_j not in cls[grid_j].data_vars:
            raise KeyError(f"Variable '{var_j}' not found in grid '{grid_j}'.")
        
        # -- Define i,j vector components -- #
        da_i = cls[grid_i][var_i]
        da_j = cls[grid_j][var_j]

        # -- Neglecting the final F-grid points along i, j dimensions -- #
        gridF = cls[grid_i.replace("U", "F")]
        e1f = gridF["e1f"].isel({i_name: slice(None, -1), j_name: slice(None, -1)})
        e2f = gridF["e2f"].isel({i_name: slice(None, -1) , j_name: slice(None, -1)})

        e1u = cls[grid_i]["e1u"]
        e2v = cls[grid_j]["e2v"]

        # -- Calculate vertical curl component on F-points -- #
        # TODO: Add mask to avoid NaN values in curl calculation.
        dvar_i = (e2v * da_j).diff(dim=i_name, label="lower")
        dvar_i.coords[i_name] = dvar_i.coords[i_name] + 0.5

        dvar_j = (e1u * da_i).diff(dim=j_name, label="lower")
        dvar_j.coords[j_name] = dvar_j.coords[j_name] + 0.5

        curl = (1 / (e1f * e2f)) * (dvar_i - dvar_j)

        # -- Update DataArray properties -- #
        if dom != '.':
            dom_str = f"{dom}_"
        else:
            dom_str = ""

        if f"{dom_str}depthu" in curl.coords:
            curl = curl.drop_vars([f"{dom_str}depthu"])
        if f"{dom_str}depthv" in curl.coords:
            curl = curl.drop_vars([f"{dom_str}depthv"])

        curl = curl.drop_vars([f"{dom_str}glamu", f"{dom_str}gphiu",
                               f"{dom_str}glamv", f"{dom_str}gphiv",
                               ])
        curl.name = f"curl_{var_i}_{var_j}"
                

        return curl

    # TODO: Add 'laplacian' method to calculate the Laplacian of scalar variables.

    # TODO: Add 'vertical_average' method to calculate vertical averages of scalar variables.

    # TODO: Add 'integral' method to calculate integrals of scalar or vector variables.

    # TODO: Add 'cumintegral' method to calculate accumulative integrals of scalar or vector variables.

    # TODO: Add 'transform' method to transform variables between grids (e.g., from T to U grid).

    # TODO: Add 'vertical_transform' method to transform variables between vertical coordinates (e.g., from z to sigma).

    # TODO: Add 'degrade' method to conservatively coarsen a variable on a NEMO model grid.

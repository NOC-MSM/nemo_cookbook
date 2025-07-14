"""
nemo_datatree.py

Description:
This module defines the NEMODataTree class, a hierarchical data structure
for analysing NEMO ocean general circulation outputs defining one or more
model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import xarray as xr
from typing import Self

from .processing import _process_parent, _process_child, _process_grandchild


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
        periodic: bool = True,
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

        periodic : bool, optional
            Whether the parent NEMO model domain is periodic (True) or closed (False) in the i-direction.
            Default is True.

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(paths, dict):
            raise TypeError("paths must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be a boolean value.")

        d_child, d_grandchild = None, None

        # TODO: Add NEMO mask variables to grids.
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

        # Assign the parent domain:
        d_tree = _process_parent(d_parent)

        # Update dict with child domains:
        if d_child is not None:
            if all(isinstance(d_child[key], dict) for key in d_child.keys()):
                for key in d_child.keys():
                    if key in nests.keys():
                        d_nests = nests[key]
                        if 'parent' in d_nests.keys():
                            d_tree = {**d_tree, **_process_child(d_child=d_child[key],
                                                                 d_nests=d_nests,
                                                                 label=int(key)
                                                                 )}
                        else:
                            raise KeyError(f"Child nest dict '{key}' does not specify a parent domain.")
                    else:
                        raise KeyError(f"Child domain '{key}' not found in nests dict.")
            else:
                raise ValueError("Invalid child domain structure. Expected a nested dict defining NEMO child domain(s).")

        # Update dict with grandchild domains:
        if d_grandchild is not None:
            if all(isinstance(d_grandchild[key], dict) for key in d_grandchild.keys()):
                for key in d_grandchild.keys():
                    if key in nests.keys():
                        d_nests = nests[key]
                        if 'parent' in d_nests.keys():
                            if d_nests['parent'] in d_child.keys():
                                parent_label = d_nests['parent']
                                d_tree = {**d_tree, **_process_grandchild(d_grandchild[key],
                                                                          d_nests=d_nests,
                                                                          label=int(key),
                                                                          parent_label=int(parent_label)
                                                                          )}
                            else:
                                raise KeyError(f"Parent domain '{parent_label}' not found in child domains.")
                        else:
                            raise KeyError(f"Grandchild nest dict '{key}' does not specify a parent domain.")
                    else:
                        raise KeyError(f"Grandchild domain '{key}' not found in nests dict.")
            else:
                raise ValueError("Invalid grandchild domain structure. Expected a nested dict defining NEMO grandchild domain(s).")
                
        # Define DataTree from parent / child / grandchild domains:
        dt = super().from_dict(d_tree)

        # Set the root node attributes:
        dt['/'].attrs['periodic'] = periodic

        return dt


    def gradient(
        cls,
        grid: str,
        var: str,
        dim: str,
    ) -> xr.DataArray:
        """
        Calculate the gradient of a scalar variable along one dimension 
        (e.g., i, j, k) of a NEMO model grid.

        Parameters
        ----------
        grid : str
            Path to the grid-node containing scalar variable (e.g., '/gridT', '/gridU/1_gridU' etc.).
        var : str, optional
            Scalar variable to calculate gradient.
        dim : str
            Dimension along which to calculate gradient (e.g., 'i', 'j', 'k').

        Returns
        -------
        xr.DataArray
            Gradient of chosen scalar variable stored on the NEMO model grid.
        """
        if grid not in list(cls.subtree):
            raise KeyError(f"Grid '{grid}' not found in the NEMODataTree.")
        if var not in cls[grid].data_vars:
            raise KeyError(f"Variable '{var}' not found in grid '{grid}'.")

        da = cls[grid][var]
        if dim not in da.dims:
            raise KeyError(f"Dimension '{dim}' not found in variable '{var}'. Dimensions available: {da.dims}.")
        
        # Check zonal peridocity if `grid` is a parent domain:
        if grid.count('/') == 1:
            iperio = cls['/'].attrs.get('periodic', True)
        else:
            iperio = False

        # Define source grid (i.e, 'T', 'U', 'V', 'W'):
        src_grid = grid[-1]

        # TODO: Differentiate scalar variables vertically.
        if "i" in dim:
            if iperio:
                da_end = da.isel(i=-1)
                da_end['i'] = da_end['i'] + 1
                da = xr.concat([da, da_end], dim=dim)
            else:
                da = da.pad({dim: (0, 1)})
            dvar = (da
                    .where(da != 0)
                    .diff(dim=dim, label="lower")
                    )
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            dvar_ddim = dvar / cls[grid.replace(src_grid, "U")]["e1u"]

        elif "j" in dim:
            dvar = (da
                    .where(da != 0)
                    .pad({dim: (0, 1)})
                    .diff(dim=dim, label="lower")
                    )
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            dvar_ddim = dvar / cls[grid.replace(src_grid, "V")]["e2v"]

        return dvar_ddim

    # TODO: Add 'laplacian' method to calculate the Laplacian of scalar variables.

    # TODO: Add 'divergence' method to calculate divergence of vector fields.

    # TODO: Add 'curl' method to calculate curl of vector fields.

    # TODO: Add 'vertical_average' method to calculate vertical averages of scalar variables.

    # TODO: Add 'integrate' method to calculate integrals of scalar or vector variables.

    # TODO: Add 'cumintegral' method to calculate accumulative integrals of scalar or vector variables.

    # TODO: Add 'transform' method to transform variables between grids (e.g., from T to U grid).

    # TODO: Add 'vertical_transform' method to transform variables between vertical coordinates (e.g., from z to sigma).
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
        global_parent: bool = True,
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

        global_parent : bool, optional
            If True, the parent domain is treated as a global domain with zonal periodicity and a closed
            southern boundary.
            If False, the parent domain is treated as a closed (regional) domain without periodicity.

        Returns
        -------
        NEMODataTree
            A hierarchical data tree of NEMO model outputs.
        """
        if not isinstance(paths, dict):
            raise TypeError("paths must be a dictionary or nested dictionary.")
        if not isinstance(nests, (dict, type(None))):
            raise TypeError("nests must be a dictionary or None.")
        if not isinstance(global_parent, bool):
            raise TypeError("global_parent must be a boolean value.")

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

        # Set parent domain type attribute:
        dt["/"].attrs["global"] = global_parent

        return dt


    def _add_masks(cls,
                   dom: str,
                   ) -> xr.DataArray:
        """
        Add mask variables for model domain to NEMODataTree.

        Parameters
        ----------
        dom : str
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Use '.' for the parent domain.
        """
        # Define T-grid path in DataTree:
        if dom == ".":
            gridT = "/gridT"
            dom_str = ""
            dom_global = cls["/"].attrs.get("global", True)
        else:
            nodes = [n[0] for n in cls.subtree_with_keys if dom in n[0]]
            gridT = [n for n in nodes if "gridT" in n][0]
            dom_str = dom
            dom_global = False

        # -- Define t_mask from top/bottom_level -- #
        ka = cls[gridT][f"k{dom_str}"].expand_dims({f"i{dom_str}": cls[gridT][f"i{dom_str}"],
                                                    f"j{dom_str}": cls[gridT][f"j{dom_str}"]
                                                    })
        # Exclude land points:
        mask_1 = ~(ka < cls[gridT].top_level)
        # Keep wet cells between top and bottom levels:
        mask_2 = (ka >= cls[gridT].top_level) & (cls[gridT].bottom_level >= ka)
        # Exclude points below the sea floor:
        mask_3 = ~(cls[gridT].bottom_level < ka)

        # -- t_mask -- #
        t_mask = (mask_1 & mask_2 & mask_3).T

        # -- u_mask -- #
        if dom_global:
            # Zonally Periodic Parent Domain:
            tmask_end = t_mask.isel(i=0)
            tmask_end["i"] = t_mask["i"].max() + 1
            tmask = xr.concat([t_mask, tmask_end], dim="i")

            tmask_1 = tmask.isel({f"i{dom_str}": slice(None, -1)})
            tmask_2 = tmask.isel({f"i{dom_str}": slice(1, None)})
            tmask_2.coords["i"] = tmask_2.coords["i"] - 1

            u_mask = tmask_1 * tmask_2
            u_mask.coords[f"i{dom_str}"] = u_mask.coords[f"i{dom_str}"] + 0.5

        else:
            # Closed Parent / Child Domain - masked at first and last U-points:
            tmask_1 = t_mask.isel({f"i{dom_str}": slice(None, -1)})
            tmask_2 = t_mask.isel({f"i{dom_str}": slice(1, None)})
            tmask_2.coords[f"i{dom_str}"] = tmask_2.coords[f"i{dom_str}"] - 1

            umask = tmask_1 * tmask_2
            umask_end = umask.isel({f"i{dom_str}": 0})
            umask_end.coords[f"i{dom_str}"] = umask[f"i{dom_str}"].max() + 1

            u_mask = xr.concat([umask, umask_end], dim=f"i{dom_str}").astype('bool')
            u_mask.coords[f"i{dom_str}"] = u_mask.coords[f"i{dom_str}"] + 0.5

        # -- v_mask -- #
        tmask_1 = t_mask.isel({f"j{dom_str}": slice(None, -1)})
        tmask_2 = t_mask.isel({f"j{dom_str}": slice(1, None)})
        tmask_2.coords[f"j{dom_str}"] = tmask_2.coords[f"j{dom_str}"] - 1

        vmask = tmask_1 * tmask_2
        vmask_end = t_mask.isel({f"j{dom_str}": -1})
        vmask_end.coords[f"j{dom_str}"] = vmask[f"j{dom_str}"].max() + 1

        v_mask = xr.concat([vmask, vmask_end], dim=f"j{dom_str}").astype('bool')
        v_mask.coords[f"j{dom_str}"] = v_mask.coords[f"j{dom_str}"] + 0.5

        # -- w_mask -- #
        # NEMO Manual pp.36: At k=1 -> sea surface, w_mask(i, j, 1) = tmask(i, j, 1)
        w_mask = t_mask.isel({f"k{dom_str}": slice(1, None)}) * t_mask.isel({f"k{dom_str}": slice(None, -1)})
        w_mask.coords[f"k{dom_str}"] = w_mask.coords[f"k{dom_str}"] + 0.5
        depth_name = [var for var in w_mask.coords if var.endswith("deptht")]
        w_mask = w_mask.drop_vars(depth_name)

        tmask_st = t_mask.isel({f"k{dom_str}": 0})
        tmask_st = tmask_st.drop_vars(depth_name)

        tmask_st.coords[f"k{dom_str}"] = t_mask.coords[f"k{dom_str}"].min() - 1
        w_mask = xr.concat([w_mask, tmask_st], dim=f"k{dom_str}").astype('bool')
        w_mask.coords[depth_name[0]] = t_mask.coords[depth_name[0]]

        # -- f_mask -- #
        f_mask = (
            t_mask.isel({f"i{dom_str}": slice(None, -1)}) * t_mask.isel({f"i{dom_str}": slice(1, None)}) *
            t_mask.isel({f"j{dom_str}": slice(None, -1)}) * t_mask.isel({f"j{dom_str}": slice(1, None)})
            )
        
        f_mask = f_mask.astype('bool')
        f_mask.coords[f"i{dom_str}"] = f_mask.coords[f"i{dom_str}"] + 0.5
        f_mask.coords[f"j{dom_str}"] = f_mask.coords[f"j{dom_str}"] + 0.5

        # -- Add mask variables to DataTree -- #
        if dom == ".":
            dom_str = ""
        else:
            dom_str = f"{dom}_"

        cls[gridT][f"{dom_str}tmask"] = t_mask
        cls[gridT.replace("T", "U")][f"{dom_str}umask"] = u_mask
        cls[gridT.replace("T", "V")][f"{dom_str}vmask"] = v_mask
        cls[gridT.replace("T", "W")][f"{dom_str}wmask"] = w_mask
        cls[gridT.replace("T", "F")][f"{dom_str}fmask"] = f_mask
        
        return cls


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
            Use '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Gradient of chosen scalar variable stored on the NEMO model grid.
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
        
        # -- Zonal periodcity for parent domain -- #
        if dom == ".":
            iperio = cls["/"].attrs.get("global", True)
        else:
            iperio = False

        if "i" in dim:
            gridU = grid.replace("T", "U")
            if iperio:
                da_end = da.isel(i=0)
                da_end["i"] = da["i"].max() + 1
                da = xr.concat([da, da_end], dim="i")
            else:
                da = da.pad({dim: (0, 1)})
            dvar = (da
                    # TODO: Add mask to avoid NaN values in gradient calculation.
                    .where(da != 0)
                    .diff(dim=dim, label="lower")
                    )
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            gradient = dvar / cls[gridU]["e1u"]
            if f"{dom_str}deptht" in gradient.coords:
                gradient = (gradient
                            .drop_vars([f"{dom_str}deptht"])
                            .assign_coords({f"{dom_str}depthu": cls[gridU][f"{dom_str}depthu"]})
                            )

        elif "j" in dim:
            gridV = grid.replace("T", "V")
            dvar = (da
                    # TODO: Add mask to avoid NaN values in gradient calculation.
                    .where(da != 0)
                    .pad({dim: (0, 1)})
                    .diff(dim=dim, label="lower")
                    )
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            gradient = dvar / cls[gridV]["e2v"]
            if f"{dom_str}deptht" in gradient.coords:
                gradient = (gradient
                            .drop_vars([f"{dom_str}deptht"])
                            .assign_coords({f"{dom_str}depthv": cls[gridV][f"{dom_str}depthv"]})
                            )

        elif "k" in dim:
            gridW = grid.replace("T", "W")
            dvar = (da
                    # TODO: Add mask to avoid NaN values in gradient calculation.
                    .where(da != 0)
                    .diff(dim=dim, label="lower")
                    )
            dvar.coords[dim] = dvar.coords[dim] + 0.5
            gradient = - dvar / cls[gridW]["e3w"].isel({dim: slice(1, None)})
            gradient = gradient.drop_vars([f"{dom_str}deptht"])

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
        Calculate the horizontal divergence of a vector variable on a NEMO model grid.

        Parameters
        ----------
        vars : str
            Name of the vector variables, structured as: ['u', 'v'], where 'u' and 'v' are
            the i and j components of the vector variable, respectively.
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Use '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Horizontal divergence of chosen vector variable stored on the NEMO model grid.
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
        Calculate the vertical (k) curl component of a vector variable on a NEMO model grid.

        Parameters
        ----------
        vars : str
            Name of the vector variables, structured as: ['u', 'v'], where 'u' and 'v' are
            the i and j components of the vector variable, respectively.
        dom : str, optional
            Prefix of NEMO domain in the DataTree (e.g., '1', '2', '3', etc.).
            Use '.' for the parent domain.

        Returns
        -------
        xr.DataArray
            Vertical curl component of a vector variable stored on the NEMO model grid.
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

    # TODO: Add 'integrate' method to calculate integrals of scalar or vector variables.

    # TODO: Add 'cumintegral' method to calculate accumulative integrals of scalar or vector variables.

    # TODO: Add 'transform' method to transform variables between grids (e.g., from T to U grid).

    # TODO: Add 'vertical_transform' method to transform variables between vertical coordinates (e.g., from z to sigma).
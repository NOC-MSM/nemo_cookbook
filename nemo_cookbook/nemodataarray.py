"""
nemodataarray.py

Description:
This module defines the NEMODataArray class, a grid-aware xarray.DataArray wrapper
for NEMO ocean model variables, providing discrete operators such as differences,
derivatives, and integrals, and supporting transformation between NEMO grids.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
from __future__ import annotations

import operator
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    # Avoid circular import at runtime:
    from nemo_cookbook.nemodatatree import NEMODataTree
from nemo_cookbook.integrate import compute_depth_integral
from nemo_cookbook.interpolate import interpolate_grid
from nemo_cookbook.transform import transform_vertical_coords

_NEMO_DIFF_MAP = {
    # -- NEMO Arakawa C-grid Finite Difference Mapping -- #
    # (input_grid_suffix, diff_dim): (output_grid_suffix, output_hgrid_suffix, output_scale_factor)
    ("T", "i"): ("u", "u", "e1u"),
    ("T", "j"): ("v", "v", "e2v"),
    ("T", "k"): ("w", "t", "e3w"),
    ("U", "i"): ("t", "t", "e1t"),
    ("U", "j"): ("f", "f", "e2f"),
    ("U", "k"): ("uw", "u", "e3uw"),
    ("V", "i"): ("f", "f", "e1f"),
    ("V", "j"): ("t", "t", "e2t"),
    ("V", "k"): ("vw", "v", "e3vw"),
    ("W", "i"): ("u", "u", "e1u"),
    ("W", "j"): ("v", "v", "e2v"),
    ("W", "k"): ("t", "t", "e3t"),
    ("F", "i"): ("v", "v", "e1v"),
    ("F", "j"): ("u", "u", "e2u"),
    ("F", "k"): ("fw", "f", "e3fw"),
}


class NEMODataArray:
    """
    Grid-aware xarray.DataArray wrapper for NEMO ocean model variables.

    This class interfaces with the NEMODataTree to provide discrete
    operators such as differences, derivatives, and integrals alongside
    transformation between NEMO model grids.
    """
    def __init__(self, da: xr.DataArray, tree: NEMODataTree, grid: str):
        """
        Create a NEMODataArray from an xarray.DataArray and NEMODataTree.

        Parameters
        ----------
        da : xr.DataArray
            Variable defined on a NEMO model grid.
        tree: NEMODataTree
            NEMODataTree to which the variable belongs.
        grid: str
            Path to NEMO model grid where variable is defined (e.g., 'gridT').

        Returns
        -------
        NEMODataArray
            Grid-aware xarray.DataArray storing variable defined on NEMO model grid.
        """
        # -- Validate Inputs -- #
        if not isinstance(da, xr.DataArray):
            raise TypeError("da must be specified as an xarray.DataArray.")
        else:
            self._da = da

        if not isinstance(tree, xr.DataTree):
            raise TypeError("tree must be specified as a NEMODataTree.")
        else:
            self._tree = tree

        if not isinstance(grid, str):
            raise TypeError("grid must be specified as a string.")
        else:
            self._grid = grid

        # -- Validate DataArray compatibility -- #
        grid_keys = list(dict(self._tree.subtree_with_keys).keys())
        if self._grid not in grid_keys:
            raise KeyError(
                f"{self._grid} not found in available NEMODataTree grids {grid_keys}."
            )

        # Dimension must exist & values must be within NEMODataTree dimension values:
        if not all(dim in list(self._tree[self._grid].dims) for dim in self._da.dims):
            raise ValueError(f"DataArray dimensions {self._da.dims} not all in NEMO model '{self._grid}' dimensions {self._tree[self._grid].dims}.")

        if not all((self._da[dim].min() >= self._tree[self._grid][dim].min()) & (self._da[dim].max() <= self._tree[self._grid][dim].max()) for dim in self._da.dims):
            raise ValueError(f"DataArray dimension values {self._da.dims} not all within NEMO model '{self._grid}' dimension values {self._tree[self._grid].dims}.")

        # Core coordinates (glam, gphi, depth, time_counter) must exist & values must be within NEMODataTree coordinate values:
        core_coords = [coord for coord in self._da.coords if "glam" in coord or "gphi" in coord or "depth" in coord or "time_counter" in coord]
        if not all(dim in list(self._tree[self._grid].coords) for dim in core_coords):
            raise ValueError(f"DataArray coordinates {core_coords} not all in NEMO model '{self._grid}' coordinates {self._tree[self._grid].coords}.")

        if not all((self._da[coord].min() >= self._tree[self._grid].coords[coord].min()) & (self._da[coord].max() <= self._tree[self._grid].coords[coord].max()) for coord in core_coords):
            raise ValueError(f"DataArray coordinate values {core_coords} not all within NEMO model '{self._grid}' coordinate values {self._tree[self._grid].coords}.")

        # -- Assign NEMO domain number, grid path and grid type -- #
        dom, dom_prefix, dom_suffix, grid_suffix = self._tree._get_properties(grid=self._grid, infer_dom=True)
        self._dom = dom
        self._dom_prefix = dom_prefix
        self._dom_suffix = dom_suffix
        self._grid_suffix = grid_suffix

        # -- Assign NEMO dimension names -- #
        # Spatial dimensions:
        ijk_names = self._tree._get_ijk_names(grid=self._grid)
        self.i_name = ijk_names["i"]
        self.j_name = ijk_names["j"]
        self.k_name = ijk_names["k"]
        # Temporal dimension except for time-independent variables:
        # Use coords to handle single time slices of time-dependent variables.
        t_list = [dim for dim in self._da.coords if "time" in dim]
        self.t_name = t_list[0] if len(t_list) != 0 else None


    # -----------
    # Properties 
    # -----------
    @property
    def data(self) -> xr.DataArray:
        """
        Access underlying xarray.DataArray.
        """
        return self._da
    
    @property
    def grid(self) -> str:
        """
        Access path to parent NEMO model grid.
        """
        return self._grid
    
    @property
    def grid_type(self) -> str:
        """
        Access type of parent NEMO model grid (e.g. 't', 'u', etc.).
        """
        return self._grid_suffix
    
    @property
    def metrics(self) -> dict[str, "NEMODataArray"]:
        """
        Access grid scale factors for the parent NEMO model grid.
        """
        # 2-dimensional: Horizontal grid scale factors:
        d_metrics = {"e1": self._tree[f"{self._grid}/e1{self._grid_suffix}"].sel_like(self._da),
                     "e2": self._tree[f"{self._grid}/e2{self._grid_suffix}"].sel_like(self._da),
                     }

        if f"{self._dom_prefix}depth{self._grid_suffix}" in self._da.coords:
            # 3-dimensional: Vertical grid scale factor:
            d_metrics["e3"] = self._tree[f"{self._grid}/e3{self._grid_suffix}"].sel_like(self._da)

        return d_metrics
    
    @property
    def mask(self) -> xr.DataArray:
        """
        Access variable land-sea mask for the parent NEMO model grid.
        """
        if (self._da.ndim == 1) and (self.t_name in self._da.dims):
            raise ValueError("land-sea mask does not exist for variables without spatial dimensions.")
        else:
            if (f"{self._dom_prefix}depth{self._grid_suffix}" in self._da.coords) & (self.k_name in self._da.dims):
                # 3-dimensional land-sea mask:
                mask_name = f"{self._grid_suffix}mask"
            else:
                # 2-dimensional land-sea mask:
                mask_name = f"{self._grid_suffix}maskutil"

        # Select land-sea mask values according to variable dimension labels:
        result = self._tree[f"{self._grid}/{mask_name}"].sel_like(self._da).data
        
        return result
    
    @property
    def masked(self):
        """
        Apply land-sea mask to variable defined on NEMO model grid.
        """
        return self.apply_mask(mask=None, drop=False)
    
    @property
    def iperio(self):
        """
        Zonal periodicity of variable defined on NEMO model grid.
        """
        # Determine zonal periodicity of parent NEMO model grid:
        iperio = self._tree[self._grid].attrs.get("iperio", False)

        # Update zonal periodicity if variable is defined on a subset
        # of i-dimension labels of the parent NEMO model grid:
        if iperio:
            if self._da.coords[self.i_name].size != self._tree[self._grid].coords[self.i_name].size:
                iperio = False

        return iperio

    # ---------------
    # Public Methods 
    # ---------------
    def apply_mask(
        self,
        mask: xr.DataArray | None = None,
        drop: bool = False
    ) -> Self:
        """
        Apply NEMO parent grid land-sea mask or combined land-sea & custom mask
        to variable defined on a NEMO model grid.

        Parameters
        ----------
        mask : xr.DataArray | None
            Boolean mask to apply to variable defined on NEMO model grid. Default is None
            meaning only land-sea mask of NEMO parent grid is applied.
        drop : bool, optional
            If True, coordinate labels that only correspond to False values of the condition
            are dropped from the result. Default is False.

        Returns
        -------
        NEMODataArray

        Examples
        --------
        Apply a custom boolean mask `my_mask` to sea surface temperature `tos_con` defined on
        scalar T-points in a NEMO model parent domain:

        >>> nemo['gridT/tos_con'].apply_mask(mask=my_mask)

        Apply custom mask `my_mask` to sea surface salinity `sos_abs` and drop masked data values:

        >>> nemo['gridT/sos_abs'].apply_mask(mask=my_mask, drop=True)

        See Also
        --------
        masked
        """
        # -- Validate Inputs -- #
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise TypeError("mask must be specified as an xarray.DataArray or None.")
        if mask is None:
            mask = self.mask
        else:
            mask = (self.mask & mask)

        # -- Apply mask & return NEMODataArray -- #
        if drop:
            warnings.warn(message="Indexing with a boolean dask array is not allowed. Mask will be computed first using .compute(). This may result in high memory usage for large masks.",
                          category=RuntimeWarning,
                          stacklevel=2
                          )
            mask = mask.load()

        result = self._da.where(mask, drop=drop)

        return self._wrap(result)
    
    def sel_like(
        self,
        other: Self | xr.DataArray,
    ) -> Self:
        """
        Return a new NEMODataArray whose data is given by matching the dimension
        index labels present in another NEMODataArray or xarray.DataArray.

        Parameters
        ----------
        other : NEMODataArray | xr.DataArray
            NEMODataArray or xarray.DataArray used to select dimension index labels.

        Returns
        -------
        NEMODataArray
             A new NEMODataArray with data selected according to dimension index labels
             of input object.

        Examples
        --------
        Indexing conservative temperature `thetao_con` defined on scalar T-points in a NEMO
        model parent domain to match a subset of the absolute salinity `so_abs`:

        >>> nda = nemo['gridT/so_abs'].sel(time_counter=slice("2020-01", "2024-01"), k=1)

        >>> nemo['gridT/thetao_con'].sel_like(nda)

        See Also
        --------
        sel
        """
        # -- Validate Inputs -- #
        if not (isinstance(other, NEMODataArray) or isinstance(other, xr.DataArray)):
            raise TypeError("other must be specified as a NEMODataArray or xarray.DataArray.")

        if isinstance(other, NEMODataArray):
            other = other.data

        # -- Index data using dimension labels of other object & return NEMODataArray -- #
        coord_dims = [self.t_name, self.k_name, self.j_name, self.i_name]
        d_dims = {}
        for dim in coord_dims:
            # Select only dimensions present in both objects:
            if (dim in other.coords) and (dim in self.data.coords):
                # Select only dimensions whose index labels do not already match:
                if other.coords[dim].size != self.data.coords[dim].size:
                    d_dims[dim] = other.coords[dim].data

        # Indexing only required if dimensions modified:
        if len(d_dims) > 0:
            result = self.data.sel(d_dims)
            return self._wrap(result)
        # Otherwise return original NEMODataArray:
        else:
            return self

    def weighted_mean(
        self,
        dims : list,
        mask: xr.DataArray | None = None,
        skipna : bool | None = None
    ) -> Self:
        """
        Calculate grid-aware weighted mean of a variable defined on a NEMO model grid.

        Parameters
        ----------
        dims : list
            Dimensions over which to apply weighted mean (e.g., ['i', 'j']).
        mask: xr.DataArray, optional
            Boolean mask identifying NEMO model grid points to be included (1)
            or neglected (0) from integration.
        skipna : bool | None
            If True, skip missing values (as marked by NaN).
            By default, only skips missing values for float dtypes.

        Returns
        -------
        NEMODataArray
            Grid-aware weighted mean of variable defined on a NEMO model grid.

        Examples
        --------
        Area-weighted mean of the sea surface temperature `tos_con` defined on scalar T-points
        in a NEMO model nested child domain:

        >>> nemo["gridT/1_gridT/tos_con"].weighted_mean(dims=["i", "j"], skipna=True)

        See Also
        --------
        masked_statistic
        """
        # -- Validate Input -- #
        if not isinstance(dims, list):
            raise TypeError("dims must be specified as a list.")
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise TypeError("mask must be an xarray.DataArray.")
            if any(dim not in self.dims for dim in mask.dims):
                raise ValueError(
                    f"mask must have dimensions subset from {self.dims}."
                )
        if skipna is not None:
            if not isinstance(skipna, bool):
                raise TypeError("skipna must be specified as a boolean or None.")

        # -- Calculate weighted mean & return NEMODataArray -- #
        da = self.apply_mask(mask=mask).data
        weight_dims = [dim.replace(self._dom_suffix, "") for dim in dims]
        weights = self._tree._get_weights(grid=self._grid, dims=weight_dims)
        result = da.weighted(weights).mean(dim=dims, skipna=skipna)
        result.name = f"wmean_{'_'.join(dims)}({self.name})"

        return self._wrap(result)
    
    def diff(
        self,
        dim: str,
        fillna: bool = False,
    ) -> Self:
        """
        Calculate the discrete difference of a variable along a given dimension
        (e.g., 'i', 'j', 'k') of a NEMO model grid.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference (e.g., 'i', 'j', 'k').
        fillna : bool, optional
            Fill NaN values in NEMODataArray with zeros prior to finite differencing. Default is False.
        Returns
        -------
        NEMODataArray
            Discrete difference of variable defined on a new NEMO model grid. For example, the
            discrete difference along the i-dimension of a scalar variable defined on a T-grid
            returns a NEMODataArray defined on the U-grid. 

        Examples
        --------
        Compute the difference of sea surface temperature `tos_con` values
        along the NEMO parent domain `j` dimension:

        >>> nemo['gridT/tos_con'].diff(dim="j")

        Compute the difference of sea surface temperature `tos_con` values along a regional subset of a global,
        zonally periodic domain NEMO parent domain `i` dimension:

        >>> nemo['gridT/tos_con'].sel(i=slice(10, 80)).diff(dim="i")

        Here, the zonal periodicity inherited from the NEMO model grid is automatically set to `False` since a subset
        of the global domain is no longer zonally periodic.

        Compute the discrete difference of absolute salinity `so_abs` values along the first NEMO
        nested child domain `k` dimension:

        >>> nemo['gridT/1_gridT/so_abs'].diff(dim="k")

        See Also
        --------
        derivative
        """
        # -- Validate Inputs -- #
        if not isinstance(dim, str):
            raise ValueError(
                "dim must be a string specifying dimension along which to calculate the gradient (e.g., 'i', 'i1', 'j', 'j1', 'k', 'k1')."
            )
        if dim not in self.dims:
            raise KeyError(
                f"dimension '{dim}' not found in {self.name or 'unnamed'} dimensions {self.dims}."
            )
        if not isinstance(fillna, bool):
            raise TypeError(
                "`fillna` must be specified as a boolean. Default is False."
            )

        # -- Get NEMO model grid properties -- #
        grid_paths = self._tree._get_grid_paths(dom=self._dom)

        # -- Calculate 1st-order discrete difference -- #
        if fillna:
            da = self.masked.data.fillna(value=0)
        else:
            da = self.masked.data

        if dim == self.k_name:
            match self._grid_suffix:
                case "w":
                    # W-grid located at k = 0.5, 1.5 ... Nk-0.5
                    result = da.diff(dim=self.k_name, n=1, label="lower")
                    # Fill final T-point [k=Nk] -> NaN:
                    result = result.pad({self.k_name: (0, 1)}, constant_values=np.nan)
                    result.coords[self.k_name] = (result.coords[self.k_name] + 0.5).astype(int)
                case _:
                    # T/U/V/F-grids located at k = 1, 2 ... Nk
                    result = da.diff(dim=self.k_name, n=1, label="lower")
                    # Fill initial W-point [k=0.5] -> NaN:
                    result = result.pad({self.k_name: (1, 0)}, constant_values=np.nan)
                    result.coords[self.k_name] = (result.coords[self.k_name].fillna(0) + 0.5)

        elif dim == self.i_name:
            match self._grid_suffix:
                case "t" | "v" | "w":
                    # T/V/W-grids located at i = 1, 2 ... Ni
                    if self.iperio:
                        result = da.roll({self.i_name: -1}) - da
                    else:
                        result = da.shift({self.i_name: -1}) - da
                    result.coords[self.i_name] = result.coords[self.i_name] + 0.5
                case "u" | "f":
                    # U/F-grids located at i = 1.5, 2.5 ... Ni+0.5
                    if self.iperio:
                        result = da - da.roll({self.i_name: 1})
                        result.coords[self.i_name] = result.coords[self.i_name] - 0.5
                    else:
                        # Fill initial U/F-point [i=0.5] -> NaN or 0:
                        result = da.pad({self.i_name: (1, 0)}, constant_values=0)
                        result = result.diff(dim=self.i_name, n=1, label="lower")
                        result.coords[self.i_name] = (result.coords[self.i_name].fillna(0.5) + 0.5).astype(int)

        elif dim == self.j_name:
            match self._grid_suffix:
                case "t" | "u" | "w":
                    # T/U/W-grids located at j = 1, 2 ... Nj
                    result = da.shift({self.j_name: -1}) - da
                    result.coords[self.j_name] = result.coords[self.j_name] + 0.5
                case "v" | "f":
                    # V/F-grids located at j = 1.5, 2.5 ... Nj+0.5
                    # Fill initial V/F-point [j=0.5] -> NaN or 0:
                    result = da.pad({self.j_name: (1, 0)}, constant_values=0)
                    result = result.diff(dim=self.j_name, n=1, label="lower")
                    result.coords[self.j_name] = (result.coords[self.j_name].fillna(0.5) + 0.5).astype(int)
        else:
            raise ValueError(f"Invalid dimension {dim}. Dimension must be one of (i{self._dom}, j{self._dom}, k{self._dom}).")

        # -- Updating NEMO model grid coordinates -- #
        geo_coords = [coord for coord in da.coords if coord not in (self.t_name, self.k_name, self.i_name, self.j_name)]

        # Determine new NEMO model grid type & dims:
        new_ijk_dims = [dim for dim in result.dims if dim != 'time_counter']
        new_grid_suffix, new_hgrid_suffix, _ = _NEMO_DIFF_MAP[(self._grid_suffix.upper(), dim.replace(self._dom_suffix, ""))]
        if len(new_grid_suffix) > 1:
            new_vgrid_suffix = new_grid_suffix[1]
        else:
            new_vgrid_suffix = new_grid_suffix

        # Define new geographical coordinates (glam, gphi, depth) based on new grid type:
        # NOTE: Subsets of NEMODataTree geographical coordinates are supported implicitly since the
        # (i, j, k) coordinates of the grid dimensions (i, j, k) can be used to align incoming (i.e., _tree) coordinates.
        new_coords = {
            f"{self._dom_prefix}glam{new_hgrid_suffix}": self._tree[grid_paths[f"grid{new_hgrid_suffix.upper()}"]][f"{self._dom_prefix}glam{new_hgrid_suffix}"],
            f"{self._dom_prefix}gphi{new_hgrid_suffix}": self._tree[grid_paths[f"grid{new_hgrid_suffix.upper()}"]][f"{self._dom_prefix}gphi{new_hgrid_suffix}"],
        }
        if self.k_name in new_ijk_dims:
            new_coords[f"{self._dom_prefix}depth{new_vgrid_suffix}"] = self._tree[grid_paths[f"grid{new_vgrid_suffix.upper()}"]][f"{self._dom_prefix}depth{new_vgrid_suffix}"]

        # -- Update DataArray properties -- #
        result = result.drop_vars(geo_coords)
        result = result.assign_coords(new_coords)
        result.name = f"diff_{dim}({self.name})"

        # -- Apply land-sea mask & return NEMODataArray -- #
        new_grid = f"{self._grid.replace(self._grid[-1], new_grid_suffix.upper())}"
        result = NEMODataArray(da=result, tree=self._tree, grid=new_grid)
        result = result.masked

        return result
    
    def derivative(
        self,
        dim: str,
        fillna: bool = False,
    ) -> Self:
        """
        Calculate the derivative of a variable along a given dimension
        (e.g., 'i', 'j', 'k') of a NEMO model grid.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the derivative (e.g., 'i', 'j', 'k').
        fillna : bool, optional
            Fill NaN values in NEMODataArray with zeros prior to finite differencing. Default is False.

        Returns
        -------
        NEMODataArray
            Derivative of variable defined on a NEMO model grid.

        Examples
        --------
        Compute the derivative of sea surface temperature `tos_con` values
        along the NEMO parent domain `j` dimension:

        >>> nemo['gridT/tos_con'].derivative(dim="j")

        Compute the derivative of sea surface temperature `tos_con` values along a regional subset of a global,
        zonally periodic domain NEMO parent domain `i` dimension:

        >>> nemo['gridT/tos_con'].sel(i=slice(10, 80)).derivative(dim="i")

        Here, the zonal periodicity inherited from the NEMO model grid is automatically set to `False` since a subset
        of the global domain is no longer zonally periodic.

        Compute the discrete difference of absolute salinity `so_abs` values along the first NEMO
        nested child domain `k` dimension:

        >>> nemo['gridT/1_gridT/so_abs'].derivative(dim="k")

        See Also
        --------
        diff
        """
        # -- Validate Inputs -- #
        if not isinstance(dim, str):
            raise ValueError(
                "dim must be a string specifying dimension along which to calculate the gradient (e.g., 'i', 'i1', 'j', 'j1', 'k', 'k1')."
            )
        if dim not in self.dims:
            raise KeyError(
                f"dimension '{dim}' not found in {self.name or 'unnamed'} dimensions {self.dims}."
            )
        if not isinstance(fillna, bool):
            raise TypeError(
                "`fillna` must be specified as a boolean. Default is False."
            )

            
        # -- Calculate 1st-discrete derivative along dimension -- #
        # Determine NEMO model grid type and scale factors for derivative:
        new_grid_suffix, _, new_grid_weights = _NEMO_DIFF_MAP[(self._grid_suffix.upper(),
                                                               dim.replace(self._dom_suffix, "")
                                                               )]
        # Define path to derivative NEMO model grid:
        new_grid = f"{self._grid.replace(self._grid[-1], new_grid_suffix.upper())}"

        # Collect derivative grid scale factors (e.g., e1u, e3t etc.):
        try:
            weights = self._tree[new_grid][new_grid_weights]
        except KeyError as e:
            raise KeyError(
                f"NEMO model grid: '{new_grid}' does not contain grid scale factor '{new_grid_weights}' required to calculate derivatives along the {dim}-dimension."
            ) from e

        # Calculate 1st-finite difference along dimension:
        da = self.diff(dim=dim, fillna=fillna)

        # Calculate derivative (i.e., diff(var) / e{1/2/3}{t/u/v/w}):
        if dim in [self.k_name]:
            # Vertical derivative [k increasing downward]:
            result = - da.data / weights
        else:
            # Horizontal derivative [i/j increasing eastward/northward]:
            result = da.data / weights

        # -- Update DataArray properties & return NEMODataArray -- #
        result.name = f"d({self.name})/d{dim}"
        result = NEMODataArray(da=result, tree=self._tree, grid=new_grid)

        return result


    def integral(
        self,
        dims: list,
        cum_dims: list | None = None,
        dir: str | None = None,
        mask: xr.DataArray | None = None,
    ) -> Self:
        """
        Integrate variable along one or more dimensions of a NEMO model grid.

        Parameters
        ----------
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
        NEMODataArray
            Variable integrated along specified dimensions of the NEMO model grid.


        Examples
        --------
        Compute the integral of conservative temperature `thetao_con` along the vertical
        `k` dimension in the NEMO parent domain:

        >>> nemo['gridT/thetao_con'].integral(dims=["k"])

        Compute the vertical meridional overturning stream function from the meridional
        velocity `vo` (zonally integrated meridional velocity accumulated with increasing
        depth):

        >>> nemo['gridV/vo'].integral(dims=["i", "k"],
        ...                           cum_dims=["k"],
        ...                           dir="+1",
        ...                           )

        See Also
        --------
        depth_integral
        """
        # -- Validate Input -- #
        if cum_dims is not None:
            for dim in cum_dims:
                if dim not in dims:
                    raise ValueError(
                        f"cumulative integration dimension '{dim}' not included in `dims`."
                    )
            if dir not in ["+1", "-1"]:
                raise ValueError(
                    f"invalid direction of cumulative integration '{dir}'. Expected '+1' or '-1'."
                )
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                raise ValueError("mask must be an xarray.DataArray.")
            if any(dim not in self.dims for dim in mask.dims):
                raise ValueError(
                    f"mask must have dimensions subset from {self.dims}."
                )

        # -- Collect variable, weights & mask -- #
        da = self.apply_mask(mask=mask).data
        weights = self._tree._get_weights(grid=self._grid, dims=dims)

        # -- Perform integration -- #
        if cum_dims is not None:
            sum_dims = [dim for dim in dims if dim not in cum_dims]
            if dir == "+1":
                # Cumulative integration along ordered dimension:
                result = (
                    da.weighted(weights)
                    .sum(dim=sum_dims, skipna=True)
                    .cumsum(dim=cum_dims, skipna=True)
                )
            elif dir == "-1":
                # Cumulative integration along reversed dimension:
                result = (
                    da.weighted(weights)
                    .sum(dim=sum_dims, skipna=True)
                    .reindex({dim: self.data[dim][::-1] for dim in cum_dims})
                    .cumsum(dim=cum_dims, skipna=True)
                )
        else:
            result = da.weighted(weights).sum(dim=dims, skipna=True)

        # -- Apply land-sea mask & return NEMODataArray -- #
        result.name = f"integral_{''.join(dims)}({self.name})"
        result = self._wrap(result)

        return result
    
    def depth_integral(
        self,
        limits: tuple[int | float]
    ) -> Self:
        """
        Integrate a variable in depth coordinates between two limits.

        Parameters
        ----------
        limits : tuple[int | float]
            Limits of depth integration given as a tuple of the form
            (depth_min, depth_max) where depth_min and depth_max are
            the lower and upper limits of vertical integration, respectively.

        Returns
        -------
        NEMODataArray
            Vertical integral of chosen variable between two depth surfaces (depth_min, depth_max).

        Examples
        --------
        Vertically integrate the conservative temperature variable `thetao_con` defined in a
        NEMO model parent domain from the sea surface to 100 m depth:

        >>> nemo["gridT/thetao_con"].depth_integral(limits=(0, 100))

        See Also
        --------
        integral
        """
        # -- Validate Input -- #
        if (not isinstance(limits, tuple)) | (len(limits) != 2):
            raise TypeError(
                "depth limits of integration should be given by a tuple of the form (depth_min, depth_max)"
            )
        if (limits[0] < 0) | (limits[1] < 0):
            raise ValueError("depth limits of integration must be non-negative.")
        if limits[0] >= limits[1]:
            raise ValueError(
                "lower depth limit must be less than upper depth limit."
            )

        # -- Define input variables -- #
        var_in = self.masked.data
        e3_in = self.metrics["e3"].masked.data

        # -- Vertically integrate w.r.t depth -- #
        result = xr.apply_ufunc(
            compute_depth_integral,
            e3_in,
            var_in,
            np.array([limits[1]]),
            np.array([limits[0]]),
            input_core_dims=[[self.k_name], [self.k_name], [], []],
            output_core_dims=[["k_new"]],
            dask="parallelized",
            output_dtypes=[var_in.dtype],
            dask_gufunc_kwargs={"output_sizes": {"k_new": 1}},
        )

        # -- Define integral variable DataArray -- #
        dim_list = [dim for dim in [self.t_name, "k_new", self.j_name, self.i_name] if (dim is not None) and (dim in result.dims)]
        result = result.transpose(*dim_list).squeeze()
        result.name = f"integral_z({self.name})"

        # -- Apply land-sea mask & return NEMODataArray -- #
        result = self._wrap(result)

        return result

    def masked_statistic(
        self,
        lon_poly: list | np.ndarray,
        lat_poly: list | np.ndarray,
        statistic: str,
        dims: list,
        skipna: bool | None = None,
    ) -> Self:
        """
        Compute masked statistic of a variable defined on a NEMO model grid.

        Parameters
        ----------
        lon_poly : list | np.ndarray
            Longitudes of closed polygon.
        lat_poly : list | np.ndarray
            Latitudes of closed polygon.
        statistic : str
            Name of the statistic to calculate (e.g., 'mean', 'weighted_mean' 'sum').
        dims : list
            Dimensions over which to apply statistic (e.g., ['i', 'j']).
        skipna : bool | None
            If True, skip missing values (as marked by NaN). By default, only skips missing values for float dtypes.

        Returns
        -------
        NEMODataArray
            Masked statistic of variable defined on a NEMO model grid.

        Examples
        --------
        Compute the grid cell area-weighted mean sea surface temperature `tos_con` for a
        region enclosed in a polygon defined by `lon_poly` and `lat_poly` in a NEMO nested
        child domain:

        >>> nemo["gridT/1_gridT/tos_con"].masked_statistic(lon_poly=lon_poly,
        ...                                                lat_poly=lat_poly,
        ...                                                statistic="weighted_mean",
        ...                                                dims=["i", "j"]
        ...                                                )

        See Also
        --------
        NEMODataTree.binned_statistic
        """
        # -- Validate input -- #
        if not isinstance(statistic, str):
            raise TypeError("statistic must be specified as a string.")
        if not isinstance(dims, list):
            raise TypeError("dims must be specified as a list.")
        if skipna is not None:
            if not isinstance(skipna, bool):
                raise TypeError("skipna must be specified as a boolean or None.")

        # -- Create polygon mask using coordinates -- #
        mask_poly = self._tree.mask_with_polygon(
            lon_poly=lon_poly, lat_poly=lat_poly, grid=self._grid
        )

        # -- Apply masks & calculate statistic -- #
        da = self.apply_mask(mask=mask_poly).data

        match statistic:
            case "mean":
                result = da.mean(dim=dims, skipna=skipna)

            case "weighted_mean":
                weight_dims = [dim.replace(self._dom_suffix, "") for dim in dims]
                weights = self._tree._get_weights(grid=self._grid, dims=weight_dims)
                result = da.weighted(weights).mean(dim=dims, skipna=skipna)

            case "min":
                result = da.min(dim=dims, skipna=skipna)

            case "max":
                result = da.max(dim=dims, skipna=skipna)

            case "sum":
                result = da.sum(dim=dims, skipna=skipna)

            case _:
                raise ValueError(
                    f"Unsupported statistic '{statistic}'. Supported statistics are: 'mean', 'weighted_mean', 'min', 'max', 'sum'."
                )
            
        # -- Update DataArray properties -- #
        result.name = f"masked_{statistic}({self.name})"
        result = self._wrap(result)

        return result

    def interp_to(
        self,
        to: str,
    ) -> Self:
        """
        Linearly interpolate variable to a neighbouring horizontal grid.

        For flux variables defined at U/V-points, the specified variable
        is first weighted by grid cell face area prior to linear interpolation,
        and is then normalised by the area of the target grid cell face following
        interpolation.

        Parameters
        ----------
        to : str
            Suffix of neighbouring horizontal NEMO model grid to linear interpolate
            variable to. Options are 'T', 'U', 'V', 'F'.

        Returns
        -------
        NEMODataArray
            Variable linearly interpolated onto a neighbouring horizontal grid.

        Examples
        --------
        Linearly interpolate conservative temperature `thetao_con` defined on scalar T-points
        to neighbouring V-points in a NEMO model parent domain:

        >>> nemo['gridT/thetao_con'].interp_to(to='V')

        See Also
        --------
        transform_vertical_grid
        """
        # -- Validate input -- #
        if not isinstance(to, str):
            raise TypeError(f"'to' must be a string, got {type(to)}.")
        if to not in ["T", "U", "V", "F"]:
            raise ValueError(f"'to' must be one of ['T', 'U', 'V', 'F'], got {to}.")

        # -- Get NEMO model grid properties -- #
        ijk_names = self._tree._get_ijk_names(grid=self._grid)
        target_grid = f"{self._grid.replace(self._grid[-1], to)}"

        # -- Collect variable grid scale factors -- #
        if self._grid_suffix.upper() in ["U", "V"]:
            weight_dims = (
                [self.k_name, self.j_name] if self._grid_suffix.upper() == "U" else [self.k_name, self.i_name]
            )
            if f"{self._dom_prefix}depth{self._grid_suffix}" in self.coords:
                # 3-D variables - weight by grid cell face area:
                weights = self._tree._get_weights(grid=self._grid, dims=weight_dims, fillna=False)
                target_weights = self._tree._get_weights(
                    grid=target_grid, dims=weight_dims, fillna=False
                )
            else:
                # 2-D variables - weight by grid cell width:
                weights = self._tree._get_weights(grid=self._grid, dims=weight_dims[1], fillna=False)
                target_weights = self._tree._get_weights(
                    grid=target_grid, dims=weight_dims[1], fillna=False
                )
            da = self.masked.data * weights
        else:
            # Scalar variables:
            da = self.masked.data

        # -- Linearly interpolate variable -- #
        result = interpolate_grid(
            da=da,
            mask=self.mask,
            source_grid=self._grid_suffix.upper(),
            target_grid=to,
            iperio=self.iperio,
            ijk_names=ijk_names,
        )

        # -- Updating NEMO model grid coordinates -- #
        geo_coords = [coord for coord in da.coords if coord not in (self.t_name, self.k_name, self.i_name, self.j_name)]

        # Define new geographical coordinates (glam, gphi, depth) based on new grid type:
        # NOTE: Subsets of NEMODataTree geographical coordinates are supported implicitly since the
        # (i, j, k) coordinates of the grid dimensions (i, j, k) can be used to align incoming (i.e., _tree) coordinates.
        new_coords = {
            f"{self._dom_prefix}glam{to.lower()}": self._tree[target_grid][f"{self._dom_prefix}glam{to.lower()}"],
            f"{self._dom_prefix}gphi{to.lower()}": self._tree[target_grid][f"{self._dom_prefix}gphi{to.lower()}"],
        }
        if self.k_name in result.coords:
            new_coords[f"{self._dom_prefix}depth{to.lower()}"] = self._tree[target_grid][f"{self._dom_prefix}depth{to.lower()}"]

        # -- Update DataArray properties -- #
        result = result.drop_vars(geo_coords)
        result = result.assign_coords(new_coords)
        # Retain original variable name:
        result.name = da.name

        # Reorder dimensions (time_counter, [k], j, i):
        var_dims = [dim for dim in [self.t_name, self.k_name, self.j_name, self.i_name] if (dim is not None) and (dim in result.dims)]
        result = result.transpose(*var_dims).squeeze()

        # Normalise by target grid cell weights for flux variables:
        if self._grid_suffix.upper() in ["U", "V"]:
            result = result / target_weights

        # -- Apply land-sea mask & return NEMODataArray -- #
        result = NEMODataArray(da=result, tree=self._tree, grid=target_grid)
        result = result.masked

        return result

    def transform_vertical_grid(
        self, e3_new: xr.DataArray
    ) -> xr.Dataset:
        """
        Transform variable defined on a NEMO model grid to a new vertical grid using conservative interpolation.

        Parameters
        ----------
        e3_new : xarray.DataArray
            Grid cell thicknesses of the new vertical grid.
            Must be a 1-dimensional xarray.DataArray with
            dimension 'k_new'.

        Returns
        -------
        xr.Dataset
            Variable defined at the centre of each vertical
            grid cell on the new grid, and vertical grid cell
            thicknesses adjusted for model bathymetry.

        Examples
        --------
        Transform the conservative temperature variable `thetao_con` defined in a
        NEMO model parent domain from it's native 75 unevenly-spaced z-levels to
        regularly spaced z-levels at 200 m intervals:

        >>> e3t_target = xr.DataArray(np.repeat(200.0, 30), dims=['k_new'])

        >>> nemo['gridT/thetao_con'].transform_vertical_grid(e3_new=e3t_target)

        See Also
        --------
        transform_to
        """
        # -- Validate input -- #
        if e3_new.dims != ("k_new",) or (e3_new.ndim != 1):
            raise ValueError(
                "e3_new must be a 1-dimensional xarray.DataArray with dimension 'k_new'."
            )

        # -- Define input variables -- #
        var_in = self.masked.data
        e3_in = self.metrics["e3"].masked.data
        if e3_new.sum(dim="k_new") < var_in[f"depth{self._grid_suffix}"].max(dim=self.k_name):
            raise ValueError(
                f"e3_new must sum to at least the maximum depth ({var_in[f'depth{self._grid_suffix}'].max(dim=self.k_name).item()} m) of the original vertical grid."
            )

        # -- Transform variable to target vertical grid -- #
        var_out, e3_out = xr.apply_ufunc(
            transform_vertical_coords,
            e3_in,
            var_in,
            e3_new.astype(e3_in.dtype),
            input_core_dims=[[self.k_name], [self.k_name], ["k_new"]],
            output_core_dims=[["k_new"], ["k_new"]],
            dask="parallelized",
            output_dtypes=[var_in.dtype, e3_in.dtype],
            dask_gufunc_kwargs={"output_sizes": {"k_new": e3_new.sizes["k_new"]}},
        )

        # -- Construct transformed variable Dataset -- #
        var_dims = [dim for dim in [self.t_name, "k_new", self.j_name, self.i_name] if (dim is not None) and (dim in var_out.dims)]
        var_out = var_out.transpose(*var_dims).squeeze()
    
        e3_dims = [dim for dim in [self.t_name, "k_new", self.j_name, self.i_name] if (dim is not None) and (dim in e3_out.dims)]
        e3_out = e3_out.transpose(*e3_dims).squeeze()

        result = xr.Dataset(
            data_vars={self.name: var_out, f"e3{self._grid_suffix}_new": e3_out},
            coords={
                f"depth{self._grid_suffix}_new": ("k_new", e3_new.cumsum(dim="k_new").data)
            },
        )

        return result

    # ----------------
    # Binary Operators 
    # ----------------
    def _binary_op(self, other: Self | xr.DataArray | int | float, op) -> Self:
        if isinstance(other, NEMODataArray):
            if self.grid != other.grid:
                raise ValueError(
                    f"Cannot perform binary operation between NEMODataArrays defined on different NEMO grids ({self.name} -> {self.grid}, {other.name} -> {other.grid})."
                )
        result = op(self._da, self._unwrap(other))
        return self._wrap(result)

    def _rbinary_op(self, other: Self | xr.DataArray | int | float, op) -> Self:
        if isinstance(other, NEMODataArray):
            if self.grid != other.grid:
                raise ValueError(
                    f"Cannot perform binary operation between NEMODataArrays defined on different NEMO grids ({self.name} -> {self.grid}, {other.name} -> {other.grid})."
                )
        result = op(self._unwrap(other), self._da)
        return self._wrap(result)

    def __add__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._binary_op(other, operator.add)
    def __radd__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._rbinary_op(other, operator.add)
    
    def __sub__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._binary_op(other, operator.sub)
    def __rsub__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._rbinary_op(other, operator.sub)
    
    def __mul__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._binary_op(other, operator.mul)
    def __rmul__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._rbinary_op(other, operator.mul)

    def __truediv__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._binary_op(other, operator.truediv)
    def __rtruediv__(self, other: Self | xr.DataArray | int | float) -> Self:
        return self._rbinary_op(other, operator.truediv)
    
    # ----------------
    # Utility Methods 
    # ----------------
    def _wrap(self, da: xr.DataArray) -> Self:
        """
        Wrap xarray.DataArray to preserve existing NEMO domain & grid attributes.
        """
        return NEMODataArray(da=da, tree=self._tree, grid=self._grid)

    def _unwrap(self, other: Any) -> Any:
        """
        Unwrap NEMODataArray to access underlying xarray.DataArray.
        """
        if isinstance(other, NEMODataArray):
            return other.data
        return other

    def _conditional_wrap(self, result: Any) -> Any:
        """
        Conditionally wrap result of methods when returning xarray.DataArray.
        """
        if isinstance(result, xr.DataArray):
            try:
                # Attempt to return NEMODataArray:
                return self._wrap(result)
            except Exception:
                # Otherwise, return unwrapped xarray.DataArray:
                return result 
        return result

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attributes to xarray.DataArray and attempt to return
        result as NEMODataArray.
        """
        attr = getattr(self._da, name)

        if callable(attr):
            @wraps(attr)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                return self._conditional_wrap(result)
            return wrapper

        return attr

    def __repr__(self) -> str:
        return (
            f"<NEMODataTree '{self._tree.name or 'unnamed'}'>\n"
            f"  <NEMODataArray '{self.name or 'unnamed'}' (Domain: '{self._dom}', "
            f"Grid: '{self._grid}', Grid Type: '{self._grid_suffix.upper()}')>\n\n"
            f"{repr(self._da)}"
        )

    def _repr_html_(self) -> str:
        banner = f"""
        <div style="margin-bottom: 10px;">
            <div">
                <span style="color:#6a737d;">NEMODataTree</span> '{self._tree.name or "unnamed"}'
            </div>

            <div style="margin-left: 18px; margin-top:4px;">
                <span style="color:#6a737d;">NEMODataArray</span> '{self.name or "unnamed"}'
                <div>├─ <strong>Domain:</strong> '{self._dom}'</div>
                <div>├─ <strong>Grid Path:</strong> '{self._grid}'</div>
                <div>└─ <strong>Grid Type:</strong> '{self._grid_suffix.upper()}'</div>
            </div>
        </div>
        """

        return f"""
        <div style="border:1px solid #ddd; padding:12px; border-radius:8px;">
            {banner}
            {self.data._repr_html_()}
        </div>
        """


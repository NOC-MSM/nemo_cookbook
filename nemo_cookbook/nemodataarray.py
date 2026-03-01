"""
nemodataarray.py

Description:
This module defines the NEMODataArray class, a grid-aware xarray.DataArray wrapper
for NEMO ocean model variables, providing discrete operators such as differences,
derivatives, and integrals, and supporting transformation between NEMO grids.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
from typing import Self

import numpy as np
import xarray as xr

from nemo_cookbook import NEMODataTree
from nemo_cookbook.integrate import compute_depth_integral
from nemo_cookbook.interpolate import interpolate_grid


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

        if not isinstance(tree, NEMODataTree):
            raise TypeError("tree must be specified as a NEMODataTree.")
        else:
            self._tree = tree

        if not isinstance(grid, str):
            raise TypeError("grid must be specified as a string.")
        else:
            self._grid = grid

        # -- Validate DataArray compatibility -- #
        grid_keys = list(dict(self._tree.subtree_with_keys).keys())
        if grid not in grid_keys:
            raise KeyError(
                f"grid '{grid}' not found in available NEMODataTree grids {grid_keys}."
            )
        if not all(da[dim].identical(self._tree[grid][dim]) for dim in da.dims):
            raise ValueError(
                f"DataArray dimensions {da.dims} not identical to NEMO model '{grid}' dimensions."
            )
        if not all(da.coords[coord].identical(self._tree[grid].coords[coord]) for coord in da.coords):
            raise ValueError(
                f"DataArray coordinates {da.coords} not identical to NEMO model '{grid}' coordinates."
            )
        
        # -- Assign NEMO domain number, grid path and grid type -- #
        dom, dom_prefix, dom_suffix, grid_suffix = self._tree._get_properties(grid=grid, infer_dom=True)
        self._dom = dom
        self._dom_prefix = dom_prefix
        self._dom_suffix = dom_suffix
        self._grid_suffix = grid_suffix

    # -----------
    # Properties 
    # -----------
    @property
    def data(self):
        """
        Access underlying xarray.DataArray.
        """
        return self._da
    
    @property
    def grid(self):
        """
        Access path to parent NEMO model grid.
        """
        return self._grid
    
    @property
    def grid_type(self):
        """
        Access type of parent NEMO model grid (e.g. 't', 'u', etc.).
        """
        return self._grid_suffix
    
    @property
    def metrics(self):
        """
        Access grid scale factors for the parent NEMO model grid.
        """
        # 2-dimensional: Horizontal grid scale factors:
        d_metrics = {"e1": self._tree[f"{self._grid}/e1{self._grid_suffix}"],
                     "e2": self._tree[f"{self._grid}/e2{self._grid_suffix}"],
                     }

        if f"{self._dom_prefix}depth{self._grid_suffix}" in self._da.coords:
            # 3-dimensional: Vertical grid scale factor:
            d_metrics["e3"] = self._tree[f"{self._grid}/e3{self._grid_suffix}"]

        return d_metrics
    
    @property
    def mask(self):
        """
        Access variable land-sea mask for the parent NEMO model grid.
        """
        if f"{self._dom_prefix}depth{self._grid_suffix}" in self.coords:
            # 3-dimensional land-sea mask:
            mask_name = f"{self._grid_suffix}mask"
        else:
            # 2-dimensional land-sea mask:
            mask_name = f"{self._grid_suffix}maskutil"
        
        return self._tree[self._grid][mask_name]
    
    @property
    def masked(self):
        """
        Access land-sea masked variable define on NEMO model grid.
        """
        return self.apply_mask()

    # ---------------
    # Public Methods 
    # ---------------
    def apply_mask(
        self,
        mask: xr.DataArray | None = None,
        ) -> Self:
        """
        Apply NEMO parent grid land-sea mask or combined land-sea & custom mask
        to variable defined on a NEMO model grid.

        Parameters
        ----------
        mask : xr.DataArray | None
            Boolean mask to apply to variable defined on NEMO model grid. Default is None
            meaning only land-sea mask of NEMO parent grid is applied.

        Returns
        -------
        NEMODataArray
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
        result = self._da.where(mask, drop=False)

        return self._wrap(result)
    
    def weighted_mean(
        self,
        dims : list,
        skipna : bool | None = None
        ) -> Self:
        """
        Calculate grid-aware weighted mean of a variable defined on a NEMO model grid.

        Parameters
        ----------
        dims : list
            Dimensions over which to apply weighted mean (e.g., ['i', 'j']).
        skipna : bool | None
            If True, skip missing values (as marked by NaN).
            By default, only skips missing values for float dtypes.

        """
        # -- Validate Input -- #
        if not isinstance(dims, list):
            raise TypeError("dims must be specified as a list.")
        if skipna is not None:
            if not isinstance(skipna, bool):
                raise TypeError("skipna must be specified as a boolean or None.")

        # -- Calculate weighted mean & return NEMODataArray -- #
        weight_dims = [dim.replace(self._dom_suffix, "") for dim in dims]
        weights = self._tree._get_weights(grid=self._grid, dims=weight_dims)
        result = self.weighted(weights).mean(dim=dims, skipna=skipna)

        return self._wrap(result)
    
    def diff(
        self,
        dim: str,
    ) -> xr.DataArray:
        """
        Calculate the 1st-order discrete difference of a variable along a given dimension
        (e.g., 'i', 'j', 'k') of a NEMO model grid.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference (e.g., 'i', 'j', 'k').

        Returns
        -------
        xr.DataArray
            1st-order discrete difference of variable defined on a NEMO model grid.

        Examples
        --------
        Compute the 1st discrete difference of sea surface temperature `tos_con` values
        along the NEMO parent domain `j` dimension:

        >>> nemo['gridT/tos_con'].diff(dim="j")

        Compute the 1st discrete difference of absolute salinity `so_abs` values along the first NEMO
        nested child domain `k` dimension:

        >>> nemo['gridT/1_gridT/so_abs'].diff(dim="k")

        See Also
        --------
        derivative
        """
        # -- Validate input -- #
        if not isinstance(dim, str):
            raise ValueError(
                "dim must be a string specifying dimension along which to calculate the gradient (e.g., 'i', 'i1', 'j', 'j1', 'k', 'k1')."
            )
        if dim not in self.dims:
            raise KeyError(
                f"dimension '{dim}' not found in {self.name or 'unnamed'} dimensions {self.dims}."
            )

        # -- Get NEMO model grid properties -- #
        grid_paths = self._tree._get_grid_paths(dom=self._dom)
        ijk_names = self._tree._get_ijk_names(grid=self._grid)
        i_name, j_name, k_name = ijk_names["i"], ijk_names["j"], ijk_names["k"]
        iperio = self._tree[self._grid].attrs.get("iperio", False)

        # -- Calculate 1st-order discrete difference -- #
        da = self.masked.data

        if dim == k_name:
            match self._grid_suffix:
                case "w":
                    # W-grid located at k = 0.5, 1.5 ... Nk-0.5
                    result = da.diff(dim=k_name, n=1, label="lower")
                    result = result.pad({k_name: (0, 1)}, constant_values=np.nan)
                    result.coords[k_name] = (result.coords[k_name] + 0.5).astype(int)
                case _:
                    # T/U/V/F-grids located at k = 1, 2 ... Nk
                    result = da.diff(dim=k_name, n=1, label="lower")
                    result = result.pad({k_name: (1, 0)}, constant_values=np.nan)
                    result.coords[k_name] = (result.coords[k_name].fillna(0) + 0.5)

        elif dim == i_name:
            match self._grid_suffix:
                case "t" | "v" | "w":
                    # T/V/W-grids located at i = 1, 2 ... Ni
                    if iperio:
                        result = da.roll({i_name: -1}) - da
                    else:
                        result = da.shift({i_name: -1}) - da
                    result.coords[i_name] = result.coords[i_name] + 0.5
                case "u" | "f":
                    # U/F-grids located at i = 1.5, 2.5 ... Ni+0.5
                    if iperio:
                        result = da - da.roll({i_name: 1})
                        result.coords[i_name] = result.coords[i_name] - 0.5
                    else: 
                        result = da.pad({i_name: (1, 0)}, constant_values=np.nan)
                        result = result.diff(dim=i_name, n=1, label="lower")
                        result.coords[i_name] = (result.coords[i_name].fillna(0.5) + 0.5).astype(int)

        elif dim == j_name:
            match self._grid_suffix:
                case "t" | "u" | "w":
                    # T/U/W-grids located at j = 1, 2 ... Nj
                    result = da.shift({j_name: -1}) - da
                    result.coords[j_name] = result.coords[j_name] + 0.5
                case "v" | "f":
                    # V/F-grids located at j = 1.5, 2.5 ... Nj+0.5
                    result = da.pad({j_name: (1, 0)}, constant_values=np.nan)
                    result = result.diff(dim=j_name, n=1, label="lower")
                    result.coords[j_name] = (result.coords[j_name].fillna(0.5) + 0.5).astype(int)
        else:
            raise ValueError(f"Invalid dimension {dim}. Dimension must be one of (i{self._dom}, j{self._dom}, k{self._dom}).")

        # -- Updating NEMO model grid coordinates -- #
        geo_coords = [coord for coord in da.coords if coord not in ('time_counter', k_name, i_name, j_name)]

        # Determine new grid type based using integer vs. fractional i/j/k dimensions:
        new_geo_dims = [dim for dim in result.dims if dim != 'time_counter']
        new_dim_frac = [dim for dim in new_geo_dims if all((result[dim] % 1) != 0)]
        if (i_name in new_dim_frac) and (j_name in new_dim_frac):
            new_grid_suffix = "f"
        elif (i_name in new_dim_frac):
            new_grid_suffix = "u"
        elif (j_name in new_dim_frac):
            new_grid_suffix = "v"
        else:
            new_grid_suffix = "t"

        # Define new geographical coordinates (glam, gphi, depth) based on new grid type:
        new_coords = {
            f"{self._dom_prefix}glam{new_grid_suffix}": self._tree[grid_paths[f"grid{new_grid_suffix.upper()}"]][f"{self._dom_prefix}glam{new_grid_suffix}"],
            f"{self._dom_prefix}gphi{new_grid_suffix}": self._tree[grid_paths[f"grid{new_grid_suffix.upper()}"]][f"{self._dom_prefix}gphi{new_grid_suffix}"],
        }
        if k_name in new_geo_dims:
            depth_suffix = "w" if k_name in new_dim_frac else new_grid_suffix
            depth_grid = grid_paths["gridW"] if depth_suffix == "w" else grid_paths[f"grid{new_grid_suffix.upper()}"]
            new_coords[f"{self._dom_prefix}depth{depth_suffix}"] = self._tree[depth_grid][f"{self._dom_prefix}depth{depth_suffix}"]

        # -- Update DataArray properties -- #
        result = result.drop_vars(geo_coords)
        result = result.assign_coords(new_coords)
        result.name = f"diff_{dim}({self.name})"

        # -- Apply land-sea mask & return NEMODataArray -- #
        new_grid = f"{self._grid.replace(self._grid[-1], 'W' if k_name in new_dim_frac else new_grid_suffix.upper())}"
        result = NEMODataArray(da=result, tree=self._tree, grid=new_grid)
        result = result.masked

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
        # -- Validate input -- #
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
        da = (
            self.masked.data.where(mask)
            if mask is not None
            else self.masked.data
            )
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
                    .reindex({dim: self[dim][::-1] for dim in cum_dims})
                    .cumsum(dim=cum_dims, skipna=True)
                )
        else:
            result = da.weighted(weights).sum(dim=dims, skipna=True)

        # -- Apply land-sea mask & return NEMODataArray -- #
        result.name = f"integral_{', '.join(dims)}({self.name})"
        result = self._wrap(result).masked

        return result
    
    def depth_integral(
        self, limits: tuple[int | float]
    ) -> xr.Dataset:
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
        # -- Validate input -- #
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

        # -- Get NEMO model grid properties -- #
        ijk_names = self._tree._get_ijk_names(dom=self._dom)
        i_name, j_name, k_name = ijk_names["i"], ijk_names["j"], ijk_names["k"]

        # -- Define input variables -- #
        var_in = self.masked.data
        e3_in = self.metrics["e3"]

        # -- Vertically integrate w.r.t depth -- #
        result = xr.apply_ufunc(
            compute_depth_integral,
            e3_in,
            var_in,
            np.array([limits[1]]),
            np.array([limits[0]]),
            input_core_dims=[[k_name], [k_name], [None], [None]],
            output_core_dims=[["k_new"]],
            dask="allowed",
        )

        # -- Define integral variable DataArray -- #
        t_name = var_in.dims[0]
        result = result.transpose(t_name, "k_new", j_name, i_name).squeeze()
        result.name = f"integral_z({self.name})"

        # -- Apply land-sea mask & return NEMODataArray -- #
        result = self._wrap(result).masked

        return result

    def transform_to(
        self,
        to: str,
    ) -> xr.DataArray:
        """
        Transform variable to a neighbouring horizontal grid using linear interpolation.

        For flux variables defined at U/V-points, the specified variable
        is first weighted by grid cell face area prior to linear interpolation,
        and is then normalised by the area of the target grid cell face following
        interpolation.

        Parameters
        ----------
        to : str
            Suffix of neighbouring horizontal NEMO model grid to transform
            variable to. Options are 'T', 'U', 'V', 'F'.

        Returns
        -------
        NEMODataArray
            Variable linearly interpolated onto a neighbouring horizontal grid.

        Examples
        --------
        Transform conservative temperature `thetao_con` defined on scalar T-points
        to neighbouring V-points in a NEMO model parent domain:

        >>> nemo['gridT/thetao_con'].transform_to(to='V')

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
        i_name, j_name, k_name = ijk_names["i"], ijk_names["j"], ijk_names["k"]
        iperio = self._tree[self._grid].attrs.get("iperio", False)
        target_grid = f"{self._grid.replace(self._grid[-1], to)}"

        # -- Collect variable grid scale factors -- #
        if self._grid_suffix.upper() in ["U", "V"]:
            weight_dims = (
                [k_name, j_name] if self._grid_suffix.upper() == "U" else [k_name, i_name]
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
            iperio=iperio,
            ijk_names=ijk_names,
        )

        # Retain input variable name:
        result.name = da.name
        # Reorder dimensions (time_counter, [k], j, i):
        new_dims = (result.dims[-1], *result.dims[:-1])
        result = result.transpose(*new_dims)

        # Update NEMO grid coords:
        result[i_name] = self._tree[target_grid][i_name]
        result[j_name] = self._tree[target_grid][j_name]
        if k_name in result.dims:
            result[k_name] = self._tree[target_grid][k_name]

        # Drop NEMO source grid coords:
        drop_vars = [f"{self._dom_prefix}glam{self._grid_suffix}", f"{self._dom_prefix}gphi{self._grid_suffix}"]
        if f"{self._dom_prefix}depth{self._grid_suffix}" in da.coords:
            drop_vars.append(f"{self._dom_prefix}depth{self._grid_suffix}")
        result = result.drop_vars(drop_vars)

        # Normalise by target grid cell weights for flux variables:
        if self._grid_suffix.upper() in ["U", "V"]:
            result = result / target_weights

        # -- Apply land-sea mask & return NEMODataArray -- #
        result = NEMODataArray(da=result, tree=self._tree, grid=target_grid)
        result = result.masked

        return result
    
    # -----------------------------
    # Wrapped Reduction Operations
    # -----------------------------
    def mean(self, *args, **kwargs):
        result = self._da.mean(*args, **kwargs)
        return self._wrap(result)

    def median(self, *args, **kwargs):
        result = self._da.median(*args, **kwargs)
        return self._wrap(result)

    def var(self, *args, **kwargs):
        result = self._da.var(*args, **kwargs)
        return self._wrap(result)

    def std(self, *args, **kwargs):
        result = self._da.std(*args, **kwargs)
        return self._wrap(result)

    def min(self, *args, **kwargs):
        result = self._da.min(*args, **kwargs)
        return self._wrap(result)

    def max(self, *args, **kwargs):
        result = self._da.max(*args, **kwargs)
        return self._wrap(result)

    def sum(self, *args, **kwargs):
        result = self._da.sum(*args, **kwargs)
        return self._wrap(result)

    def prod(self, *args, **kwargs):
        result = self._da.prod(*args, **kwargs)
        return self._wrap(result)
    
    # ----------------
    # Utility Methods 
    # ----------------
    def _wrap(self, da):
        """
        Wrap xarray.DataArray to preserve existing NEMO domain & grid attributes.

        Parameters
        ----------
        da : xr.DataArray
            Variable defined on a NEMO model grid.
        """
        return NEMODataArray(da=da, tree=self._tree, grid=self._grid)
    
    def __getattr__(self, name):
        """
        Delegate attributes to underlying xarray.DataArray.
        """
        return getattr(self._da, name)

    def __repr__(self):
        return (
            f"<NEMODataTree '{self._tree.name or 'unnamed'}'>\n"
            f"  <NEMODataArray '{self.name or 'unnamed'}' (Domain: '{self._dom}', "
            f"Grid: '{self._grid}', Grid Type: '{self._grid_suffix.upper()}')>\n\n"
            f"{repr(self._da)}"
        )

    def _repr_html_(self):
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
        
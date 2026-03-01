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

import xarray as xr

from nemo_cookbook import NEMODataTree


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
        
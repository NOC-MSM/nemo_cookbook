# A Quickstart Guide to Common Operations with NEMODataTree

In this section, we describe some of the most common `NEMODataTree` and `NEMODataArray` operations in a concise how-to guide (inspired by the excellent documentation of [**Icechunk**](https://icechunk.io/en/latest/howto/)).

For more detailed documentation on `NEMODataTree` and `NEMODataArray` structures, users should visit the [API Reference].

[API Reference]: reference.md

### Create a NEMODataTree from Local Files

We can create a `NEMODataTree` from a dictionary of paths to local netCDF files using the `.from_paths()` constructor:

```python
paths = {"parent": {
         "domain": "/path/to/domain_cfg.nc",
         "gridT": "path/to/*_gridT.nc",
         "gridU": "path/to/*_gridV.nc",
         "gridV": "path/to/*_gridV.nc",
         "gridW": "path/to/*_gridW.nc",
         "icemod": "path/to/*_icemod.nc",
        },
        }

NEMODataTree.from_paths(paths, iperio=True, nftype="T")
```

In the example above, we consider only a global parent domain, which is zonally periodic (`iperio=True`) and north-folding on **T** grid points (`nftype="T"`). Note, that we are only required to specify paths for one or more NEMO model grid types (e.g., `*_gridT.nc`).

For NEMO models using a linear free surface approximation (i.e., vertical scale factors are time-independent), we should also specify `key_linssh=True` to read these directly from the domain_cfg file included in the `paths` dictionary.

### Create a NEMODataTree from `xarray.Datasets`

Alternatively, we can create a `NEMODataTree` from a dictionary of single or multi-file `xarray.Datasets`. This is particularly valuable when working with remote NEMO model data or Coupled Model Intercomparison Project (CMIP) outputs which require us to reformat coordinate dimensions (see **Example NEMODataTrees**).

```python
ds_domain = xr.open_zarr("https://some_remote_data/domain_cfg.zarr")
ds_gridT = xr.open_zarr("https://some_remote_data/MY_MODEL_gridT.zarr")

datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT}}

nemo = NEMODataTree.from_datasets(datasets=datasets, read_masks=True)
```

This example would be applicable to the outputs of a regional NEMO model configuration which is neither zonally periodic nor north-folding (by default, `iperio=False` & `nftype=None`).

If all land-sea masks expected by `NEMODataTree` are included in `ds_domain`, we can also specify `read_mask=True` to read rather than calculate land-sea masks when constructing our `NEMODataTree`. This is recommended for larger NEMO model domains (e.g., eORCA12, ORCA36 etc.)

### Access NEMO Variables

To access a variable stored in a given grid node of a `NEMODataTree` as an `xarray.DataArray`, we can use the following syntax:

```python
nemo[{grid_name}][{variable_name}]
```

However, if we want to access the chosen variable as a grid-aware `NEMODataArray` (**recommended**), we can instead provide the direct path to the variable:

```python
nemo["gridT/thetao_con"]
```

### Adding a NEMO Variable to a NEMODataTree

To add a new variable stored as a `NEMODataArray` to the grid node of an existing `NEMODataTree`, we can use the following syntax:

```python
nemo["gridT/my_var"] = nemo["gridT/thetao_con"].depth_integral(limits=(0, 100))
```

Note, this simply adds the underlying `xarray.DataArray` to the NEMO model **T**-grid node `xarray.Dataset` in our `NEMODataTree` with the name `my_var`.

### Adding a New Grid Node to a NEMODataTree

To create a new grid node from an `xarray.Dataset` containing NEMO model grid variables in an existing `NEMODataTree`, we can use the following syntax:

```python
nemo["gridP"] = ds_gridP
```

Before the new grid node is added to an existing `NEMODataTree`, the `xarray.Dataset` will be validated to ensure it contains:

- NEMO grid dimension coordinates (`i`, `j`).

- Longitude and Latitude coordinates named `gphi{x}` and `glam{x}`.

- Depth coordinate named `depth{x}` if NEMO grid dimension `k` exists.

where `x` is final character of the specified grid node. In this example, `gphip`, `glamp` and `depthp` would be expected since we are introducing a new grid node named `"gridP"`.

### Access NEMO Variable Properties

Each variable stored as a `NEMODataArray` has a collection of useful properties to support grid-aware calculations:

* Path to NEMO model grid node where variable is stored:
```python
nemo["gridT/thetao_con"].grid
```

* Type of NEMO model grid where variable is defined (i.e., t, u, v, etc.):
```python
nemo["gridU/uo"].grid_type
```

* Dictionary of variable NEMO grid scale factors (e.g., e1t, e2t, e3t):
```python
nemo["gridV/vo"].metrics
```

* Variable land-sea mask:
```python
nemo["gridT/so_abs"].mask
```

### Apply Land-Sea Mask to NEMO Variable

To mask a given `NEMODataArray` variable with its associated land-sea mask, we can use the `.masked` property:

```python
nemo["gridT/thetao_con"].masked
```

In the example above, the 4-dimensional conservative temperature variable `thetao_con` is masked using the 3-dimensional `tmask` before being returned as a `NEMODataArray`.

### Apply Custom Mask to NEMO Variable

To apply a custom mask to a given `NEMODataArray` variable, we can use the `.apply_mask()` method:

```python
nemo["gridT/so_abs"].apply_mask(mask=my_mask)
```

Here, the 4-dimensional absolute salinity variable `so_abs` is masked using the `my_mask` boolean mask before being returned as a `NEMODataArray`. 

To drop the absolute salinity values where `my_mask` is `False`, we can optionally use `.apply_mask(mask=my_mask, drop=True)`.

### Index a NEMODataArray to Match the Dimension Labels of Another (NEMO)DataArray

In addition to the more familiar `.sel()` and `.isel()` label based selection methods, the `.sel_like()` method can be used to index a `NEMODataArray` according to the dimension index labels of another `NEMODataArray` or `xarray.DataArray` as follows:

```python
nda = nemo["gridT/so_abs"].sel(time_counter=slice('2000-01', '2025-01', k=1))

nemo["gridT/thetao_con"].sel_like(nda)
```

### Calculate Grid Cell Areas

To calculate the area of a model grid cell face, we can use the `.cell_area()` method.

For example, to compute the horizontal area of cells centered on **T** grid points in the parent domain:

```python
nemo.cell_area(grid="gridT", dim="k")
```

Importantly, the `dim` argument represents the dimensional orthogonal to the grid cell area to be computed. For **T** grid points, this results in the following grid cell areas: 

| `dim`   | Grid Cell Area           |
| ----------- | ---------------------- |
| `i`         | e2t * e3t             |
| `j`         | e1t * e3t        |
| `k`         | e1t * e2t        |

### Calculate Grid Cell Volumes

To calculate the volume of model grid cells, we can use the `.cell_volume()` method.

For example, to compute the volume of each grid cell centered on a **V** grid point in the model parent domain:

```python
nemo.cell_volume(grid="gridV")
```

### Indexing with Geographical Coordinates

To subset variables of a given model grid using their longitude & latitude coordinates (i.e., `glam{t/u/v/w}(j, i)` & `gphi{t/u/v/w}(j, i)`), we can add these geographical variables as indexes using the `.add_geoindex()` method.

For example, to enable geographical indexing of the parent **T** grid points & select the values of this dataset nearest to (-30°E, 60°N):

```python
nemo_geo = nemo.add_geoindex(grid="gridT")

nemo_geo["gridT"].dataset.sel(gphit=60, glamt=-30, method="nearest")
```

### Clip a NEMO Model Grid

To clip a given model grid using a geographical bounding box defined by a tuple of the form (`lon_min`, `lon_max`, `lat_min`, `lat_max`), we can use the `.clip_grid()` method.

For example, to clip the parent **T**-grid in the bounding box (-80°E, 0°E, 40°N, 80°N):

```python
bbox = (-80, 0, 40, 80)

nemo.clip_grid(grid="gridT", bbox=bbox)
```

### Clip a NEMO Model Domain

To clip all of the model grids of a given NEMO model domain:

```python
nemo.clip_domain(dom=".", bbox=bbox)
```

where `dom` is the prefix of the chosen NEMO model domain. Note `dom="."` for the parent domain.

### Calculate Horizontal Derivatives

To calculate the derivative of a scalar variable `var` along one of the horizontal dimensions (e.g., `i`, `j`) of a given NEMO model grid, we can use the `.derivative()` method.

For example, to compute the 'meridional' derivative of sea surface temperature `tos_con` along the NEMO model parent domain `j` dimension:

```python
nemo["gridT/tos_con"].derivative(dim="j")
```

Alternatively, to compute the derivative of sea surface temperature `tos_con` along a regional subset of a global, zonally periodic domain NEMO model parent domain `i` dimension:

```python
nemo["gridT/tos_con"].sel(i=slice(10, 100)).derivative(dim="i", iperio=False)
```

Note, using `iperio=False` overrides the zonal periodicity inherited from the NEMO model grid since the selected subset of the global domain is no longer zonally periodic.

Both the `.diff()` and `.derivative()` methods provide flexibility on how to handle NaN values using the `fillna` argument. By default, `fillna=False`, meaning that NaN values are not filled with zeros prior to performing finite difference operations. However, in the case of velocity variables this may be inappropriate in the vicinity of coastlines and users may prefer to use `fillna=True` to impose zero-magnitude velocity components along land-sea boundaries prior to calculating derivatives. 

### Calculate Vertical Derivative

To calculate the vertical derivative of a scalar variable `var` along the `k` dimension of a given NEMO model grid, we can also use the `.derivative()` method.

For example, to compute the vertical gradient of absolute salinity in our first NEMO model nested child domain:

```python
nemo["gridT/so_abs"].derivative(dim="k")
```

### Calculate Divergence

To calculate the horizontal divergence from the `i` and `j` components of a vector field, we can use the `.divergence()` method.

For example, to compute the horizontal divergence from the seawater velocity field in the NEMO model parent domain:

```python
nemo.divergence(dom=".", vars=["uo", "vo"])
```

where `vars` is a list specifying the names of `i` and `j` vector components, respectively.

### Calculate Curl

To calculate the vertical `k` component of the curl of a horizontal vector field, we can use the `.curl()` method.

For example, to compute the vertical component of the curl of the seawater velocity field in the second NEMO nested child domain:

```python
nemo.curl(dom="2", vars=["uo", "vo"])
```

where, as in the case of `.divergence()`, the `vars` argument expects a list of the `i` and `j` components of the vector field, respectively.

### Calculate Integrals

To integrate a variable along one or more dimensions of a given NEMO model grid, we can use the `.integral()` method.

For example, to compute the integral of conservative temperature `thetao_con` along the vertical `k` dimension in the NEMO model parent domain:

```python
nemo["gridT/thetao_con"].integral(dims=["k"])
```

which will return an `NEMODataArray` with one less dimension than `thetao_con`, in this case `k` since we have integrated vertically.

### Calculate Cumulative Integrals

We can also use the `.integral()` method to calculate cumulative integrals along one or more dimensions of a given NEMO model grid.

For example, to calculate the vertical meridional overturning stream function from the meridional velocity `vo` (*zonally integrated meridional velocity accumulated with increasing depth*):

```python
nemo["gridV/vo",].integral(dims=["i", "k"], cum_dims=["k"], dir="+1",)
```
where `dims` is a list of grid dimension names along which integration will be performed, and `cum_dims` specifies which of the dimensions in `dims` should be cumulatively integrated.

The `dir` argument is used to define the direction of cumulative integration, where `dir = "+1"` means accumulating along the chosen dimension, such that grid indices are increasing. Conversely, `dir = "-1"` means that cumulative integration is performed after reversing the chosen dimension, such that grid dimensions are decreasing.

Note, we can also pass the `mask` argument to `.integral()` to mask the variable `var` prior to performing the integration.

### Calculate Depth Integrals

To integrate a variable of a given NEMO model grid in depth coordinates between two limits, we can use the `.depth_integral()` method.

For example, to compute the vertical integral of conservative temperature `thetao_con` in the upper 100 m in the NEMO model parent domain:

```python
nemo["gridT/thetao_con"].depth_integral(limits=(0, 100))
```

where `limits` is a tuple of the form (depth_min, depth_max) where depth_min and depth_max are the lower and upper limits of vertical integration, respectively.

### Calculate Weighted Average

To calculate a grid-aware weighted mean of a variable defined on a NEMO model grid, we can use the `.weighted_mean()` method.

For example, to compute the grid cell area-weighted mean sea surface temperature `tos_con` in a NEMO model nested child domain:

```python
nemo["gridT/1_gridT/tos_con"].weighted_mean(dims=["i", "j"], skipna=True)
```

where `dims` represent the dimensions of the NEMO model grid to average over. In this example, `dims=["i", "j"]` is equivalent to computing the mean of variable `tos_con` using the horizontal cell area of **T** grid points (i.e., e1t * e2t) as weights.

### Create Regional Masks using Polygons

To define a regional mask using the geographical coordinates of a closed polygon, we can use the `.mask_with_polygon()` method:

```python
nemo.mask_with_polygon(grid="gridT", lon_poly, lat_poly)
```

where `lon_poly` and `lat_poly` are lists or ndarrays containing the longitude and latitude coordinates defining the closed polygon.

### Calculate Statistics for a Region Masked using a Polygon

To calculate an aggregated statistic from only the model grid cells contained inside a geographical polygon, we can use the `.masked_statistic()` method.

For example, to compute the grid cell area-weighted mean sea surface temperature `tos_con` for a region enclosed in a polygon defined by `lon_poly` and `lat_poly` in a NEMO model nested child domain:

```python
nemo["gridT/1_gridT/tos_con",].masked_statistic(lon_poly,
                                                lat_poly,
                                                statistic="weighted_mean",
                                                dims=["i", "j"]
                                                )
```

where `dims` represent the dimensions of the NEMO model grid used for aggregation. In this example, combining `statistic="weighted_mean"` and `dims=["i", "j"]` is equivalent to computing the mean of variable `tos_con` using the horizontal cell area of **T** grid points (i.e., e1t * e2t) as weights.

### Calculate Binned Statistics

To calculate aggregated statistics of a variable binned according to the values of one or more variables, we can use the `.binned_statistic()` method. 

This is a generalization of a histogram function, enabling the computation of the sum, mean, median, or other statistic of the values assigned to each bin.

For example, to compute the mean depth associated with each isopycnal in discrete potential density (`sigma0`) coordinates:

```python
sigma0_bins = np.arange(22, 29.05, 0.05)

nemo.binned_statistic(grid="gridT",
                      vars=["sigma0"],
                      values="deptht",
                      keep_dims=["time_counter"],
                      bins=[sigma0_bins],
                      statistic="nanmean",
                      )
```

where `vars` is a list of the names of variables to be binned using the bin edges passed to `bins`, and `values` is the name of the variable over which the `statistic` will be performed once values have been grouped into each bin.

We can use `keep_dims` to specify the dimensions of the `xarray.DataArray` named `values` to retain. In the example above, using `keep_dims="time_counter"` will return the average depths of water in each potential density bin for each time-slice of available NEMO model output.

### Interpolate Variable to a Neighbouring Horizontal Grid

To linearly interpolate a variable defined on a given NEMO horizontal grid to a neighbouring grid, we can use the `.interp_to()` method.

For example, to interpolate conservative temperature `thetao_con` defined on scalar **T**-points to neighbouring **V**-points in a NEMO model parent domain:

```python
nemo["gridT/thetao_con"].interp_to(to="V")
```

We can also interpolate variables defined on **U**- and **V**-points to either scalar or vector grid points. Unlike interpolating scalar variables defined on **T**-points, this is achieved by linearly interpolating the grid cell face area-weighted flux onto the target grid, before then normalising by the grid cell face area defined on the target horizontal grid.

For example, to interpolate the zonal wind stress defined on **U**-points to neighbouring **V**-points in a NEMO model parent domain and store this in the **V**-grid node of our NEMODataTree:

```python
nemo['gridV/tauuo'] = nemo["gridU/tauuo"].interp_to(to="V")
```

### Transform a Vertical Grid

To transform a variable defined on a given NEMO model vertical grid to a new vertical grid using conservative interpolation, we can use the `.transform_vertical_grid()` method.

For example, if we wanted to transform the conservative temperature variable `thetao_con` defined in a NEMO model parent domain from it's native 75 unevenly-spaced z-levels to regularly spaced z-levels at 200 m intervals:

```python
e3t_target = xr.DataArray(np.repeat(200.0, 30), dims=['k_new'])

nemo["gridT/thetao_con"].transform_vertical_grid(e3_new = e3t_target)
```

where `e3_new` represents the time-invariant vertical grid cell thicknesses defing the vertical grid onto which the variable `var` will be conservatively interpolated. 

There are some important points to remember when transforming variables onto new vertical grids with `NEMODataTree`:

- New vertical grid cell thicknesses `e3_new` must sum to at least the maximum depth of the original vertical grid cell thicknesses (e.g., e3t).

- Currently, `e3_new` must be a 1-dimensional `xarray.DataArray` with dimension 'k_new'.

- The output `xarray.Dataset` will contain multi-dimensional `xarray.DataArrays` for both the vertically remapped variable `var(time_counter, k_new, j, i)` and the vertical grid cell thicknesses `e3t_new(time_counter, k_new, j, i)` (updated to explicitly account for partial grid cells above the seafloor).

# Core Operations

Here, we provide an introduction to the core operations that can be performed using NEMODataTree and NEMODataArray objects.

---

## **Overview**

In addition to being simply a container for a NEMO output variables, NEMODataTrees and NEMODataArrays facilitate grid-aware computation via the following operations: 

- **Combining** :material-arrow-right: `.merge()`, `.concat()`.

- **Indexing** :material-arrow-right: `.sel_like()`, `.add_geoindex()`.

- **Clipping** :material-arrow-right: `.clip()`, `.clip_grid()`, `.clip_domain()`.

- **Masking** :material-arrow-right: `.apply_mask()`, `.mask_with_polygon()`.

- **Plotting** :material-arrow-right: `.geoplot()`.

- **Grid Operators** :material-arrow-right: `.diff()`, `.derivative()`, `.integral()`, `.depth_integral()`, `.divergence()`, `.curl()`.

- **Statistics:** :material-arrow-right: `.weighted_mean()`, `.masked_statistic()`, `binned_statistic()`.

- **Grid Transformations:** :material-arrow-right: `.interp_to()`, `.transform_vertical_grid()`.

- **Extraction:** :material-arrow-right: `.extract_section()`, `.extract_zonal_section()`, `.extract_meridional_section()`, `.extract_mask_boundary()`.

**Interoperability with xarray**

NEMODataArray supports all standard `xarray.DataArray` operations by design. Why reinvent the wheel, right?

To achieve this, NEMODataArray will perform the operations on the underlying `xarray.DataArray` before attempting to return the result as an NEMODataArray whenever possible. Otherwise, the default return type of the operation is returned.

``` py
nemo["gridT/tos_con"].chunk({"i": 50})
```

Here, the chunking operation is performed using the `.data` property (`xarray.DataArray`) of the sea surface temperature variable, before being returned as a `NEMODataArray`:

```
<NEMODataTree 'My NEMO model'>
  <NEMODataArray 'tos_con' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>

<xarray.DataArray 'tos_con' (time_counter: 48, j: 331, i: 360)> Size: 23MB
dask.array<rechunk-merge, shape=(48, 331, 360), dtype=float32, chunksize=(1, 331, 50), chunktype=numpy.ndarray>
Coordinates:
  * time_counter   (time_counter) datetime64[ns] 384B 1976-07-02 ... 2023-07-...
    time_centered  (time_counter) datetime64[ns] 384B dask.array<chunksize=(1,), meta=np.ndarray>
    gphit          (j, i) float64 953kB dask.array<chunksize=(331, 50), meta=np.ndarray>
    glamt          (j, i) float64 953kB dask.array<chunksize=(331, 50), meta=np.ndarray>
  * j              (j) int64 3kB 1 2 3 4 5 6 7 8 ... 325 326 327 328 329 330 331
  * i              (i) int64 3kB 1 2 3 4 5 6 7 8 ... 354 355 356 357 358 359 360
```

**Method Chaining**

One of the most valuable features of a NEMODataArray is its native support for method-chaining, enabling us to build complex diagnostics, including:

``` py
nemo["gridT/thetao_con"].apply_mask(mask=my_mask).weighted_mean(dims=["i", "j"], skipna=True).plot()
```

where we apply `my_mask` to the global sea surface temperature field, calculate the horizontal grid cell area-weighted mean, and plot the resulting time-series in a single line of code.

Support for method-chaining also includes indexing operations, such as `.sel()`, `.isel()` and .`sel_like()`, and reduction operations, such as `.mean()`, `.min()` and `.max()`, which will modify the shape of the NEMODataArray.

## **Combining**

**Merging**

To combine the variables in multiple NEMODataTree objects into one larger object, we can use the `merge()` method, which takes a list of NEMODataTree objects to merge.

For example, to merge variables contained in two NEMODataTree objects `nemo` and `nemo_other` into a single NEMODataTree `nemo_merged`:

``` py
nemo_merged = nemo.merge([nemo_other], compat="no_conflicts")
```

Note, we can also pass additional keyword arguments to `xarray.merge()` alongside our list of NEMODataTree objects.

**Concatenating**

To combine NEMODataTree objects along an existing or new dimension into a larger object, we can use the `concat()` method, which takes a list of NEMODataTree objects and a dimension name or `xarray.DataArray` defining the dimension along which to concatenate variables.

For example, to concantenate two NEMODataTree objects `nemo` and `nemo_other` along the existing `time_counter` dimension:

``` py
nemo_concat = nemo.concat([nemo_other], dim="time_counter")
```

The above example works well when the two NEMODataTrees represent different periods of a time-series, but we can also use `concat()` to concatenate multiple NEMODataTrees representing ensemble members of the same simulations along a new dimension.

First we will define a new dimensions which identifies ensemble members by a unique integer, and then pass the resulting dimension `xarray.DataArray` to the `concat()` method:

``` py
ens = xr.DataArray(data=[1,2],
                   dims="ens",
                   coords={"ens": ("ens", [1,2])}
                   )

nemo_concat = nemo.concat([nemo_other], dim=ens)
```

Both the `merge()` and `concat()` methods are simply convienence wrappers to `xarray.merge()` and `xarray.concat()` functions, which handle the process of assembling the resulting `xarray.DataTree` object back into a compliant NEMODataTree.

## **Indexing**

The most basic way to access elements of a NEMODataArray is to use positional indexing (i.e., `nda[0, :, :]`) to return a subsetted NEMODataArray.

We can also index a NEMODataArray by integer indices using the `isel()` method (e.g., `nda.isel(time_counter=0)`) or by coordinate labels using the `sel()` method (e.g., nda.sel(time_counter="2025-01")).

In addition to the more familiar `sel()` and `isel()` label based selection methods above, NEMODataArray includes a `sel_like()` method to index a NEMODataArray according to the dimension index labels of another NEMODataArray or xarray.DataArray.

For example, to index the conservative temperature `thetao_con` defined on **T**-points to match a subset of the absolute salinity `so_abs` which has been indexed in both space and time:

``` py
nda = nemo["gridT/so_abs"].sel(time_counter=slice('2000-01', '2025-01'), k=1)

nemo["gridT/thetao_con"].sel_like(nda)
```

We can also index variables defined on a given grid using their longitude & latitude coordinates (i.e., `glam{t/u/v/w}(j, i)` & `gphi{t/u/v/w}(j, i)`).

To do this, we must first add these geographical variables as indexes using the `add_geoindex()` method of the NEMODataTree. For example, let's enable geographical indexing of the global parent domain **T**-points:

``` py
nemo_geo = nemo.add_geoindex(grid="gridT")
```

We can then use `sel()` to select the values of all variables defined on **T**-points nearest to (-30°E, 60°N):

``` py
nemo_geo["gridT"].dataset.sel(gphit=60, glamt=-30, method="nearest")
```

Note that we must use the `dataset` view to perform geographical indexing using our grid node and we must specify `method="nearest"`. 

## **Clipping**

Clipping is the process by which we can subset a NEMODataTree or NEMODataArray using a geographical bounding box.

There are three methods available to clip domains, grids, or variables using a bounding box defined by a tuple of the form (`lon_min, lon_max, lat_min, lat_max`)...

**Domains**

We can clip all of the grids contained in a given NEMO model domain (i.e., global parent / regional child domain) using the NEMODataTree `clip_domain()` method. For example, let's clip the global parent domain (`dom="."`) in the bounding box (-80°E, 0°E, 40°N, 80°N): 

``` py
bbox = (-80, 0, 40, 80)

nemo.clip_domain(dom=".", bbox=bbox)
```

**Grids**

Alternatively, we can clip a single grid contained in a given NEMO model domain using the NEMODataTree `clip_grid()` method. For example, let's clip the global parent **T**-grid using the same bounding box as above: 

``` py
nemo.clip_grid(grid="gridT", bbox=bbox)
```

**Variables**

Finally, we can clip a NEMODataArray using the `clip()` method. For example, let's clip the sea surface height variable `zos` defined on **T**-points in the bounding box (-40°E, 10°E, 35°N, 60°N):

``` py
bbox = (-40, 10, 35, 60)

nemo["gridT/zos"].clip(bbox=bbox)
```

## **Masking**

**Applying Masks**

Clipping methods on NEMODataTree or NEMODataArray objects generally return a subset of the original data. However, it is often useful to generate an object with the same shape as the original data, but with some elements masked.

One of the most common masking operations is to apply the land-sea mask to a variable to remove any non-NaN values used to fill land points. With NEMODataArray, we can use the `.masked` property to return the variable with the appropriate land-sea mask applied. For example, to apply the 3-dimensional `tmask` to the conservative temperature variable `thetao_con` defined on **T**-points:

``` py
nemo["gridT/thetao_con"].masked
```

Importantly, the `.masked` property can also be used with a subset of the original NEMODataArray as the internally the appropriate `mask` variable is aligned to match our NEMODataArray using `sel_like()` prior to masking.

To apply a custom mask to a given NEMODataArray variable, we can use the `apply_mask()` method. For example, let's mask the absolute salinity variable `so_abs` using a custom boolean mask `my_mask`: 

```python
nemo["gridT/so_abs"].apply_mask(mask=my_mask)
```
The advantage of using the `apply_mask()` method over xarray's in-built `DataArray.where()` method is that both our custom mask and the appropriate land-sea mask will be applied.

We can optionally drop the values of the variable where `my_mask` is `False` by passing `drop=True` to the `.apply_mask()` method.

**Creating Masks**

To define a regional mask on one of the grids comprising our NEMO ocean mesh, we can use the `NEMODataTree.mask_with_polygon()` method.

The `.mask_with_polygon()` method takes the path to a given grid in our NEMODataTree and two `lists` or `numpy.ndarrays` containing the geographical coordinates of a closed polygon. For example, we could define a mask on **T**-points corresponding to a simple bounding box (-40°E, 10°E, 35°N, 60°N) as follows:

```python
nemo.mask_with_polygon(grid="gridT", lon_poly=[-40, 10, 10, -40, -40], lat_poly=[35, 35, 60, 60, 35])
```

The example above is clearly better suited to the `NEMODataTree.clip_grid()` method, however it highlights that there are often many approaches to achieve the same result using NEMO Cookbook.

For a more typical example, let's consider how we would define a regional mask for the Labrador Sea in the North Atlantic Ocean. To help with this, NEMO Cookbook includes a selection of useful resources accessible via cloud object storage, which can be downloaded using the `nemo_cookbook.examples.get_filepaths()` convenience function.

Let's start by opening the IHO World Seas v3 polygons from cloud object storage as a Pandas DataFrame.

``` py
filepaths = nemo_cookbook.examples.get_filepaths("IHO")

df_IHO_World_Seas = pd.read_parquet(filepaths['IHO_World_Seas_v3_polygons.parquet'])
```

Next, we need to define the longitude and latitude arrays for the Labrador Sea polygon by filtering the DataFrame for the `'Labrador Sea'` entry.

``` py
lon_poly = df_IHO_World_Seas[df_IHO_World_Seas['Name'] == 'Labrador Sea']['Longitudes'].item()[0]
lat_poly = df_IHO_World_Seas[df_IHO_World_Seas['Name'] == 'Labrador Sea']['Latitudes'].item()[0]
```

Finally, we can pass these coordinate arrays to `.mask_with_polygon()` to generate a boolean mask for the Labrador Sea defined on **T**-points.

``` py
LSea_mask = nemo.mask_with_polygon(grid='gridT', lon_poly=lon_poly, lat_poly=lat_poly)
```

## **Discrete Operators**

One of the most valuable features of the NEMODataTree and NEMODataArray objects is their support for discrete, grid-aware computation, including finite differencing, differentiation, integration etc.

**Differencing**

We can calculate the discrete first-order difference of a variable along a given NEMO grid dimension (e.g., `i, j, k`) using the `.diff()` method.

For example, let's compute the difference of the sea surface temperature variable `tos_con` defined at **T**-points along the `i`-dimension:

$$\delta_{i + 1/2}[q] = q(i + 1) - q(i)$$

``` py
nemo["gridT/tos_con"].diff(dim="i")
```

where the resulting differenced NEMODataArray is defined on **U**-points.

Note that the `.diff()` method includes a `fillna` argument to provide flexibility on how to handle NaN values. By default, `fillna=False`, meaning that NaN values are not filled with zeros prior to performing finite differencing. However, in the case of velocity variables this may be inappropriate in the vicinity of coastlines and users may prefer to use `fillna=True` to impose zero-magnitude velocity components along land-sea boundaries prior to calculating derivatives.

**Differentiation**

We can also calculate the derivative of a variable along one of the dimensions (e.g., `i, j, k`) of a given NEMO model grid using the `.derivative()` method.

For example, let's compute the derivative of the sea surface temperature variable `tos_con` along the NEMO model parent domain `j` dimension:

$$\nabla q\ .\ j = \frac{1}{e_{2}} \delta_{j + 1/2}[q]$$

``` py
nemo["gridT/tos_con"].derivative(dim="j")
```

Alternatively, we can compute the derivative of the sea surface temperature variable `tos_con` along a regional subset of a global, zonally periodic domain NEMO model parent domain `i` dimension:

``` py
nemo["gridT/tos_con"].sel(i=slice(10, 100)).derivative(dim="i", iperio=False)
```

Note, in the above example, `iperio` is set to `False` overriding the zonal periodicity inherited from the NEMO model grid since the selected subset of the global domain is no longer zonally periodic.

Finally, we can calculate the vertical derivative of the absolute salinity variable `so_abs` along the `k` dimension of a given NEMO model grid:

$$\nabla q\ .\ k = \frac{1}{e_{3}} \delta_{k + 1/2}[q]$$

``` py
nemo["gridT/so_abs"].derivative(dim="k")
```

**Integration**

We can integrate a variable along one or more dimensions of a given NEMO model grid using the `NEMODataArray.integral()` method.

For example, let's compute the integral of conservative temperature variable `thetao_con` along the vertical `k` dimension in the NEMO model parent domain:

``` py
nemo["gridT/thetao_con"].integral(dims=["k"])
```

which will return an `NEMODataArray` with one less dimension than `thetao_con`, in this case `k` since we have integrated vertically.

We can also use the `.integral()` method to calculate cumulative integrals along one or more dimensions of a given NEMO model grid.

For example, to calculate the vertical meridional overturning stream function from the meridional velocity variable `vo` (*zonally integrated meridional velocity accumulated with increasing depth*):

``` py
nemo["gridV/vo"].integral(dims=["i", "k"], cum_dims=["k"], dir="+1")
```
where `dims` is a list of grid dimension names along which integration will be performed, and `cum_dims` specifies the dimensions in `dims` that should be cumulatively integrated.

The `dir` argument is used to define the direction of cumulative integration, where `dir = "+1"` means accumulating along the chosen dimension, such that grid indices are increasing. Conversely, `dir = "-1"` means that cumulative integration is performed after reversing the chosen dimension, such that grid dimensions are decreasing.

We can also pass a `mask` argument to `.integral()` to mask a variable prior to performing the integration.

**Depth Integration**

Instead of estimating a vertical integral by integrating over the `Nk` uppermost vertical levels, such that `deptht[Nk]` is closest to our target depth, we can use the `NEMODataArray.depth_integral()` method to perform vertical integration between two depth surfaces.

For example, to compute the vertical integral of conservative temperature variable `thetao_con` in the upper 100 m in the NEMO model parent domain:

``` py
nemo["gridT/thetao_con"].depth_integral(limits=(0, 100))
```

where `limits` is a tuple of the form (depth_min, depth_max) where depth_min and depth_max are the lower and upper limits of vertical integration, respectively.

## **Statistics**

**Weighted Average**

We can use the `NEMODataArray.weighted_mean()` method to calculate a grid-aware weighted average of a variable defined on a NEMO model grid.

For example, to compute the grid cell area-weighted mean of the sea surface temperature variable `tos_con` in a NEMO model nested child domain:

``` py
nemo["gridT/1_gridT/tos_con"].weighted_mean(dims=["i", "j"], skipna=True)
```

where `dims` represents the dimensions of the NEMO model grid to average over. Here, `dims=["i", "j"]` is equivalent to computing the mean of variable `tos_con` using the horizontal cell area of **T**-points (i.e., `e1t * e2t`) as weights.

**Masked Statistics**

We can also use the `NEMODataArray.masked_statistic()` method to calculate an aggregated statistic from only the grid cells contained inside a geographical polygon.

For example, to compute the grid cell area-weighted mean sea surface temperature `tos_con` for a region enclosed in a polygon defined by `lon_poly` and `lat_poly` in a NEMO model nested child domain:

``` py
nemo["gridT/1_gridT/tos_con"].masked_statistic(lon_poly,
                                               lat_poly,
                                               statistic="weighted_mean",
                                               dims=["i", "j"]
                                               )
```

where `dims` represent the dimensions of the NEMO model grid used for aggregation. Here, combining `statistic="weighted_mean"` and `dims=["i", "j"]` is equivalent to computing the mean of variable `tos_con` using the horizontal cell area of **T**-points as weights.

**Binned Statistics**

We can also calculate aggregated statistics of a variable binned according to the values of one or more other variables, using the `NEMODataTree.binned_statistic()` method. 

This is a generalisation of a histogram function, enabling the computation of the `sum`, `mean`, `median`, or other statistics of the values assigned to each bin.

For example, let's compute the total sea water volume contained each isopycnal layer in discrete potential density (`sigma0`) coordinates:

``` py
sigma0_bins = np.arange(22, 29.05, 0.05)

nemo["gridT/volcello"] = nemo.cell_volume(grid="gridT")

nemo.binned_statistic(grid="gridT",
                      vars=["sigma0"],
                      values="volcello",
                      keep_dims=["time_counter"],
                      bins=[sigma0_bins],
                      statistic="nansum",
                      )
```

where `vars` is a list of the names of variables to be binned using the bin edges passed to `bins`, and `values` is the name of the variable over which the `statistic` will be performed once values have been grouped into each bin.

We can use `keep_dims` to specify the dimension labels of the `values` variable to retain. Here, using `keep_dims=["time_counter"]` will return the total volume of sea water in each potential density bin for each time-slice of available NEMO model output.

## **Transformations**

When performing computations using NEMO Cookbook, we often need to transform one or more NEMODataArrays to a neighbouring horizontal grid or a new vertical grid using interpolation.  

**Transforming variables to neighbouring horizontal grid**

We can use the `NEMODataArray.interp_to()` method to linearly interpolate a variable defined on a given NEMO horizontal grid to a neighbouring grid.

For example, let's interpolate the conservative temperature variable `thetao_con` defined on **T**-points to neighbouring **V**-points in a NEMO model parent domain:

``` py
nemo["gridT/thetao_con"].interp_to(to="V")
```

We can also interpolate variables defined on **U** and **V**-points to either scalar or vector grid points. Unlike interpolating scalar variables defined on **T**-points, this is achieved by linearly interpolating the grid cell face area-weighted flux (e.g., `u * e2u * e3u`) onto the target grid, before then normalising by the grid cell face area defined on the target horizontal grid.

For example, to interpolate the zonal wind stress defined on **U**-points to neighbouring **V**-points in a NEMO model parent domain and store this in the **V**-grid node of our NEMODataTree:

``` py
nemo['gridV/tauuo'] = nemo["gridU/tauuo"].interp_to(to="V")
```

**Transforming variables to a new vertical grid**

We can use the `NEMODataArray.transform_vertical_grid()` method to transform a variable defined on a given NEMO model vertical grid to a new vertical grid using conservative interpolation.

For example, let's transform the conservative temperature variable `thetao_con` defined in a NEMO model parent domain from it's native 75 unevenly-spaced z-levels to regularly spaced z-levels at 200 m intervals:

```python
e3t_target = xr.DataArray(np.repeat(200.0, 30), dims=['k_new'])

nemo["gridT/thetao_con"].transform_vertical_grid(e3_new = e3t_target)
```

where `e3_new` represents the time-invariant vertical grid cell thicknesses defing the vertical grid onto which the variable `var` will be conservatively interpolated. 

There are some important points to remember when transforming variables onto new vertical grids with `NEMODataTree`:

- New vertical grid cell thicknesses `e3_new` must sum to at least the maximum depth of the original vertical grid cell thicknesses (e.g., e3t).

- Currently, `e3_new` must be a 1-dimensional `xarray.DataArray` with dimension 'k_new'.

- The output `xarray.Dataset` will contain multi-dimensional `xarray.DataArrays` for both the vertically remapped variable `var(time_counter, k_new, j, i)` and the vertical grid cell thicknesses `e3t_new(time_counter, k_new, j, i)` (updated to explicitly account for partial grid cells above the seafloor).

## **Plotting**

**Basic Plotting**

Since NEMODataArray supports all standard `xarray.DataArray` operations by design, we can use the `.plot()` method to plot time-series, maps and histograms of a variable defined on a NEMO model grid.

For example, let's plot the time-mean of the sea surface salinity variable `sos_abs`:

``` py
nemo["gridT/sos_abs"].mean(dim="time_counter").plot()
```

Note that this will return a `xarray.plot.pcolormesh()` of the time-mean sea surface salinity field plotted using the horizontal (`i, j`) dimensions of the NEMO model **T**-grid.

* For more details on plotting using xarray, see the [Plotting section of the xarray User Guide](https://docs.xarray.dev/en/stable/user-guide/plotting.html).

**Plotting in Geographical Coordinates**

To plot a 2-dimensional slice of a variable as a Cartopy `GeoQuadMesh` using its longitude & latitude coordinates (i.e., `glam{t/u/v/w}(j, i)` & `gphi{t/u/v/w}(j, i)`), we can use the `NEMODataArray.geoplot()` method.

For example, to create a simple geographical plot of sea surface temperature variable `tos_con`:

``` py
nemo["gridT/tos_con"].isel(time_counter=0).geoplot()
```

Similarly to `xarray.DataArray.plot`, the `.geoplot()` method allows for significant customisation of geographical plots.

For example, to create a geographical plot of the sea ice concentration variable `siconc` using a North Polar Stereographic projection with a bounding box (-180°E, 180°E, 50°N, 90°N):

``` py
(nemo["gridT/siconc"]
 .isel(time_counter=0)
 .geoplot(projection=ccrs.NorthPolarStereo(),
          vmin=0, vmax=1,
          extent=(-180, 180, 50, 90),
          cmap='Blues_r',
          clabel_kwargs={'label': 'Sea Ice Concentration [fraction of unity]'}
          )
 )
```

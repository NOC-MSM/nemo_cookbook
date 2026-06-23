# What's New

## v2026.06.01 - 23rd June 2026

### New Features

* **Added `extract_zonal_section()` method to `NEMODataTree`:**
     * Enables users to extract an approximately zonal section at a chosen latitude from a NEMODataTree.
     * Hydrographic section will be located at the constant j-coordinate whose average latitude (following selection between lon_min and lon_max) is closest to the given latitude.

* **Added `from_xesmf()` accessor for`NEMODataArray` enabling users to import an xESMF-compatible `xarray.Dataset` to a NEMO model variable.**

* **Add reference vertical scale factors `e3{t/u/v/w}_0` and water column heights `h{t/u/v/w}_0` to `NEMODataTree` construction.**

### New Recipes

* **Two additional NEMO Cookbook Recipes:**
   1. Extracting Zonal Hydrographic Sections - `recipe_extract_rapid.ipynb`
   2. Working with CMORISED NEMO ocean model outputs - `recipe_cmip6.ipynb`

### Breaking Changes

* **`extract_section()` expected parameters `lon`, `lat` renamed to `lon_section`, `lat_section`.**

### Bug Fixes

* **Fix interp_to(), depth_integral() and transform_vertical_grid() to preserve core dimensions when size(time_counter)=1  - @robertsmalcolm**
* **Update `_check_grid_dims()` to handle singleton dims in domain_cfg**
* **Update `processing.py` to improve performance of merging NEMO gridT and icemod datasets - @atb299**

### Documentation

* **Added required core dimensions for `NEMODataTree` constructors to User Guide & How To... Guide.**
* **Added two new NEMO Cookbook recipes to documentation.**

### Pull Requests

* 29 [feature] add from_xesmf method by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/30
* 32 feature add extract zonal section method to nemodatatree by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/33
* 34 feature add cmorised nemodatatree recipe by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/35


**Full Changelog**: https://github.com/NOC-MSM/nemo_cookbook/compare/v2026.04.0b1...v2026.06.01


# What's New

## v2026.04.b1 - 30th April 2026

### New Features

* **Introduction of `from_icechunk()` and `from_zarr()` constructors  for `NEMODataTree`:**
     * Enables users to open a `NEMODataTree` from an `xarray.DataTree` stored in a hierarchical Zarr store or Icechunk store where each group contains one NEMO model grid node dataset.
     * Definition of `NEMODataTree` grid nodes is now validated to ensure expected coordinates, dimensions, land-sea masks and grid cell scale factors are included.

* **Addition of `to_xesmf()` accessor for`NEMODataArray` enabling users to export a NEMO model variable to an xESMF-compatible `xarray.Dataset`.**

### New Recipes

* **Three additional NEMO Cookbook Recipes:**
   1. Regridding using xESMF- `recipe_xesmf.ipynb`
   2. Sea Ice Diagnostics - `recipe_sea_ice.ipynb`
   3. Extracting mask boundaries - `recipe_extract_mask_boundary.ipynb`

### Breaking Changes

* **`read_mask=True` expects `t/u/v/fmaksutil` 2-dimensional land-sea unique point masks must be included in the input domain_cfg dataset, otherwise use `read_mask=False`**

### Bug Fixes

* **Fix `.extract_mask_boundary()` and `extract.py` to support nested NEMO model domains where coordinates are not `glamb`, `gphib`**
* **Improved performance of `NEMODataArray` validation and `set_like()` methods.

### Documentation

* **Update How To... guide to include `to_xesmf` usage to produce xESMF-compatible `xarray.Datasets` from `NEMODataArrays`.**
* **Added Marimo and Google Colab notebook previews of all existing NEMO Cookbook recipes, including adding badges to table of recipes in docs.**

### Pull Requests
* 16 refactor improve performance of nemodataarray methods by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/23
* 19 add marimo notebook support for recipes by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/24
* 17 feature add from icechunk nemodatatree constructor by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/25
* 22 feature add accessor to support xesmf regridding by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/26
* 20 add new nemo cookbook recipes by @oj-tooth in https://github.com/NOC-MSM/nemo_cookbook/pull/27


**Full Changelog**: https://github.com/NOC-MSM/nemo_cookbook/compare/v2026.03.0b1...v2026.04.0b1


## v2026.03.b1 - 30th March 2026

### New Features

* **Introduction of `NEMODataArray` object, including manual and `NEMODataTree` constructors:**

    * `NEMODataArray` properties to support grid-aware computation (e.g., `.mask`, `.metrics`, `.grid` etc.).

    * `NEMODataArray` methods:
         - **Masking:** `.apply_mask()`.
         - **Selections:** `.sel_like()`.
         - **Grid Operators**: `.diff()`, `.derivative()`, `.integral()`, `.depth_integral()`.
         - **Statistics:** `.weighted_mean()`, `.masked_statistic()`.
         - **Interpolation:** `.interp_to()`.
         - **Grid Transformations:**`.transform_vertical_grid()`.

* **Addition of `.depth_integral()` method to integrate a variable defined on a given NEMO model grid in depth coordinates between two depth limits.**

* **Addition of `.weighted_mean()` method to perform a grid-aware weighted mean of a variable defined on a NEMO model grid.**

* **Update to `NEMODataTree` constructors:**

   * `NEMODataTree` supports NEMO model outputs using the linear free-surface approximation (i.e., time-independent vertical scale factors) via `key_linssh=True`.

* **Defined Pixi workspace & environments (`pixi.toml`) to support NEMO Cookbook development.**

### New Recipes

* **Two additional NEMO Cookbook Recipes:**
   1. Upper Ocean Heat Content - `recipe_heat_content.ipynb`
   2. Meridional Ocean Heat Transport - `recipe_heat_transport.ipynb`

### Breaking Changes

* **`.transform_to()` method to linearly interpolate a variable to a neighbouring NEMO grid renamed to `interp_to()`.**

* **Using dictionary syntax to access a variable from a `NEMODataTree` no longer returns a land-sea masked variable. Instead, an unmasked `NEMODataArray` is returned, which can be subsequently masked using the `.masked` attribute**.

* **Started Deprecation Cycle for legacy `NEMODataTree` methods & included deprecation warnings for affected methods. These methods will be removed from NEMO Cookbook in v2026.05 later this year.**

### Bug Fixes

* **Fix `.add_geoindex()` using `xoak.SklearnGeoBallTreeAdapter` & added `xoak` dependency.**
* **Fix `.extract_section()` to improve boundary indexing and add land-sea masks**
* **Fix nested test class `TestNEMODataTreeUtils()` in `test_nemodatatree.py`**
* **Fix potential NameError in `create_section_polygon()` where section endpoints share same latitude.**
* **Fix TypeError (keep_dims=None) in compute_binned_statistic() in stats.py.**
* **Fix alignment error in `interpolate_grid()` utility function.**

### Documentation

* **Reorganise User Guide to add NEMO Fundamentals section and improve `NEMODataArray` description.**
* **Update contributing guidance in docs to add getting started with Pixi & GitHub.**
* **Add .ipynb template and GitHub Issue template for NEMO Cookbook recipes.**
* **Update How To... guide to include `NEMODataArray` usage.**

**Full Changelog**: https://github.com/NOC-MSM/nemo_cookbook/compare/v2026.01.0b1...v2026.03.0b1


## v2026.01.b1 - 18th January 2026

### New Features

* **Introduction of `NEMODataTree` object, including `.from_paths()` and `.from_datasets()` constructors.**
    * `NEMODataTree` methods to calculate model grid cell properties (e.g., `.cell_area()`).
    * `NEMODataTree` methods to perform grid-aware scalar and vector operations (e.g., `.gradient()`, `.integral()`.
    * `NEMODataTree` methods to calculate grid-aware masked and binned statistics (e.g., `.masked_statistic()`, `.binned_statistic()`)
    * `NEMODataTree` methods to transform model grid coordinates (e.g., `.transform_to()`)

### New Recipes

* **NEMO Cookbook Recipes, including**
    1. Meridional overturning stream function in an arbitrary tracer coordinates.
    2. Meridional overturning stream function in depth coordinates (z/z*).
    3. Meridional heat & salt transports.
    4. Surface-forced water mass transformation in potential density coordinates.
    5. Volume census in T-S coordinates.
    6. Masked statistics using bounding boxes and polygons.
    7. Extracting volume transports and properties along the Overturning in the Subpolar North Atlantic array.
    8. Vertical coordinate transformations.
    9. Barotropic stream functions. 

**Full Changelog**: https://github.com/NOC-MSM/nemo_cookbook/commits/v2026.01.0b1
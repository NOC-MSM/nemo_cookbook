# What's New

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
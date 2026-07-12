# Getting Started

<img src="./assets/icons/noc_logo_dark.png" alt="drawing" width="150"/>
<img src="./assets/icons/NEMO_Cookbook_Logo.png" alt="drawing" width="150"/>

**Welcome to the documentation for NEMO Cookbook :wave:**

## What is NEMO Cookbook? :cook: :book:
NEMO Cookbook extends the familiar xarray data model with grid-aware data structures designed for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide a collection of recipes implementing the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

Each recipe uses the `NEMODataTree` and `NEMODataArray` structures to leverage [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries (think of these are your cooking utensils) to calculate a diagnostic with NEMO ocean model outputs (i.e., the raw ingredients - that's where you come in!).

## NEMO Data Structures

At the core of NEMO Cookbook are two abstractions:

* **NEMODataTree** → a hierarchical container for organising NEMO model outputs extending the `xarray.DataTree`.
* **NEMODataArray** → a NEMO grid-aware extension of `xarray.DataArray`.

If you already use `xarray`, NEMO Cookbook should feel immediately natural:

* `NEMODataTree` builds directly on `xarray.DataTree`.
* `NEMODataArray` behaves like `xarray.DataArray`.
* All standard `xarray` operations are still available (let's not reinvent the wheel!).

What’s new is that these objects understand the NEMO grid, meaning  you no longer need to manually track:

* which NEMO model grid a variable belongs to (e.g., T, U, V, F, W).
* how variables relate across NEMO model grids.
* where to find grid scale factors.
* how to consistently apply grid-aware operations.

### NEMODataTree
`NEMODataTree` is an extension of the `xarray.DataTree` object and an alternative to the [**xgcm grid**](https://xgcm.readthedocs.io/en/latest/) object.

`NEMODataTree` organises NEMO model outputs into a single, coherent data structure, where each node in the tree represents an `xarray.Dataset` of variables from one NEMO model grid. This allows us to:

* Store output variables defined on NEMO T, U, V, W, F grids using the model’s native (i, j, k) curvilinear coordinate system.
* Analyse parent, child and grandchild domains of nested configurations using a single DataTree.
* Pre-process model outputs (i.e., removing ghost points and generating t/u/v/f masks without needing a mesh_mask file).

### NEMODataArray
`NEMODataArray` extends `xarray.DataArray` to give each variable knowledge of its:

* NEMO model grid location (e.g., T, U, V, W, F)
* parent `NEMODataTree`
* associated NEMO grid metrics (grid scale factors)

This knowledge enables reproducible grid-aware computation. For example, a `NEMODataArray` can be used to:

* Automatically access correct grid metrics.
* Apply operators (e.g., derivative, integral) as formulated in NEMO.
* Calculate grid-aware diagnostics, including masked & binned statistics.
* Perform vertical grid coordinate transformations via conservative interpolation.

*Crucially, this happens without changing how you write **xarray** code — you still work with labeled arrays, but with far more NEMO understanding behind the scenes.*

---

## Installation :hammer:

We recommend that users install the latest release of **NEMO Cookbook** into a new virtual environment via GitHub:

```{bash}
pip install nemo_cookbook
```

Alternatively, users can install **NEMO Cookbook** (including the latest commits) via GitHub:

```{bash}
pip install git+https://github.com/NOC-MSM/nemo_cookbook.git
```

??? tip "Helpful Tip..."

    * **We strongly recommend setting-up a virtual environment before installing nemo_cookbook with pip.**

    The simplest way to create a new virtual environment is to use [venv](https://docs.python.org/3/library/venv.html):

    ```sh
    python3.13 -m venv "env_nemo_cookbook"
    ```

    Alternatively, using an existing [miniforge](https://github.com/conda-forge/miniforge) installation:

    ```sh
    conda env create -f environment.yml
    ```

---

## Quick Start :rocket:

**NEMO Cookbook** is designed to make complex grid-aware analysis of NEMO model outputs feel as simple as working with standard `xarray` objects.

### Pre-Processing Made Simple

* Create a `NEMODataTree` from the National Oceanography Centre's eORCA1 JRA55v1 ocean sea-ice hindcast simulation stored in Analysis-Ready Cloud Optimised (**ARCO**) Zarr stores...

```python
# Open eORCA1 NEMO domain_cfg:
ds_domain = xr.open_zarr("https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca1-jra55v1/domain_cfg", consolidated=True, chunks={})

# Open eORCA1 NEMO gridT dataset:
ds_gridT = xr.open_zarr("https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca1-jra55v1/T1y")

# Define dictionary of grid datasets defining eORCA1 parent model domain:
datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT}}

# Initialise new NEMODataTree with zonally periodic parent domain north-folding on F-points:
nemo = NEMODataTree.from_datasets(datasets=datasets, iperio=True, nftype="F", read_mask=True)
```

### Exploring NEMO Model Outputs

* Access land-sea masked conservative temperature variable defined on NEMO model T-grid points as a `NEMODataArray`...

```python
nemo["gridT/thetao_con"].masked
```

* Access NEMO grid scale factors of zonal velocity variable defined on NEMO model U-grid points...

```python
nemo["gridU/uo"].metrics
```

* Access familiar `xarray` operations...

```python
nemo["gridT/tos_con"].mean(dim="time_counter")
```

### Calculating Grid-Aware Diagnostics

* Calculate meridional ocean heat transport using a constant reference density `rho0` and specific heat capacity of seawater `cp0`...

```python
(rho0 * cp0 * nemo["gridT/thetao_con"].transform_to(to='V') * nemo["gridV/vo"]).integral(dim=["i", "k"])
```

* Transform conservative temperature variable `thetao_con` defined on a NEMO model T-point from it's native 75 z*-levels to regularly spaced geopotential levels at 200 m intervals...

```python
# Define target vertical grid cell thicknesses:
e3t_target = xr.DataArray(np.repeat(200.0, 30), dims=['k_new'])

# Transform conservative temperature to new vertical coordinate system:
nemo["gridT/thetao_con"].transform_vertical_grid(e3_new = e3t_target)
```

---

## **Funding**
The ongoing development of NEMO Cookbook is funded by the following projects: 

- **AtlantiS**: [Atlantic Climate and Environment Strategic Science](https://atlantis.ac.uk)
- **ARIA - PROMOTE**: [Progressing earth system Modelling for Tipping Point Early warning systems](https://aria.org.uk/opportunity-spaces/scoping-our-planet/forecasting-tipping-points/)
- **EPOC**: [Explaining & Predicting the Ocean Conveyor](https://epoc-eu.org)

---

### Next Steps...

* To learn more about **NEMO Data Structures**, see the [User Guide] and [How To] pages - this is an especially starting point for new NEMO users!

* To get started working with the recipes in the **NEMO Cookbook**, visit the to [Recipes] page.

* For those looking for more detailed documentation, explore the [API Reference].

* To contribute your own recipes to **NEMO Cookbook**, see the [Contributing] page

[User Guide]: user_guide.md
[Recipes]: recipes.md
[How To]: howto.md
[API Reference]: reference.md
[Contributing]: contributing.md

<p align="left">
    <img src="./docs/docs/assets/icons/NEMO_Cookbook_Logo.png" alt="Logo" width="220" height="100">
</p>

<p align="left">
<strong>Reproducible analysis of NEMO ocean general circulation model outputs using xarray.</strong>
</a>
<br />
<br />
<a href="https://noc-msm.github.io/nemo_cookbook/"> <strong>Documentation</strong></a>
:rocket:
<a href="https://github.com/NOC-MSM/nemo_cookbook/issues"><strong>Report an Issue</strong></a>
</p>

## **About**

NEMO Cookbook extends the familiar xarray data model with grid-aware data structures designed for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide a collection of recipes implementing the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

Each recipe uses the `NEMODataTree` and `NEMODataArray` structures to leverage [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries (think of these are your cooking utensils) to calculate a diagnostic with NEMO ocean model outputs (i.e., the raw ingredients - that's where you come in!).

## **NEMO Data Structures**

At the core of NEMO Cookbook are two abstractions:

* NEMODataTree → a hierarchical container for organising NEMO model outputs extending the `xarray.DataTree`.
* NEMODataArray → a NEMO grid-aware extension of `xarray.DataArray`.

If you already use `xarray`, NEMO Cookbook should feel immediately natural:

* `NEMODataTree` builds directly on `xarray.DataTree`.
* `NEMODataArray` behaves like `xarray.DataArray`.
* All standard `xarray` operations are still available!

What’s new is that these objects understand the NEMO grid, meaning  you no longer need to manually track:

* which NEMO model grid a variable belongs to (e.g., T, U, V, F, W).
* how variables relate across NEMO model grids.
* where to find grid scale factors.
* how to consistently apply grid-aware operations.

### **NEMODataTree**
`NEMODataTree` is an extension of the `xarray.DataTree` object and an alternative to the [**xgcm grid**](https://xgcm.readthedocs.io/en/latest/) object.

`NEMODataTree` organises NEMO model outputs into a single, coherent data structure, where each node in the tree represents an `xarray.Dataset` of variables from one NEMO model grid. This allows us to:

* Store output variables defined on NEMO T, U, V, W, F grids using the model’s native (i, j, k) curvilinear coordinate system.
* Analyse parent, child and grandchild domains of nested configurations using a single DataTree.
* Pre-process model outputs (i.e., removing ghost points and generating t/u/v/f masks without needing a mesh_mask file).

### **NEMODataArray**

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

## **Getting Started**

### **Installation**

Users are recommended to installing **NEMO Cookbook** into a new virtual environment via GitHub:

```{bash}
pip install git+https://github.com/NOC-MSM/nemo_cookbook.git
```

Alternatively, users can clone the latest version of the nemo_cookbook repository using Git:
```{bash}
git clone git@github.com:NOC-MSM/nemo_cookbook.git
```

Then, install the dependencies in a new conda virtual environment and pip install **NEMO Cookbook** in editable mode:
```{bash}
cd nemo_cookbook

conda env create -f environment.yml
conda activate env_nemo_cookbook

pip install -e .
```

## **Usage**

NEMO Cookbook is designed to make complex grid-aware analysis of NEMO model outputs feel as simple as working with standard `xarray` objects.

### Pre-Processing Made Simple

### Exploring NEMO Model Outputs


### Calculating Grid-Aware Diagnostics

### Scalable from Laptops to Clusters


## **Documentation**

To learn more about NEMO Cookbook & to start exploring our current recipes, visit our documentation [**here**](https://noc-msm.github.io/nemo_cookbook/).

## **Recipes**

NEMO Cookbook recipes are Jupyter Notebooks available to view statically in our [**documentation**](https://noc-msm.github.io/nemo_cookbook/recipes/) or download and edit via the `recipes/` directory:

#### **Available Recipes:**

- Meridional overturning stream function in an arbitrary tracer coordinates.

- Meridional overturning stream function in depth coordinates (z/z*).

- Upper ocean heat content.

- Meridional heat & salt transports.

- Surface-forced water mass transformation in potential density coordinates.

- Volume census in T-S coordinates.

- Masked statistics using bounding boxes and polygons.

- Extracting volume transports and properties along the Overturning in the Subpolar North Atlantic array.

- Vertical coordinate transformations.

- Barotropic stream functions.

#### **Recipes In Development:**

- Meridional overturning stream functions in depth coordinates (MEs).

- Mixed layer heat content. 

- Sea ice diagnostics.

- Vorticity diagnostics.

## **Funding**
Development of NEMO Cookbook has been funded by the following projects: 

- **AtlantiS**:
- **ARIA PROMOTE**: 
- **EPOC**: Explaining & Predicting the Ocean Conveyor

## **Contact**

Ollie Tooth (oliver.tooth@noc.ac.uk)
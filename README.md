<p align="left">
    <img src="./docs/docs/assets/icons/NEMO_Cookbook_Logo.png" alt="Logo" width="220" height="100">
</p>

<p align="left">
<strong>Recipes for reproducible analysis of NEMO ocean general circulation model outputs using xarray.</strong>
</a>
<br />
<br />
<a href="https://noc-msm.github.io/nemo_cookbook/"> <strong>Documentation</strong></a>
:rocket:
<a href="https://github.com/NOC-MSM/nemo_cookbook/issues"><strong>Report an Issue</strong></a>
</p>

## **About**

NEMO Cookbook is a collection of recipes for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide Python implementations of the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

## **NEMODataTree**

NEMO Cookbook introduces the `NEMODataTree` structure, which is an extension of the `xarray.DataTree` object and an alternative to the [**xgcm**](https://xgcm.readthedocs.io/en/latest/) grid object.

`NEMODataTree` enables users to:

* Store output variables defined on NEMO T/U/V/W grids using the model’s native (i, j, k) curvilinear coordinate system.
* Analyse parent, child and grandchild domains of nested configurations using a single DataTree.
* Pre-process model outputs (i.e., removing ghost points and generating t/u/v/f masks without needing a mesh_mask file).
* Perform scalar (e.g., gradient) and vector (e.g., divergence, curl) operations as formulated in NEMO.
* Calculate grid-aware diagnostics, including masked & binned statistics.
* Perform vertical grid coordinate transformations via conservative interpolation. 

Each recipe uses `NEMODataTree` to leverage [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries (think of these are your cooking utensils) to calculate a diagnostic with NEMO ocean model outputs (the raw ingredients - that's where you come in!).

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

## **Documentation**

To learn more about NEMO Cookbook & to start exploring our current recipes, visit our documentation [**here**](https://noc-msm.github.io/nemo_cookbook/).

## **Recipes**

#### **Available:**

The following recipes are available [**here**](https://noc-msm.github.io/nemo_cookbook/recipes/):

- Meridional overturning stream function in an arbitrary tracer coordinates.

- Meridional overturning stream function in depth coordinates (z/z*).

- Meridional heat & salt transports.

- Surface-forced water mass transformation in potential density coordinates.

- Volume census in T-S coordinates.

- Masked statistics using bounding boxes and polygons.

- Extracting volume transports and properties along the Overturning in the Subpolar North Atlantic array.

- Vertical coordinate transformations.

- Barotropic stream functions.

#### **In Development:**

- Meridional overturning stream functions in depth coordinates (MES).

- Ocean heat content & mixed layer heat content. 

- Sea ice diagnostics.

- Vorticity diagnostics.

## **Contact**

Ollie Tooth (oliver.tooth@noc.ac.uk)
# **Getting Started**

<img src="./assets/icons/noc_logo_dark.png" alt="drawing" width="150"/>
<img src="./assets/icons/NEMO_blue.png" alt="drawing" width="150"/>

**Welcome to the documentation for the NEMO Cookbook :wave:**

## What is the NEMO Cookbook? :cook: :book:
NEMO Cookbook is a collection of recipes for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide Python implementations of the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

NEMO Cookbook introduces the `NEMODataTree` structure, which is an extension of the `xarray.DataTree` object and an alternative to the [**xgcm**](https://xgcm.readthedocs.io/en/latest/) grid object. To learn more about the `NEMODataTree`, see our User Guide.

Each recipe uses the `NEMODataTree` to leverage the [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries (think of these are your cooking utensils) to calculate one or more diagnostics using NEMO ocean model outputs (the raw ingredients - that's where you come in!).

---

## Quick Start :rocket:

### Installation

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

??? tip "Helpful Tip..."

    * **We strongly recommend setting-up a virtual environment before installing nemo_cookbook with pip.**

    The simplest way to create a new virtual environment is to use venv:

    ```sh
    python3.13 -m venv "env_nemo_cookbook"
    ```

    Alternatively, using an existing miniconda or miniforge installation:

    ```sh
    conda env create -f environment.yml
    ```

---

### **Recipes**

#### **Available:**

The following recipes are available on the **Recipes** page:

1. Meridional overturning stream function in an arbitrary tracer coordinates.

2. Meridional overturning stream function in depth coordinates (z/z*).

3. Meridional heat & salt transports.

4. Surface-forced water mass transformation in potential density coordinates.

5. Volume census in T-S coordinates.

6. Masked statistics using bounding boxes and polygons.

7. Extracting volume transports and properties along the Overturning in the Subpolar North Atlantic array.

8. Vertical coordinate transformations.

#### **In Development:**

1. Barotropic stream functions.

2. Meridional overturning stream functions in depth coordinates (MES).

3. Ocean heat content & mixed layer heat content. 

4. Sea ice diagnostics.

5. Vorticity diagnostics.

### Learning More...

To learn more about the NEMO Cookbook, including how to contribute your own recipes, see the [NEMODataTree] user guide and [Contributing] pages.

[NEMODataTree]: user_guide.md
[Contributing]: contributing.md

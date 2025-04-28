# **Getting Started**

**Welcome to the documentation for the NEMO Cookbook :wave:**

## What is the NEMO Cookbook? :cook: :book:
**NEMO Cookbook** is a collection of Python recipes for performing reproducible analyses of [**NEMO**](https://www.nemo-ocean.eu) ocean general circulation model outputs.

Our aim is to provide Python implementations of the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS/tree/master) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

Importantly, the **NEMO Cookbook** does aim to be a generic ocean circulation model analysis framework such as [**xgcm**](https://xgcm.readthedocs.io/en/latest/). As such, recipes do not require users to generate grid objects in advance of their calculations; only the necessary NEMO grid variables are required similar to CDFTOOLS.

Each recipe comprises of one or more functions built using the [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries (think of these are your cooking utensils!) & the raw ingredients (ocean model outputs) - that's where you come in!

---

## Quick Start :rocket:

### Installation

There are currently two options available to install the **NEMO Cookbook**:

**Method 1: Downloading & Installing with GitHub**

First, clone the latest version of the nemo_cookbook repository using Git:
```{bash}
git clone git@github.com:NOC-MSM/nemo_cookbook.git
```

Next, install the dependencies in a new conda virtual environment before pip installing **NEMO Cookbook** in editable mode:
```{bash}
cd nemo_cookbook

conda env create -f environment.yml

conda activate env_nemo_cookbook

pip install -e .
```

**Method 2: Installing Directly with pip**

```{bash}
pip install git+git@github.com:NOC-MSM/nemo_cookbook.git
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

### Current Recipes

The following recipes are currently available in the **NEMO Cookbook** and be explored on the **Recipes** page:

1. Meridional Overturning Stream Function in an arbitrary tracer coordinates.

2. Meridional Overturning Stream Function in depth coordinates (z/z*).

3. Meridional Heat & Salt Transports.

4. Surface-Forced Overturning Stream Functions in potential density coordinates.

5. Volume census in temperature - salinity coordinates.

6. Extracting the Overturning in the Subpolar North Atlantic array.

### Recipe Wishlist

The following recipes are in development:

1. Barotropic Stream Functions.

2. Meridional Overturning Stream Functions in depth coordinates (MES).

3. Ocean Heat Content & Mixed Layer Heat Content. 

4. Sea Ice Diagnostics.

5. Vorticity Diagnostics.

### Learning More...

To learn more about the NEMO Cookbook, including how to contribute your own recipes, see the [User Guide] and [Contributing] pages.

[User Guide]: user_guide.md
[Contributing]: contributing.md

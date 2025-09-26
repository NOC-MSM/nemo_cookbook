# Getting Started

<img src="./assets/icons/noc_logo_dark.png" alt="drawing" width="150"/>
<img src="./assets/icons/NEMO_Cookbook_Logo.png" alt="drawing" width="150"/>

**Welcome to the documentation for the NEMO Cookbook :wave:**

## What is the NEMO Cookbook? :cook: :book:
NEMO Cookbook is a collection of recipes for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide Python implementations of the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MEs).

## NEMODataTree

NEMO Cookbook utilises the `NEMODataTree` object, which is an extension of the `xarray.DataTree` and an alternative to the [**xgcm**](https://xgcm.readthedocs.io/en/latest/) grid object.

`NEMODataTree` enables users to:

* Store output variables defined on NEMO T/U/V/W grids using the model’s native (i, j, k) curvilinear coordinate system.

* Analyse parent, child and grandchild domains of nested configurations using a single DataTree.

* Pre-process model outputs (i.e., removing ghost points and generating t/u/v/f masks without needing a mesh_mask file).

* Perform scalar (e.g., gradient) and vector (e.g., divergence, curl) operations as formulated in NEMO.

* Calculate grid-aware diagnostics, including masked & binned statistics.

* Perform vertical grid coordinate transformations via conservative interpolation. 

Each recipe in the **NEMO Cookbook** uses `NEMODataTree` to leverage [**xarray**](https://xarray.dev), [**flox**](https://flox.readthedocs.io/en/latest/) & [**dask**](https://www.dask.org) libraries to calculate a diagnostic with NEMO ocean model outputs.

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

### Next Steps...

* To learn more about **NEMODataTree**, see the [User Guide] and [How To] pages - this is an especially starting point for new NEMO users!

* To get started working with the recipes in the **NEMO Cookbook**, visit the to [Recipes] page.

* For those looking for more detailed documentation, explore the [NEMODataTree API].

* To contribute your own recipes to **NEMO Cookbook**, see the [Contributing] page

[User Guide]: nemodatatree.md
[Recipes]: recipes.md
[How To]: howto.md
[NEMODataTree API]: reference.md
[Contributing]: contributing.md

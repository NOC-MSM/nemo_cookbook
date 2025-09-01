
<h1 align="center">NEMO Cookbook</h1>
<p align="center">
<strong>Recipes for reproducible analysis of NEMO ocean general circulation model outputs using xarray.<strong>
</a>
<br />
<br />
-
<a href="https://noc-msm.github.io/nemo_cookbook/"><strong>Documentation</strong></a>
-
<a href="https://github.com/NOC-MSM/nemo_cookbook/issues"><strong>Report an Issue</strong></a>
-
</p>

## **About**

NEMO Cookbook is a collection of recipes for performing reproducible analyses of the Nucleus for European Modelling of the Ocean ([**NEMO**](https://www.nemo-ocean.eu)) ocean general circulation model outputs.

Our aim is to provide Python implementations of the post-processing & analysis functions available in [**CDFTOOLS**](https://github.com/meom-group/CDFTOOLS) alongside new diagnostics (e.g., surface-forced water mass transformation), which are compatible with generalised vertical coordinate systems (e.g., MES).

NEMO Cookbook introduces the `NEMODataTree` structure, which is an extension of the `xarray.DataTree` object and an alternative to the [**xgcm**](https://xgcm.readthedocs.io/en/latest/) grid object.

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

## **Recipes**

We are steadily adding more recipes to the NEMO Cookbook. Here, we include a list of currently available recipes & several more that are in development.

**Available:**

1. Meridional Overturning Stream Function in an arbitrary tracer coordinates.

2. Meridional Overturning Stream Function in depth coordinates (z/z*).

3. Meridional Heat & Salt Transports.

4. Surface-Forced Water Mass Transformation in potential density coordinates.

5. Volume census in temperature - salinity coordinates.

**In Development:**

1. Barotropic Stream Functions.

2. Meridional Overturning Stream Functions in depth coordinates (MES).

3. Ocean Heat Content & Mixed Layer Heat Content. 

4. Sea Ice Diagnostics.

5. Vorticity Diagnostics.
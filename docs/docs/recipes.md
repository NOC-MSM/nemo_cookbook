# Summary

**Welcome to the NEMO Cookbook Recipe page :wave:**

Each recipe uses the `NEMODataTree` and `NEMODataArray` data structures and their associated methods to calculate a diagnostic using NEMO ocean model outputs.

Just like the [**Cosima Cookook**](https://cosima-recipes.readthedocs.io/en/latest/), each recipe is a self-contained and internally documented Jupyter notebook. 

There are several ways to explore our available Recipes:

1. Open a static version of the recipe in your browser by selecting **View in Docs**.
2. Open the recipe in preview mode as a cloud-hosted marimo notebook using **Open in molab**.
    > You can also fork your Recipes and edit it iteractively in the browser using [**molab**](https://molab.marimo.io/notebooks)!
3. Open the recipe as an interactive cloud hosted Jupyter notebook using **Open In Colab**.
    > Note that using [**Google Colab**](https://colab.research.google.com) requires you to pip install nemo_cookbook and its dependencies at the start of your notebook!
4. Clone the [**NEMO Cookbook**](https://github.com/NOC-MSM/nemo_cookbook) GitHub repository and run the Jupyter notebooks directly from the `recipes/` directory yourself.

### :book: Available Recipes
---

<div class="grid cards" markdown>

-   __Meridional Overturning - Tracer__

    ---

    Meridional overturning stream function in an arbitrary tracer coordinates.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_moc_tracer/)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_moc_tracer.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_moc_tracer.ipynb)

-   __Meridional Overturning - Vertical__

    ---

    Meridional overturning stream function in vertical coordinates (z/z*).

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_moc_z/) 
    
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_moc_z.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_moc_z.ipynb)

-   __Ocean Heat Transport__

    ---

    Global meridional ocean heat transport.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_heat_transport/)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_heat_transport.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_heat_transport.ipynb)

-   __Volume Census__

    ---

    Seawater volume census in discrete temperature - salinity coordinates.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_volume_census)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_volume_census.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_volume_census.ipynb)

-   __Water Mass Transformation__

    ---

    Surface-forced water mass transformation in discrete potential density coordinates.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_sfwmt_sigma0)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_sfwmt_sigma0.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_sfwmt_sigma0.ipynb)

-   __Sea Ice Diagnostics__

    ---

    Arctic Ocean sea ice area and sea ice extent.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_sea_ice)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_sea_ice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_sea_ice.ipynb)

-   __Masked Statistics__

    ---

    Grid-aware masked statistics using bounding boxes and geographical polyons.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_masked_stats)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_masked_stats.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_masked_stats.ipynb)

-   __Barotropic Stream Function__

    ---

    Regional barotropic stream functions using grid-aware integration.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_barotropic_sf)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_barotropic_sf.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_barotropic_sf.ipynb)

-   __Ocean Heat Content__

    ---

    Upper ocean heat content using depth integration.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_heat_content/)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_heat_content.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_heat_content.ipynb)

-   __Vertical Coordinate Transformations__

    ---

    Transforming vertical coordinates using conservative remapping.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_transform_vertical_coords)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_transform_vertical_coords.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_transform_vertical_coords.ipynb)

-   __Extracting Hydrographic Sections__

    ---

    Extracting volume transports and properties along the Overturning in the Subpolar North Atlantic array.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_extract_osnap)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_extract_osnap.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_extract_osnap.ipynb)

-   __Regridding using xESMF__

    ---

    Regridding global sea surface temperature from the NEMO curvilinear grid to a regular rectilinear grid using xESMF.

    [:octicons-arrow-right-24: View in Docs](https://noc-msm.github.io/nemo_cookbook/recipe_xesmf)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_xesmf.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NOC-MSM/nemo_cookbook/blob/main/recipes/recipe_xesmf.ipynb)

</div>


### :hammer: Recipes In Development
---

1. Meridional overturning stream functions in multi-envelope sigma coordinates.

2. Ocean heat content & mixed layer heat content. 

3. Sea ice diagnostics.

4. Vorticity diagnostics.

### Contributing New Recipes...
---

If you've used `NEMODataTree` to calculate a commonly used diagnostic not currently included in the **Recipe Lists** above, we'd strongly encourage you to visit the [Contributing] page to learn more about contributing to the **NEMO Cookbook**.

[Contributing]: contributing.md
# **NEMODataTree**

Each recipe in the NEMO Cookbook leverages the `NEMODataTree` object to store NEMO ocean model outputs and to help perform diagnostic calculations.

In this User Guide, we provide a detailed introduction to the `NEMODataTree` and present an example using outputs from the NEMO version 5 `AGRIF_DEMO` reference configuration.

For further details on the available methods to construct a `NEMODataTree` and computation patterns, users are referred to the API documentation.

## What is a DataTree? :fontawesome-solid-folder-tree:

Ocean model simulations produce large groups of datasets, including ocean physics, biogeochemistry, sea ice diagnostics, which are defined on different grids. Moreover, ocean models are often configured using nested domains, where groups of datasets are produced for each of the parent, child, grandchild domains. Managing these related datasets of model diagnostics in a single structure has traditionally been a major challenge.

This is where `xarray.DataTree` object comes in.

It extends the familiar collection of xarray data structures to allow hierarchical grouping of datasets, similar to a file system. Each `xarray.DataTree` is composed of a hierarcy of nodes, where each node contains an `xarray.Dataset`. 

```
DataTree('root')
├── global
└── regional_nest
```

The root node sits at the top of the DataTree ('/') and each of its child nodes can have children (sub-groups) of their own. In the example above, the root node has two children storing the outputs of the global and nested regional domains of an ocean model simulation.

In summary, an `xarray.DataTree` can help ocean modellers organise complex outputs (nested domains, groups of variables) in a natural, hierarchical way by acting as a container for a collection of related  `xarray.Datasets`.


## What is a NEMODataTree? :ocean: x :fontawesome-solid-folder-tree:

`NEMODataTree` is an extension of the `xarray.DataTree` structure designed to store NEMO model output datasets as nodes in a hierarchical tree.

Many users will be familiar with the typical output format of NEMO model simulations, which includes separate netCDF files for groups of variables defined at the same location on NEMO's generalised Arakawa “C” grid (Mesinger and
Arakawa, 1976).

- `...grid_T.nc` :material-arrow-right: scalar variables (e.g., conservative temperature & absolute salinity) defined at the centre of each model grid cell.

- `...grid_U.nc` :material-arrow-right: vector variables (e.g., zonal seawater velocity) defined at the centre of each eastern grid cell face.

- `...grid_V.nc` :material-arrow-right: vector variables (e.g., meridional seawater velocity) defined at the centre of each northern grid cell face.

- `...grid_W.nc` :material-arrow-right: vector variables (e.g., vertical seawater velocity) defined at the centre of each lower grid cell face.

- `...grid_F.nc` :material-arrow-right: vector variables (e.g., relative vorticity) defined at the centre of each vertical edge.

- `...icemod.nc` :material-arrow-right: sea ice variables (e.g., sea ice concetration) defined at the centre of each model grid cell.

For a typical NEMO model configuration, consisting of a global parent domain only, we can define a simple `NEMODataTree`:
```
<xarray.DataTree 'nemo'>
Group: /
├── Group: /gridT
├── Group: /gridU
├── Group: /gridV
├── Group: /gridW
└── Group: /gridF
```

where the `gridT` node stores all of the output `...grid_T.nc` files in a single `xarray.Dataset` and so on.

For a nested NEMO model configuration, including a parent, child and grandchild domain, we can define a more complex `NEMODataTree`:
```
<xarray.DataTree 'nemo'>
Group: /
├── Group: /gridT
|   └── Group: /gridU/1_gridU
|       └── Group: /gridU/1_gridU/2_gridU
├── Group: /gridU
|   └── Group: /gridU/1_gridU
|       └── Group: /gridU/1_gridU/2_gridU
├── Group: /gridV
|   └── Group: /gridV/1_gridV
|       └── Group: /gridV/1_gridV/2_gridV
├── Group: /gridW
|   └── Group: /gridW/1_gridW
|       └── Group: /gridW/1_gridW/2_gridW
└── Group: /gridF
    └── Group: /gridF/1_gridF
        └── Group: /gridF/1_gridF/2_gridF
```

where each parent grid node (e.g., `gridT`) has a corresponding child grid node (e.g., `1_gridT`), which itself has a corresponding child (grandchild) node (e.g., `2_gridT`).

Importantly, we do not need a `domain` node containing the grid scale factors and masks defining each model domain since these variables are assigned to their respective grid nodes (e.g., horizontal grid scale factors `e1t` and `e2t` are stored in `gridT` etc.).

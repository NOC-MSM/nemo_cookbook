# Data Structures

NEMO Cookbook extends the familiar xarray data model by introducing the `NEMODataTree` and `NEMODataArray` objects to store NEMO ocean model outputs and perform grid-aware computation.

Here, we provide an introduction to the `NEMODataTree` and `NEMODataArray` objects, their properties and the methods we can use to create them.

---

## NEMODataTree :ocean: :fontawesome-solid-folder-tree:

`NEMODataTree` is an extension of the `xarray.DataTree` structure designed to store NEMO model output datasets as nodes in a hierarchical tree.

### **What is a DataTree?** :fontawesome-solid-folder-tree:

Ocean model simulations produce large collections of datasets, including physics, biogeochemistry, and sea ice diagnostics, which are defined on different grids. Moreover, ocean models configuration often include nested domains, where datasets of model diagnostics are produced for each of the parent, child and grandchild domains.

Organising these gridded datasets into a single, interpretable data structure has traditionally been a major challenge for researchers when developing their data analysis workflows.

This is where the `xarray.DataTree` comes in.

The `xarray.DataTree` extends the more familiar collection of xarray data structures (e.g., `xarray.Dataset`) to allow hierarchical grouping of datasets, similar to a local file system. Each `xarray.DataTree` is composed of a hierarchy of nodes, each containing a separate `xarray.Dataset`. 

```
<xarray.DataTree 'OceanModel'>
Group: /
└── Group: /global
    ├── Group: /global/regional_nest_1
    └── Group: /global/regional_nest_2
```

The root node sits at the top of the DataTree ('/') and each of its child nodes can have children (or sub-groups) of their own. In the example above, the root node has a single child node (`global`) storing the global domain outputs of an ocean model simulation `OceanModel`. This in-turn has two child nodes (`regional_nest_1` & `regional_nest_2`) storing the outputs of two regional nests located inside the global domain.

We can hence describe each node in a DataTree in terms of the `parent` to which the node belongs, and its `children` - child nodes to which it is the parent. The root node is an important exception however, since it has no `parent` node.

To access a node in our DataTree, we use Python's standard dictionary syntax to define the path to the target node in the DataTree as follows:

```
dt['global/regional_nest']
```

We can then access the variables stored in the `xarray.Dataset` associated with a given node as follows:

```
ds['global/regional_nest']['var_name']
```

In summary, an `xarray.DataTree` can help ocean modellers organise complex outputs (nested domains, groups of variables) in a natural, hierarchical way by acting as a container for a collection of related  `xarray.Datasets`.

### **NEMO Model Outputs**

Although some experienced researchers will be familiar with the typical output format of NEMO model simulations, we provide a brief summary below for new users.

NEMO model simulations write instantanous or time-averaged diagnostics to output files in NetCDF4 format using an external I/O
library and server named [XIOS](https://gitlab.in2p3.fr/ipsl/projets/xios-projects/xios).

Typically, separate NetCDF files are produced at each time-averaging interval (e.g., monthly) for groups of variables located at the same type of grid points. This results in the following types of NetCDF files:

- `...grid_T.nc` :material-arrow-right: scalar variables (e.g., conservative temperature & absolute salinity) defined at the centre of each model grid cell.

- `...grid_U.nc` :material-arrow-right: vector variables (e.g., zonal seawater velocity) defined at the centre of each eastern grid cell face.

- `...grid_V.nc` :material-arrow-right: vector variables (e.g., meridional seawater velocity) defined at the centre of each northern grid cell face.

- `...grid_W.nc` :material-arrow-right: vector variables (e.g., vertical seawater velocity) defined at the centre of each bottom grid cell face.

- `...grid_F.nc` :material-arrow-right: vector variables (e.g., relative vorticity) defined at the centre of each vertical edge.

Often global scalar diagnostics (e.g., global mean temperature) are also produced, resulting in a further type of NetCDF file:

- `...scalar.nc` :material-arrow-right: 1-dimensional scalar variables calculated by aggregating a variable defined on the model **T** grid.

When the NEMO ocean engine is coupled to a sea ice model (e.g., [**SI3**](https://doi.org/10.5281/zenodo.7534900)), NetCDF files will also be produced for sea ice variables using the following suffix:

- `...icemod.nc` :material-arrow-right: sea ice variables (e.g., sea ice concentration) defined at the centre of each model grid cell.


### **Defining a NEMODataTree**

We have seen how a NEMO model can generate a large collection of NetCDF output files and how DataTrees can be used to group such datasets in a hierarchical file system-like structure. 

A NEMODataTree simply organises the outputs from one or more NEMO model domains into a hierarchical `xarray.DataTree`, such that each **domain** is represented as a level in the DataTree, while the **grids** comprising the NEMO ocean mesh (`gridT`, `gridU`, `gridV`, `gridW`, `gridF`, etc.) are stored as nodes within that domain.

#### **Basic NEMODataTrees**

For a typical NEMO model configuration, consisting of a global parent domain coupled to a sea ice model, we can define a simple `DataTree`:
```
<xarray.DataTree 'nemo'>
Group: /
├── Group: /gridT
├── Group: /gridU
├── Group: /gridV
├── Group: /gridW
└── Group: /gridF
```

where the `gridT` child node contains time series of scalar variables stored in the `...grid_T.nc` files in a single `xarray.Dataset` and so on.

**Domain Variables**

Importantly, a `NEMODataTree` does not need a `domain` node to store the grid scale factors and masks associated with each model domain. 

*Why?*

This is because domain variables are assigned to their respective grid nodes during pre-processing (e.g., horizontal grid scale factors `e1t` and `e2t` are stored in `gridT` etc.).

!!! tip "Note on Quasi-Eulerian Vertical Coordinates..."

    **The vertical grid scale factors (e.g., `e3t`, `e3u` etc.) assigned to a `NEMODataTree` are dependent upon the type of vertical coordinate used in the given NEMO model simulation.**

    Typically, NEMO model simulations use a quasi-eulerian vertical coordinate which absorbs the divergence of horizontal barotropic velocities (e.g., $z^{*}$ or $s^{*}$), meaning that vertical grid scale factors evolve through time (i.e., a time-varying free surface translates into variations in grid cell thickness).

    `NEMODataTree` considers the case of time-evolving vertical grid scale factors to be the default as the `linssh` argument to the `.from_paths()` and `.from_datasets()` constructors is set to be `False` by default. This means that vertical grid scale factors must be provided in the NetCDF files or `xarray.Datasets` used to define each NEMO model grid node in the `NEMODataTree`.

    For NEMO model simulations using a linear free surface approximation (i.e., variations in the free surface are neglected compared to the depth of the ocean), we should use `linssh=True` to indicate that vertical grid scale factors remain fixed through time and should be read directly from the reference variables contained within the domain_cfg NetCDF file or `xarray.Dataset` (e.g., `e3t_0`, `e3u_0` etc.).

**Dimensions & Coordinates**

During NEMODataTree construction, these coordinate dimensions are transformed into the NEMO model grid indices (**i**, **j**, **k**) according to the Table included in the **NEMO Model Grid** section above. This has two important implications:

1. The `xarray.Datasets` stored in each grid node share the same coordinate dimension names (`i`, `j`, `k`), but are staggered according to where variables are position on the NEMO model grid.

2. All grid indices use Fortran (1-based) indexing consistent with their definition in the original NEMO model code.

In practice, this means that a variable defined at the first T-point will be at (`i=1`, `j=1`), whereas a variable located at the first U-point will be at (`i=1.5`, `j=1`). This approach was chosen to ensure users encounter alignment errors when attempting to calculate diagnostics using variables defined on different grids. Instead, scalar or vector variables should be interpolated onto the desired grid before computation.

A further practical implication is that users should always use `.sel()` to subset data variables according to their grid indices on the NEMO ocean mesh.

#### **Nested NEMODataTrees**

For a nested NEMO model configuration, including a parent, child and grandchild domain, we can define a more complex `NEMODataTree`:
```
<xarray.DataTree 'nemo'>
Group: /
├── Group: /gridT
|   └── Group: /gridT/1_gridT
|       └── Group: /gridT/1_gridT/2_gridT
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

**Domain Variables**

Nested child / grandchild domain variables are also assigned to their respective grid nodes during pre-processing (e.g., horizontal grid scale factors `e1t` and `e2t` are stored in `gridT` etc.).

**Dimensions & Coordinates**

To ensure that the dimensions of nested child / grandchild domains are distinct from their parent, a prefix is added to all grid indices and associated geographical coordinate variables.

The prefix corresponds to the unique domain number used to identify each child and grandchild domain during the construction the `NEMODataTree`. Hence, in the example above, the child grid node `1_gridT` will have NEMO model grid indices (`i1`, `j1`, `k1`) and associated coordinates `1_glamt(j1, i1)`, `1_gphit(j1, i1)` etc.

### Creating a NEMODataTree

**Using NetCDF Files**

We can create a NEMODataTree from a dictionary of paths to local NetCDF files using the `from_paths()` constructor.

The `from_paths()` constructor takes:

* `paths`: A dictionary of paths to NEMO output NetCDF files.
* `nests`: An optional dictionary describinng the properties of nested domains.
* `name`: Name of the NEMODataTree.
* `iperio`: Zonal periodicity of the parent domain.
* `nftype`: Type of north fold lateral boundary condition implemented.

Let's define a NEMODataTree for a global parent domain, which is zonally periodic (`iperio=True`) and north-folding on **T**-points (`nftype="T"`).

``` py
paths = {"parent": {
         "domain": "/path/to/domain_cfg.nc",
         "gridT": "path/to/*_gridT.nc",
         "gridU": "path/to/*_gridV.nc",
         "gridV": "path/to/*_gridV.nc",
         "gridW": "path/to/*_gridW.nc",
         "icemod": "path/to/*_icemod.nc",
        },
        }

nemo = NEMODataTree.from_paths(paths=paths, iperio=True, nftype="T", name="My NEMODataTree")
```

Note, we are only required to specify paths for one or more NEMO model grid types (e.g., `*_gridT.nc`) to construct a NEMODataTree.

The following core dimensions are expected to be found in the input NetCDF files:

* **domain**: (`nav_lev`, `y`, `x`)

* **grid{T/U/V/W}**: (`time_counter`, `depth{p}`, `y`, `x`), where *p* is the grid point type.

It is also possible to specify the path to a grid NetCDF file (e.g., `*_gridT.nc`) including only 3-dimensional variables, in which case, the expected core dimensions are: (`time_counter`, `y`, `x`).

If the domain_cfg file includes all of the required 2 and 3-dimensonal land-sea mask variables, we should use `read_masks=True` to read these variables rather than compute them during NEMODataTree creation. By default, land-sea masks are computed from the `top_level` and `bottom_level` variables in the domain_cfg file (accounting for the north folding lateral boundary condition `nftype`). When only a partial complete domain_cfg file is available, using `read_mask=True` will attempt to read each mask variable before computing them if not found in the domain_cfg file.

A recommended pattern is to create a NEMODataTree once using `read_mask=False` if you are missing land-sea mask variables before adding these variables to your domain_cfg and writing this to a new `domain_cfg_mesh_mask.nc` file. The path to the resulting file can then be specified in the `paths` dictionary alongside `read_mask=True` to avoid recomputing land-sea mask variables repeatedly. 

For NEMO models implementing a linear free-surface approximation (i.e., vertical scale factors are time-independent), we should also specify `linssh=True` to read these directly from the domain_cfg file included in the `paths` dictionary:

``` py
nemo = NEMODataTree.from_paths(paths=paths, iperio=True, nftype="T", name="My NEMODataTree", linssh=True)
```

For NEMO models configured with more complex vertical coordinates (e.g., MEs or $\sigma$-coordinates), such that vertical reference variables vary spatially (e.g., `deptht(k, j, i)`), we should specify `vco="3d"` to include all vertical reference variables as 3-dimensional arrays analogously to using `key_vco_3d` within NEMO itself:

``` py
nemo = NEMODataTree.from_paths(paths=paths, iperio=True, nftype="T", vco="3d")
```

By default, a NEMODataTree is constructed using 1-dimensional vertical reference variables (e.g., `deptht(k)`), which are populated using the `gdept_1d` and `gdepw_1d` variables in the domain_cfg file. When the aforementioned vertical reference variables are not found in the domain_cfg file, the 1-dimensional depth variables (e.g., `deptht`) are retained from the grid NetCDF files.

We can also include the vertical reference 

**Summary**

Below we summarise the internal steps taken to define a NEMODataTree from a collection of output NetCDF files:

!!! example "Steps to Define a NEMODataTree"

    1. For each type of NetCDF output, open all available files as a single `xarray.Dataset` using either `xarray.open_dataset()` (single path) or `xarray.open_mfdataset()` (multiple paths).

    2. Add domain variables stored in the **domain_cfg.nc** file to the each grid dataset (e.g., `e1t`, `e2t` are added to `gridT`).

    3. Read or compute land-sea masks for each grid type (e.g., `tmask` is added to `gridT`).

    4. Redefine the dimensions `dims` and coordinates `coords` of each grid dataset to use `i`, `j`, `k` as used in the semi-discrete equations in NEMO.

    5. Assemble the `xarray.DataTree` using a dictionary of processed NEMO model grid datasets.

The steps above highlight that the NEMODataTree is simply a specific case of the more general `xarray.DataTree` structure.

**Using Datasets**

We can also create a NEMODataTree from a dictionary of `xarray.Datasets` using the `from_datasets()` constructor. This is particularly valuable when working with remote NEMO model data or Coupled Model Intercomparison Project (CMIP) outputs which require us to modify coordinate dimensions.

The `from_datasets()` constructor takes:

* `datasets`: A dictionary of `xarray.Datasets` containing NEMO outputs.
* `nests`: An optional dictionary describinng the properties of nested domains.
* `name`: Name of the NEMODataTree.
* `iperio`: Zonal periodicity of the parent domain.
* `nftype`: Type of north fold lateral boundary condition implemented.

Let's define a NEMODataTree for a nested NEMO configuration including a global parent domain, which is zonally periodic (`iperio=True`) and north-folding on **T**-points (`nftype="T"`), and two succesively nested regional domains.

We'll start by defining the `datasets` dictionary for the global `parent` domain and its `child` and `grandchild` domains. Note, when we define `child` and `grandchild` domains, we must also specify a unique domain number, given that we could include further child or grandchild nests.

``` py
datasets = {"parent": {
            "domain": ds_domain_cfg, "gridT": ds_gridT
            },
            "child": {
            "1": {"domain": ds_child_domain_cfg, "gridT": ds_child_gridT,}
            },
            "grandchild": {
            "2": {"domain": ds_grandchild_domain_cfg, "gridT": ds_grandchild_gridT,}
            },
            }
```

The following core dimensions are expected in the `xarray.Datasets` included in the `datasets` dictionary:

* **domain**: (`nav_lev`, `y`, `x`)

* **grid{T/U/V/W}**: (`time_counter`, `depth{p}`, `y`, `x`), where *p* is the grid point type.

Next, we need to define a `nests` dictionary which contains the properties which define each nested domain. These include:

- Unique domain number (mapping properties to entries in our `paths` directory).
- Parent domain (to which unique domain does this belong).
- Zonal periodicity of child / grandchild domain (`iperio`).
- Horizontal grid refinement factors (`rx`, `ry`).
- Start (`imin`, `jmin`) and end (`imax`, `jmax`) grid indices in both directions (**i**, **j**) of the parent grid.

The latter information should be copied directly from the `AGRIF_FixedGrids.in` anicillary file used to define nested domains in NEMO.

***
`Example AGRIF_FixedGrids.in`

**1** ------------------------> (Number of nested domains - parent).

**121 146 113 133 4 4 4** ----> (imin, imax, jmin, jmax, rx, ry, rt)

**1** ------------------------> (Number of nested domains - child)

**20 60 27 60 3 3 3** --------> (imin, imax, jmin, jmax, rx, ry, rt)

**0** ------------------------> (Number of nested domains - grandchild)

***

**Note, we must specify the start and end grid indices using Fortran (1-based) indexes rather than Python (0-based) indexes.**

``` py
nests = {
    "1": {
    "parent": "/",
    "rx": 4,
    "ry": 4,
    "imin": 121,
    "imax": 146,
    "jmin": 113,
    "jmax": 133,
    "iperio": False
    },
    "2": {
    "parent": "1",
    "rx": 3,
    "ry": 3,
    "imin": 20,
    "imax": 60,
    "jmin": 27,
    "jmax": 60,
    "iperio": False
    }
    }

nemo = NEMODataTree.from_datasets(datasets=datasets,
                                  nests=nests,
                                  iperio=True,
                                  nftype="T",
                                  name="My Nested NEMODataTree"
                                  )
```

We can also include additional keyword arguments to pass onto `xarray.open_dataset` or `xr.open_mfdataset` when opening grid NetCDF files (e.g., `*_gridT.nc`):

``` py
nemo = NEMODataTree.from_datasets(datasets=datasets, nests=nests, iperio=True, nftype="T", engine="netcdf4")
```

All nested `child` and `grandchild` domains are clipped to remove ghost grid points along the boundaries according to the value of the `nbghost_child` keyword argument during NEMODataTree creation. By default, `nbghost_child=4` following the NEMO User Guide, meaning four grid points will be removed from the each of the northern, southern, eastern and western boundaries of each `child` or `grandchild` domain. Alternatively, the entire child domain, including ghost points, can be retained by passing `nbghost_child=None`.

**Summary**

Below we summarise the internal steps taken to define a nested NEMODataTree from a collection of `xarray.Datasets`:

!!! example "Steps to Define a Nested NEMODataTree"
    1. For each grid type, validate the dimensions and coordinates of all input `xarray.Datasets`.

    2. Add domain variables stored in the **domain_cfg** dataset to the each grid dataset (e.g., `e1t`, `e2t` are added to `gridT`).

    3. Read or compute masks for each grid type (e.g., `tmask` is added to `gridT`).

    4. Re-define the dimensions `dims` and coordinates `coords` of each grid dataset to use `i{dom}`, `j{dom}`, `k{dom}` as used to define the semi-discrete equations in NEMO, where *dom* is the unique domain number.

    5. **Clip nested child domains to remove ghost points along the boundaries & add a mapping from the parent grid indices to the child grid indices to the `coords`.**

    6. **Assemble dictionaries of processed NEMO model grid datasets for each of the parent, child and grandchild domains.**

    7. Assemble the underlying `xarray.DataTree` using a nested dictionary of NEMO model domains.

### NEMODataTree Contents

Like all familiar xarray data structures, NEMODataTree implements a Python mapping interface, with values given by `DataTree` objects corresponding to the grid nodes of a NEMO ocean mesh.

To see all of the grid nodes in a our NEMODataTree, we can use the `groups` property:

``` py
nemo.groups
```

```
('/', '/gridT', '/gridU', '/gridV', '/gridW', '/gridF')
```

In this case, the resulting tuple includes the standard grid nodes for a global parent domain and `'/'` which corresponds to the root node of our NEMODataTree.

We can access the nodes in an NEMODataTree as an `xarray.DataTree` using dictionary-like syntax:

``` py
nemo["gridT"]
```

Alternatively, we can access the node through an immutable dataset view using the `dataset` property:

``` py
nemo["gridT"].dataset
```

If we need to access the contents of a grid node as a new, mutable `xarray.Dataset`, we can also use the `to_dataset()` method:

``` py
nemo["gridT"].to_dataset()
```

We can also update the contents of a NEMODataTree in-place using Python's dictionary syntax. For example, we could add a new grid node to a NEMODataTree using:

``` py
nemo["gridF"] = ds_grid_F
```

Before a new grid node is appended to an existing NEMODataTree, the input `xarray.Dataset` will be validated to ensure it contains:

* NEMO grid dimension coordinates (`i, j`).
* Longitude and Latitude coordinates named `gphi{x}` and `glam{x}`.
* Depth coordinate named `depth{x}` provided that NEMO grid dimension `k` exists.

where `x` is final character of the specified grid node.

In the above example, `gphif`, `glamf` and `depthf` would be expected since we are introducing a new grid node named `"gridF"`.

---

## NEMODataArray :ocean: :simple-databricks:

NEMODataArray is an extension of the familiar `xarray.DataArray` structure which supports grid-aware computation using variables defined on a NEMO model grid.

Each NEMODataArray interfaces with a parent NEMODataTree to provide useful properties, grid-aware operators and statistics, alongside utility methods to interpolate variables onto neighbouring NEMO model grids.

### **Creating a NEMODataArray**

There are two ways to create a NEMODataArray from a variable defined on a given NEMO model grid:

1. **Manual NEMODataArray Creation...**

    We can create a NEMODataArray from an existing `xarray.DataArray` variable and NEMODataTree using the following syntax:

``` py
NEMODataArray(da=thetao_con, tree=nemo, grid="gridT")
```

```
<NEMODataTree 'My NEMO model'>
<NEMODataArray 'thetao_con' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>

<xarray.DataArray 'thetao_con' (time_counter: 48, k: 75, j: 331, i: 360)> Size: 2GB
...
```

Here, we have created a NEMODataArray using an `xarray.DataArray` containing the conservative temperature variable (`thetao_con`) and the path to the **T**-grid node (`"gridT"`) of the NEMODataTree to which it belongs (`nemo`).

2. **Using a NEMODataTree for NEMODataArray Creation...**

    A more natural approach to NEMODataArray creation is to use the path to the variable in the NEMODataTree directly:

``` py
nemo["gridT/thetao_con"]
```

```
<NEMODataTree 'My NEMO model'>
<NEMODataArray 'thetao_con' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>

<xarray.DataArray 'thetao_con' (time_counter: 48, k: 75, j: 331, i: 360)> Size: 2GB
...
```

Note, we can also access the conservative temperature `xarray.DataArray` by first passing the path to the NEMO grid on which the variable is defined:

``` py
nemo["gridT"]["thetao_con"]
```

### **NEMODataArray Contents**

Each NEMODataArray has several key properties to support grid-aware computation.

- `.data` :material-arrow-right: Underlying `xarray.DataArray` of the NEMO output variable.

- `.grid` :material-arrow-right: Path to NEMO model grid node where variable is stored.

- `.grid_type` :material-arrow-right: Type of NEMO model grid where variable is defined.

- `.metrics` :material-arrow-right: Dictionary of NEMO model grid scale factors (e.g., `e1t`, `e2t` etc.) associated with the variable.

- `.mask` :material-arrow-right: Variable land-sea mask (`xarray.DataArray`).

- `.masked` :material-arrow-right: Returns variable `NEMODataArray` with land-sea mask applied.

For example, to access the grid scale factors associated with the conservative temperature variable `thetao_con`:

``` py
nemo["gridT/thetao_con"].metrics
```

returns a dictionary of NEMODataArrays with generic keys (`"e1", "e2", "e3"`).

```
{"e1" : <NEMODataTree  'Example NEMODataTree' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>
        <xarray.DataArray 'e1t' (j: 10, i: 10)>,
 "e2" : <NEMODataTree  'Example NEMODataTree' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>
        <xarray.DataArray 'e2t' (j: 10, i: 10)>,
 "e3" : <NEMODataTree  'Example NEMODataTree' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>
        <xarray.DataArray 'e3t' (time_counter: 10, k:10, j: 10, i: 10)>,
}
```

Note, when we access the `metrics` property using only a subset of the original NEMODataArray (i.e., following use of `isel()` or 'sel()`), the resulting scale factors are similarly subset to ensure alignment with the variable.

Similarly, when we access the `mask` property for a 4-dimensional variable with dimensions (`time_counter, k, j, i`), the full 3-dimensional land-sea mask will be returned (e.g., `tmask` for a variable defined on **T**-points). However, if we subset this NEMODataArray to select data at a single `k`-index (i.e., `sel(k=1)`) and then access the `mask` property, the 2-dimensional unique point land-sea mask will be returned (e.g., `tmaskutil` for a variable defined on **T**-points).

``` py
nemo["gridT/thetao_con"].sel(k=1).mask
```

```
<NEMODataTree  'Example NEMODataTree' (Domain: '.', Grid: 'gridT', Grid Type: 'T')>
<xarray.DataArray 'tmaskutil' (j: 10, i: 10)>,
```

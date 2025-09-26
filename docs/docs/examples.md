In this section, we demonstrate how to construct `NEMODataTrees` using example global, regional and coupled NEMO ocean sea-ice outputs.

> Users can also explore the following examples using the `example_nemodatatrees.ipynb` Jupyter Notebook available in the `recipes` directory.

To get started, let's begin by importing the Python packages we'll be using:

```python
import xarray as xr
import nemo_cookbook as nc
from nemo_cookbook import NEMODataTree
```

### **Global Ocean Sea-Ice Models:**
---

**1. `AGRIF_DEMO`**

Let's start by creating a `NEMODataTree` using example outputs from the global `AGRIF_DEMO` NEMO reference configuration.

`AGRIF_DEMO` is based on the `ORCA2_ICE_PISCES` reference configuration with the inclusion of 3 online nested domains.

Here, we will only consider the 2° global parent domain.

Further information on this reference configuration can be found [**here**](https://sites.nemo-ocean.io/user-guide/cfgs.html#agrif-demo).

---

**NEMO Cookbook** includes a selection of example NEMO model output datasets accessible via cloud object storage.

`nemo_cookbook.examples.get_filepaths()` is a convenience function used to download and generate local filepaths for an available NEMO reference configuration.

```python
filepaths = nc.examples.get_filepaths("AGRIF_DEMO")

filepaths
```

The `get_filepaths()` function will download each of the files to your local machine, returning a dictionary of filepaths for the chosen configuration (`AGRIF_DEMO`):

```
{'domain_cfg.nc': '/Users/me/Library/Caches/nemo_cookbook/AGRIF_DEMO/domain_cfg.nc',
 '2_domain_cfg.nc': '/Users/me/Library/Caches/nemo_cookbook/AGRIF_DEMO/2_domain_cfg.nc',
 '3_domain_cfg.nc': '/Users/me/Library/Caches/nemo_cookbook/AGRIF_DEMO/3_domain_cfg.nc',
 'ORCA2_5d_00010101_00010110_grid_T.nc': '/Users/me/Library/Caches/nemo_cookbook/AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_T.nc',
 ...
 '3_Nordic_5d_00010101_00010110_icemod.nc': '/Users/me/Library/Caches/nemo_cookbook/AGRIF_DEMO/3_Nordic_5d_00010101_00010110_icemod.nc'
 }
```

Next, we need to define the `paths` dictionary, which contains the filepaths corresponding to our global parent domain.

We populate the `parent` dictionary with the filepaths to the `domain_cfg` and `gridT/U/V/W` netCDF files produced for the `AGRIF_DEMO` parent domain. 

```python
paths = {"parent": {
         "domain": filepaths["domain_cfg.nc"],
         "gridT": filepaths["ORCA2_5d_00010101_00010110_grid_T.nc"],
         "gridU": filepaths["ORCA2_5d_00010101_00010110_grid_U.nc"],
         "gridV": filepaths["ORCA2_5d_00010101_00010110_grid_V.nc"],
         "gridW": filepaths["ORCA2_5d_00010101_00010110_grid_W.nc"],
         "icemod": filepaths["ORCA2_5d_00010101_00010110_icemod.nc"]
        },
        }
```

We can construct a new `NEMODataTree` called `nemo` using the `.from_paths()` constructor.

Notice, that we also need to specify that our global parent domain is zonally periodic (`iperio=True`) and north folding on T-points (`nftype = "T"`) rather than a closed (regional) domain.

```python
nemo = NEMODataTree.from_paths(paths, iperio=True, nftype="T")

nemo
```

This returns the following `xarray.DataTree`, where we have truncated the outputs for improved readability:

```
<xarray.DataTree>
Group: /
│   Dimensions:               (time_counter: 2, axis_nbounds: 2, ncatice: 5)
│   ...
│ 
├── Group: /gridT
│       Dimensions:               (time_counter: 2, axis_nbounds: 2, j: 148, i: 180,
│                                  ncatice: 5, k: 31)
│       Coordinates:
│           time_centered         (time_counter) object 16B 0001-01-03 12:00:00 0001-...
│         * deptht                (k) float32 124B 5.0 15.0 25.0 ... 4.75e+03 5.25e+03
│           time_instant          (time_counter) object 16B ...
│           gphit                 (j, i) float64 213kB ...
│           glamt                 (j, i) float64 213kB ...
│         * k                     (k) int64 248B 1 2 3 4 5 6 7 ... 25 26 27 28 29 30 31
│         * j                     (j) int64 1kB 1 2 3 4 5 6 ... 143 144 145 146 147 148
│         * i                     (i) int64 1kB 1 2 3 4 5 6 ... 175 176 177 178 179 180
│       Dimensions without coordinates: axis_nbounds
│       Data variables: (12/87)
│           time_centered_bounds  (time_counter, axis_nbounds) object 32B 0001-01-01 ...
│           time_counter_bounds   (time_counter, axis_nbounds) object 32B 0001-01-01 ...
│           simsk                 (time_counter, j, i) float32 213kB ...
│           simsk05               (time_counter, j, i) float32 213kB ...
│           simsk15               (time_counter, j, i) float32 213kB ...
│           snvolu                (time_counter, j, i) float32 213kB ...
│           ...                    ...
│           e1t                   (j, i) float64 213kB ...
│           e2t                   (j, i) float64 213kB ...
│           top_level             (j, i) int32 107kB ...
│           bottom_level          (j, i) int32 107kB ...
│           tmask                 (k, j, i) bool 826kB False False False ... False False
│           tmaskutil             (j, i) bool 27kB False False False ... False False
│       Attributes:
│           name:         ORCA2_5d_00010101_00010110_icemod
│           description:  ice variables
│           title:        ice variables
│           Conventions:  CF-1.6
│           timeStamp:    2025-Sep-13 17:44:13 GMT
│           uuid:         c6c24bd5-1d2b-4d7b-98b5-8d379c94e84b
│           nftype:       T
│           iperio:       True
├── Group: /gridU
│       Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│                                  i: 180)
│       ...
├── Group: /gridV
│       Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│                                  i: 180)
│       ...
├── Group: /gridW
│       Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│                                  i: 180)
│       ...
└── Group: /gridF
        Dimensions:       (j: 148, i: 180, k: 31)
        ...
```

--- 

**2. `NOC Near-Present Day eORCA1`**

Next, we'll consider monthly-mean outputs from the National Oceanography Centre Near-Present-Day global eORCA1 configuration of NEMO forced using JRA55-do from 1976-2024. 

For more details on this model configuration and the available outputs, users can explore the Near-Present-Day documentation [**here**](https://noc-msm.github.io/NOC_Near_Present_Day/).

---

The eORCA1 JRA55v1 NPD data are publicly accessible as remote Zarr v2 stores via [JASMIN Object Store](https://help.jasmin.ac.uk/docs/short-term-project-storage/using-the-jasmin-object-store/), so we will use the NEMODataTree `.from_datasets()` constructor. 

```python 
base_url = "https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca1-jra55v1"

# Opening domain_cfg:
ds_domain = (xr.open_zarr(f"{base_url}/domain/domain_cfg", consolidated=True, chunks={})
             .squeeze(drop=True)
             .rename({"z": "nav_lev"})
             )

# Opening gridT dataset, including sea surface temperature (°C) and salinity (g kg-1):
ds_gridT = xr.merge([xr.open_zarr(f"{base_url}/T1m/{var}", consolidated=True, chunks={})[var] for var in ['tos_con', 'sos_abs']], compat="override")
```

Next, let's create a `NEMODataTree` from a dictionary of eORCA1 JRA55v1 `xarray.Datasets`, specifying that our global domain is zonally periodic (`iperio=True`) and north folding on T-points (`nftype = "F"`).

```python
datasets = {"parent": {"domain": ds_domain, "gridT": ds_gridT}}

nemo = NEMODataTree.from_datasets(datasets=datasets, iperio=True, nftype="F")

nemo
```

```
<xarray.DataTree>
Group: /
│   Dimensions:        (time_counter: 577)
│   ...
│ 
├── Group: /gridT
│   Dimensions:        (time_counter: 577)
│       Dimensions:        (time_counter: 577, j: 331, i: 360, k: 75)
│       Coordinates:
│           time_centered  (time_counter) datetime64[ns] 5kB dask.array<chunksize=(1,), meta=np.ndarray>
│           gphit          (j, i) float64 953kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│           glamt          (j, i) float64 953kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│         * k              (k) int64 600B 1 2 3 4 5 6 7 8 9 ... 68 69 70 71 72 73 74 75
│         * j              (j) int64 3kB 1 2 3 4 5 6 7 8 ... 325 326 327 328 329 330 331
│         * i              (i) int64 3kB 1 2 3 4 5 6 7 8 ... 354 355 356 357 358 359 360
│       Data variables:
│           tos_con        (time_counter, j, i) float32 275MB dask.array<chunksize=(1, 331, 360), meta=np.ndarray>
│           sos_abs        (time_counter, j, i) float32 275MB dask.array<chunksize=(1, 331, 360), meta=np.ndarray>
│           e1t            (j, i) float64 953kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│           e2t            (j, i) float64 953kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│           top_level      (j, i) int32 477kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│           bottom_level   (j, i) int32 477kB dask.array<chunksize=(331, 360), meta=np.ndarray>
│           tmask          (k, j, i) bool 9MB False False False ... False False False
│           tmaskutil      (j, i) bool 119kB False False False ... False False False
│       Attributes:
│           nftype:   F
│           iperio:   True
├── Group: /gridU
│       Dimensions:       (j: 331, i: 360, k: 75)
│       ...
├── Group: /gridV
│       Dimensions:       (j: 331, i: 360, k: 75)
│       ...
├── Group: /gridW
│       Dimensions:       (j: 331, i: 360, k: 75)
│       ...
└── Group: /gridF
        Dimensions:       (j: 331, i: 360, k: 75)
        ...
```

### **Regional Ocean Models:**

---

**`AMM12`**

We can also construct a `NEMODataTree` using outputs from regional NEMO ocean model simulations.

Here, we will consider example outputs from the regional `AMM12` NEMO reference configuration.

The AMM, Atlantic Margins Model, is a regional model covering the Northwest European Shelf domain on a regular lat-lon grid at approximately 12km horizontal resolution. `AMM12` uses the vertical s-coordinates system, GLS turbulence scheme, and tidal lateral boundary conditions using a flather scheme.

Further information on this reference configuration can be found [**here**](https://sites.nemo-ocean.io/user-guide/cfgs.html#amm12).

---

```python
filepaths = nc.examples.get_filepaths("AMM12")

filepaths
```

```
{'domain_cfg.nc': '/Users/me/Library/Caches/nemo_cookbook/AMM12/domain_cfg.nc',
 'AMM12_1d_20120102_20120110_grid_T.nc': '/Users/me/Library/Caches/nemo_cookbook/AMM12/AMM12_1d_20120102_20120110_grid_T.nc',
 'AMM12_1d_20120102_20120110_grid_U.nc': '/Users/me/Library/Caches/nemo_cookbook/AMM12/AMM12_1d_20120102_20120110_grid_U.nc',
 'AMM12_1d_20120102_20120110_grid_V.nc': '/Users/me/Library/Caches/nemo_cookbook/AMM12/AMM12_1d_20120102_20120110_grid_V.nc'
 }
```
As we showed in the `AGRIF_DEMO` example, we need to populate the `paths` dictionary with the `domain_cfg` and `gridT/U/V` filepaths corresponding to our regional model domain.

```python
paths = {"parent": {
         "domain": filepaths["domain_cfg.nc"],
         "gridT": filepaths["AMM12_1d_20120102_20120110_grid_T.nc"],
         "gridU": filepaths["AMM12_1d_20120102_20120110_grid_U.nc"],
         "gridV": filepaths["AMM12_1d_20120102_20120110_grid_V.nc"],
        },
        }
```

Next, we can construct a new `NEMODataTree` called `nemo` using the `.from_paths()` constructor.

Note, we do not actually need to specify that our regional domain is not zonally periodic in this case, given that, by default, `iperio=False`.

```python
nemo = NEMODataTree.from_paths(paths, iperio=False)

nemo
```

```
<xarray.DataTree>
Group: /
│   Dimensions:               (time_counter: 8, axis_nbounds: 2)
│   ...
│ 
├── Group: /gridT
│   Dimensions:               (time_counter: 8, axis_nbounds: 2)
│   ...
├── Group: /gridT
│       Dimensions:               (time_counter: 8, axis_nbounds: 2, j: 224, i: 198,
│                                  k: 51)
│       Coordinates:
│           time_centered         (time_counter) datetime64[ns] 64B ...
│           gphit                 (j, i) float64 355kB ...
│           glamt                 (j, i) float64 355kB ...
│         * k                     (k) int64 408B 1 2 3 4 5 6 7 ... 45 46 47 48 49 50 51
│         * j                     (j) int64 2kB 1 2 3 4 5 6 ... 219 220 221 222 223 224
│         * i                     (i) int64 2kB 1 2 3 4 5 6 ... 193 194 195 196 197 198
│       Dimensions without coordinates: axis_nbounds
│       Data variables:
│           time_centered_bounds  (time_counter, axis_nbounds) datetime64[ns] 128B ...
│           time_counter_bounds   (time_counter, axis_nbounds) datetime64[ns] 128B ...
│           tos                   (time_counter, j, i) float32 1MB ...
│           sos                   (time_counter, j, i) float32 1MB ...
│           zos                   (time_counter, j, i) float32 1MB ...
│           e1t                   (j, i) float64 355kB ...
│           e2t                   (j, i) float64 355kB ...
│           top_level             (j, i) int32 177kB ...
│           bottom_level          (j, i) int32 177kB ...
│           tmask                 (k, j, i) bool 2MB False False False ... False False
│           tmaskutil             (j, i) bool 44kB False False False ... False False
│       Attributes:
│           nftype:   None
│           iperio:   False
├── Group: /gridU
│       Dimensions:               (time_counter: 8, axis_nbounds: 2, j: 224, i: 198,
│                                  k: 51)
│       ...
├── Group: /gridV
│       Dimensions:               (time_counter: 8, axis_nbounds: 2, j: 224, i: 198,
│                                  k: 51)
│       ...
├── Group: /gridW
│       Dimensions:       (j: 224, i: 198, k: 51)
│       ...
└── Group: /gridF
        Dimensions:       (j: 224, i: 198, k: 51)
        ...
```

### **Nested Global Ocean Sea-Ice Models:**

---

`AGRIF_DEMO`

Returning to our `AGRIF_DEMO` NEMO reference configuration, we can also construct a more complex `NEMODataTree` to store the outputs of the global parent and its child domains in a single data structure.

We will make use of the two successively nested domains located in the Nordic Seas, with the finest grid (1/6°) spanning the Denmark strait. This grandchild domain also benefits from “vertical nesting”, meaning that it has 75 geopotential z-coordinate levels, compared with 31 levels in its parent domain.

---

```python 
filepaths = nc.examples.get_filepaths("AGRIF_DEMO")
```

Let's start by defining the `paths` dictionary for the ORCA2 global parent domain and its child and grandchild domains. Notice, that for `child` and `grandchild` domains, we must also specify a unique domain number, given that we could include further child or grandchild nests.

```python
paths = {"parent": {
        "domain": filepaths["domain_cfg.nc"],
        "gridT": filepaths["ORCA2_5d_00010101_00010110_grid_T.nc"],
        "gridU": filepaths["ORCA2_5d_00010101_00010110_grid_U.nc"],
        "gridV": filepaths["ORCA2_5d_00010101_00010110_grid_V.nc"],
        "gridW": filepaths["ORCA2_5d_00010101_00010110_grid_W.nc"],
        "icemod": filepaths["ORCA2_5d_00010101_00010110_icemod.nc"]
        },
        "child": {
        "1":{
        "domain": filepaths["2_domain_cfg.nc"],
        "gridT": filepaths["2_Nordic_5d_00010101_00010110_grid_T.nc"],
        "gridU": filepaths["2_Nordic_5d_00010101_00010110_grid_U.nc"],
        "gridV": filepaths["2_Nordic_5d_00010101_00010110_grid_V.nc"],
        "gridW": filepaths["2_Nordic_5d_00010101_00010110_grid_W.nc"],
        "icemod": filepaths["2_Nordic_5d_00010101_00010110_icemod.nc"]
        }},
        "grandchild": {
        "2":{
        "domain": filepaths["3_domain_cfg.nc"],
        "gridT": filepaths["3_Nordic_5d_00010101_00010110_grid_T.nc"],
        "gridU": filepaths["3_Nordic_5d_00010101_00010110_grid_U.nc"],
        "gridV": filepaths["3_Nordic_5d_00010101_00010110_grid_V.nc"],
        "gridW": filepaths["3_Nordic_5d_00010101_00010110_grid_W.nc"],
        "icemod": filepaths["3_Nordic_5d_00010101_00010110_icemod.nc"]
        }},
        }
```

Next, we need to construct a `nests` dictionary which contains the properties which define each nested domain. These include:

- Unique domain number (mapping properties to entries in our `paths` directory).
- Parent domain (to which unique domain does this belong).
- Zonal periodicity of child / grandchild domain (`iperio`).
- Horizontal grid refinement factors (`rx`, `ry`).
- Start (`imin`, `jmin`) and end (`imax`, `jmax`) grid indices in both directions (**i**, **j**) of the parent grid.

The latter information should be copied directly from the `AGRIF_FixedGrids.in` anicillary file used to define nested domains in NEMO.

---
**`Example AGRIF_FixedGrids.in`**

**1** (Number of nested domains - parent).

**121 146 113 133 4 4 4** (imin, imax, jmin, jmax, rx, ry, rt)

**1** (Number of nested domains - child)

**20 60 27 60 3 3 3** (imin, imax, jmin, jmax, rx, ry, rt)

**0** (Number of nested domains - grandchild)

---

**Important: we must specify the start and end grid indices using Fortran (1-based) indexes rather than Python (0-based) indexes.**

```python
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
```

Finally, we can construct a new `NEMODataTree` called `nemo` using the `.from_paths()` constructor.

Again, we also need to specify that our global parent domain is zonally periodic (`iperio=True`) and north folding on T-points (`nftype = "T"`) rather than a closed (regional) domain.

We can also include additional keyword arguments to pass onto `xarray.open_dataset` or `xr.open_mfdataset` when opening NEMO model output files.

```python 
nemo = NEMODataTree.from_paths(paths=paths, nests=nests, iperio=True, nftype="T", engine="netcdf4")

nemo
```

```
<xarray.DataTree>
Group: /
│   Dimensions:               (time_counter: 2, axis_nbounds: 2, ncatice: 5)
│   ...
│ 
├── Group: /gridT
│   │   Dimensions:               (time_counter: 2, axis_nbounds: 2, j: 148, i: 180,
│   │                              ncatice: 5, k: 31)
│   │   ...
│   └── Group: /gridT/1_gridT
│       │   Dimensions:               (time_counter: 2, axis_nbounds: 2, j1: 80, i1: 100,
│       │                              ncatice: 5, k1: 29)
│       │   ...
│       └── Group: /gridT/1_gridT/2_gridT
│               Dimensions:               (time_counter: 2, axis_nbounds: 2, j2: 99, i2: 120,
│                                          ncatice: 5, k2: 60)
│               ...
├── Group: /gridU
│   │   Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│   │                              i: 180)
│   │   ...
│   └── Group: /gridU/1_gridU
│       │   Dimensions:               (k1: 29, axis_nbounds: 2, time_counter: 2, j1: 80,
│       │                              i1: 100)
│       │   ...
│       └── Group: /gridU/1_gridU/2_gridU
│               Dimensions:               (k2: 60, axis_nbounds: 2, time_counter: 2, j2: 99,
│                                          i2: 120)
│               ...
├── Group: /gridV
│   │   Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│   │                              i: 180)
│   │   ...
│   └── Group: /gridV/1_gridV
│       │   Dimensions:               (k1: 29, axis_nbounds: 2, time_counter: 2, j1: 80,
│       │                              i1: 100)
│       │   ...
│       └── Group: /gridV/1_gridV/2_gridV
│               Dimensions:               (k2: 60, axis_nbounds: 2, time_counter: 2, j2: 99,
│                                          i2: 120)
│               ...
├── Group: /gridW
│   │   Dimensions:               (k: 31, axis_nbounds: 2, time_counter: 2, j: 148,
│   │                              i: 180)
│   │   ...
│   └── Group: /gridW/1_gridW
│       │   Dimensions:               (k1: 29, axis_nbounds: 2, time_counter: 2, j1: 80,
│       │                              i1: 100)
│       │   ...
│       └── Group: /gridW/1_gridW/2_gridW
│               Dimensions:               (k2: 60, axis_nbounds: 2, time_counter: 2, j2: 99,
│                                          i2: 120)
│               ...
└── Group: /gridF
    │   Dimensions:       (j: 148, i: 180, k: 31)
    │   ...
    └── Group: /gridF/1_gridF
        │   Dimensions:       (j1: 80, i1: 100, k1: 29)
        │   ...
        └── Group: /gridF/1_gridF/2_gridF
                Dimensions:       (j2: 99, i2: 120, k2: 60)
                ...
```

### **Coupled Climate Models:**

`UKESM1-0-LL`

In addition to ocean-only and ocean sea-ice hindcast simulations (prescribing surface atmospheric forcing), NEMO models are also used as the ocean components in many coupled climate models, including the UK Earth System Model (UKESM) developed jointly by the UK Met Office and Natural Environment Research Council (NERC).

Here, we show how to construct a `NEMODataTree` from the 1° global ocean sea-ice component of [**UKESM1-0-LL**](https://doi.org/10.1029/2019MS001739) included in the sixth Coupled Model Intercomparsion Project ([**CMIP6**](https://wcrp-cmip.org/cmip-phases/cmip6/)) using outputs accessible via the [**CEDA Archive**](https://help.ceda.ac.uk/article/4801-cmip6-data).

Since CMIP6 outputs are processed and formatted according to the CMIP Community Climate Model Output Rewriter (CMOR) software, we will need to include a few additional pre-processing steps to reformat our NEMO model outputs in order to construct a `NEMODataTree`

**Important: only CMIP model outputs variables stored on their original NEMO ocean model grid (i.e, `gn`) can be used to construct a `NEMODataTree`**

```python
# Open domain_cfg:
ds_domain_cfg = xr.open_dataset("/path/to/MOHC/Ofx/domain_cfg_Ofx_UKESM1.nc")

# Define time decoder to handle CMIP6 time units:
time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)

# Open potential temperature (°C) dataset:
base_filepath = "/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r4i1p1f2/Omon/thetao/gn/latest"
ds_ukesm1_gridT = xr.open_mfdataset(f"{base_filepath}/thetao_Omon_UKESM1-0-LL_historical_r4i1p1f2_gn_*.nc",
                                    data_vars='all',
                                    decode_times=time_decoder
                                   )

# Adding mixed layer depth (m) to dataset:
ds_ukesm1_gridT['mlotst'] = xr.open_mfdataset(f"{base_filepath}/mlotst_Omon_UKESM1-0-LL_historical_r4i1p1f2_gn_*.nc",
                                              data_vars='all',
                                              decode_times=time_decoder
                                              )['mlotst']
```

Now we have defined our `domain` and `gridT` datasets, let's define a `datasets` dictionary ensuring that we rename CMORISED dimensions to be consistent with standard NEMO model outputs.

We can then define a `NEMODataTree` using the `.from_datasets()` constructor, specifying that our global parent domain is zonally periodic and north-folding on F-points.

```python
datasets = {"parent": {
                "domain": ds_domain_cfg.rename({'z':'nav_lev'}),
                "gridT": ds_ukesm1_gridT.rename({'time':'time_counter', 'i':'x', 'j':'y', 'lev':'deptht'}),
                }}

nemo = NEMODataTree.from_datasets(datasets=datasets, iperio=True, nftype="F")

nemo
```
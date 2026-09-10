# Terminology

## DataArray

A multi-dimensional array with labeled or named dimensions. `DataArray` objects add metadata such as dimension names, coordinates, and attributes to underlying “unlabeled” data structures such as numpy and Dask arrays. If its optional `name` property is set, it is a named `DataArray`.

## Dataset

A dict-like collection of `DataArray` objects with aligned dimensions. Thus, most operations that can be performed on the dimensions of a single `DataArray` can be performed on a dataset. `Datasets` have data variables, dimensions, coordinates, and attributes.

## DataTree

A tree-like collection of `Dataset` objects. A tree is made up of one or more nodes, each of which can store the same information as a single `Dataset` (accessed via .dataset). This data is stored in the same way as in a `Dataset`, i.e. in the form of data variables, dimensions, coordinates, and attributes.

## NEMODataTree

A hierarchical `DataTree` object designed to organise NEMO ocean model outputs, such that each domain is represented as a level in the `DataTree`, while the grids comprising the NEMO ocean mesh (`gridT`, `gridU`, `gridV`, `gridW`, `gridF`, etc.) are stored as nodes within that domain.

## NEMODataArray

An extension of the `DataArray` object which stores a multi-dimensional array defined on a NEMO model grid. Each NEMODataArray supports grid-aware computation by interfacing with its parent NEMODataTree to provide useful properties, discrete operators and statistics, alongside utility methods to interpolate variables onto neighbouring NEMO model grids.

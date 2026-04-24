"""
validation.py

Description:
This module includes validation functions for the core data structures
used in the NEMO Cookbook library.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import xarray as xr


def validate_nemo_grid_node(
    key: str,
    value: xr.Dataset,
    ) -> None:
    """
    Validate that an xarray.Dataset to be assigned to a NEMO model grid node
    contains the required dimensions, coordinates and properties based on the
    specified grid type.

    Parameters
    ----------
    key : str
        Variable path (e.g., '/gridT', '/gridU/1_gridU') of the NEMO model
        grid node to validate.
    value : xr.Dataset
        NEMO model grid node Dataset to validate.
     """
    # -- Collect domain prefix, suffix and grid type -- #
    key_nums = [char for char in key if char.isdigit()]
    dom_prefix, dom_suffix = (f"{key_nums[-1]}_", f"{key_nums[-1]}") if len(key_nums) > 0 else ("", "")

    grid_type = key[-1].lower()

    # -- Validate NEMO grid type -- #
    valid_grid_types = ["t", "u", "v", "w", "f", "uw", "vw", "fw", "b"]
    if grid_type not in valid_grid_types:
        raise ValueError(f"Invalid NEMO model grid type. NEMO model grid node type must be one of: {valid_grid_types}.")

    if dom_suffix != "":
        # -- Validate parent & child NEMO grid types match -- #
        if len(set([grid[-1] for grid in key.split("/") if grid != ""])) > 1:
            raise ValueError("Invalid NEMO model grid node. All NEMO model grid nodes within a parent domain must have the same grid type (e.g., T, U, V, W, F).")

    # -- Verify dimensions of NEMO grid node dataset -- #
    valid_dims = [
        (f"i{dom_suffix}", f"j{dom_suffix}"),
        (f"i{dom_suffix}", f"j{dom_suffix}", f"k{dom_suffix}"),
        (f"i{dom_suffix}", f"j{dom_suffix}", f"k{dom_suffix}", "time_counter"),
    ]
    if not any(set(core_dims).issubset(value.dims) for core_dims in valid_dims):
        raise ValueError(f"Invalid NEMO model grid dimensions. NEMO model grid node must comply with one of the following core dimension sets: {valid_dims}.")

    # -- Verify coordinates of NEMO grid node dataset -- #
    valid_coords = [
        (f"i{dom_suffix}", f"j{dom_suffix}", f"{dom_prefix}gphi{grid_type}", f"{dom_prefix}glam{grid_type}"),
        (f"i{dom_suffix}", f"j{dom_suffix}", f"k{dom_suffix}", f"{dom_prefix}gphi{grid_type}", f"{dom_prefix}glam{grid_type}", f"{dom_prefix}depth{grid_type}"),
        (f"i{dom_suffix}", f"j{dom_suffix}", f"k{dom_suffix}", "time_counter"),
    ]
    if not any(set(core_coords).issubset(value.coords) for core_coords in valid_coords):
        raise ValueError(f"Invalid NEMO model grid coordinates. NEMO model grid node must comply with one of the following core coordinate sets: {valid_coords}.")
    
    # -- Verify properties of NEMO grid node dataset -- #
    valid_properties = [
        (f"{grid_type}maskutil", f"{grid_type}mask", f"e1{grid_type}", f"e2{grid_type}", f"e3{grid_type}"),
        # Accept boundary grid nodes without `e2b` grid scale factor:
        (f"{grid_type}maskutil", f"{grid_type}mask", f"e1{grid_type}", f"e3{grid_type}"),
        (f"{grid_type}maskutil", f"e1{grid_type}", f"e2{grid_type}"),
    ]
    if not any(set(core_props).issubset(value.data_vars) for core_props in valid_properties):
        raise ValueError(f"Invalid NEMO model grid properties. NEMO model grid node must comply with one of the following core property sets: {valid_properties}.")


def validate_nemo_dataarray(
        da: xr.DataArray,
        grid: str,
        tree: xr.DataTree,
    ) -> None:
    """
    Validate that an xarray.DataArray to be transformed into a NEMODataArray
    contains the required dimensions, coordinates and properties.

    Parameters
    ----------
    da : xr.DataArray
        Variable defined on a NEMO model grid.
    tree: NEMODataTree
        NEMODataTree to which the variable belongs.
    grid: str
        Path to NEMO model grid where variable is defined (e.g., 'gridT').
     """
    # -- Verify grid exists in NEMODataTree -- #
    grid_keys = list(dict(tree.subtree_with_keys).keys())
    if grid not in grid_keys:
        raise KeyError(
            f"{grid} not found in available NEMODataTree grids {grid_keys}."
        )

    # -- Verify core dimensions exist -- #
    if not all(dim in list(tree[grid].dims) for dim in da.dims):
        raise ValueError(f"DataArray dimensions {da.dims} not all in NEMO model '{grid}' dimensions {tree[grid].dims}.")

    # -- Verify core dimension sizes -- #
    if not all(da.sizes[d] <= tree[grid].sizes[d] for d in da.dims):
        raise ValueError(f"DataArray dimension sizes {da.dims} not all less than or equal to NEMO model '{grid}' dimension sizes {tree[grid].dims}.")

    # -- Verify core coordinates exist -- #
    core_coords = [coord for coord in da.coords if "glam" in coord or "gphi" in coord or "depth" in coord or "time_counter" in coord]
    if not all(dim in list(tree[grid].coords) for dim in core_coords):
        raise ValueError(f"DataArray coordinates {core_coords} not all in NEMO model '{grid}' coordinates {tree[grid].coords}.")

    # -- Verify core coordinate sizes -- #
    if not all(da[coord].sizes[d] <= tree[grid].coords[coord].sizes[d] for coord in core_coords for d in da[coord].dims):
        raise ValueError(f"DataArray coordinate sizes {core_coords} not all less than or equal to NEMO model '{grid}' coordinate sizes {tree[grid].coords}.")

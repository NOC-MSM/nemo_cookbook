"""
stats.py

Description:
This module includes functions to compute statistics from
NEMO ocean general circulation model grids.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import xarray as xr
from flox.xarray import xarray_reduce


def compute_binned_statistic(
    vars: list[xr.DataArray],
    values: xr.DataArray,
    keep_dims: list[str] | None,
    bins: list[list | np.ndarray],
    statistic: str,
    mask: xr.DataArray | None,
) -> xr.DataArray:
    """
    Compute multi-dimensional binned statistic of a variable.

    Parameters
    ----------
    vars : list[xr.DataArray]
        Variable(s) to be grouped in discrete bins.
    values : xr.DataArray
        Values with which to compute binned statistic.
    keep_dims : list[str] | None
        Name of dimensions in `values` to keep as labels in binned statistic.
    bins : list[list | np.ndarray]
        Bin edges used to group each of the variables in `vars`.
    statistic : str
        Statistic to compute (e.g., 'count', 'sum', 'nansum', 'mean', 'nanmean',
        'max', 'nanmax', 'min', 'nanmin'). See flox.xarray.xarray_reduce for a
        complete list of aggregation statistics.
    mask : xr.DataArray | None
        Boolean mask identifying data to be included (1) or neglected (0) from
        computation.

    Returns
    -------
    xr.DataArray
        Values of the selected statistic in each bin.
    """
    # -- Validate input -- #
    if not isinstance(vars, list) or not all(
        isinstance(var, xr.DataArray) for var in vars
    ):
        raise ValueError("vars must be a list of xarray.DataArray objects.")
    if not isinstance(values, xr.DataArray):
        raise ValueError("values must be an xarray.DataArray object.")
    if keep_dims is not None:
        if not isinstance(keep_dims, list) or not all(
            isinstance(dim, str) for dim in keep_dims
        ):
            raise ValueError("keep_dims must be a list of strings or None.")
    if not isinstance(bins, list) or not all(
        isinstance(bin, (list, np.ndarray)) for bin in bins
    ):
        raise ValueError("bins must be a list of lists or ndarrays.")
    if not isinstance(statistic, str):
        raise ValueError("statistic must be a string.")
    if statistic not in [
        "all",
        "any",
        "count",
        "sum",
        "nansum",
        "mean",
        "nanmean",
        "max",
        "nanmax",
        "min",
        "nanmin",
        "argmax",
        "nanargmax",
        "argmin",
        "nanargmin",
        "quantile",
        "nanquantile",
        "median",
        "nanmedian",
        "mode",
        "nanmode",
        "first",
        "nanfirst",
        "last",
        "nanlast",
    ]:
        raise ValueError(f"statistic '{statistic}' is not supported.")
    if mask is not None and not isinstance(mask, xr.DataArray):
        raise ValueError("mask must be an xarray.DataArray object or None.")

    # -- Prepare data -- #
    values_data = values.where(mask) if mask is not None else values
    var_data = [var.where(mask) if mask is not None else var for var in vars]
    keep_vars_data = [values_data[dim] for dim in keep_dims]

    if keep_dims is None:
        keep_dims = []
    expected_groups = [None for _ in keep_dims]
    expected_groups.extend(bin for bin in bins)

    isbin = [False for _ in keep_dims]
    isbin.extend(True for _ in bins)

    # -- Calculate binned statistic -- #
    da = xarray_reduce(
        *[values_data, *keep_vars_data, *var_data],
        func=statistic,
        expected_groups=tuple(expected_groups),
        isbin=tuple(isbin),
        method="map-reduce",
        fill_value=np.nan,  # Fill missing values with NaN.
        reindex=False,  # Do not reindex during block aggregations to reduce memory at cost of performance.
        engine="numbagg",  # Use numbagg grouped aggregations.
    )

    # -- Update binned dimension coords -- #
    # Transform coords from pd.IntervalIndex to interval mid-points:
    coord_dict = {
        f"{var.name}_bins": np.array(
            [interval.mid for interval in da[f"{var.name}_bins"].values]
        )
        for var in vars
    }
    result = da.assign_coords(coord_dict)

    return result

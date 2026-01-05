"""
interpolate.py

Description:
This module includes functions to linearly interpolate between
horizontal grids of NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import numpy as np
import xarray as xr


def masked_average(
    da_list: list[xr.DataArray], mask_list: list[xr.DataArray]
) -> xr.DataArray:
    """
    Compute the average of a DataArray collection
    excluding masked values.

    Parameters
    ----------
    da_list : list[xr.DataArray]
        DataArray collection to be averaged.
    mask_list : list[xr.DataArray]
        Mask collection to apply to DataArrays.

    Returns
    -------
    xr.DataArray
        Masked average of a DataArray collection.
    """
    # Define sum of non-masked DataArray values:
    data_sum = sum([xr.where(mask, da, 0) for da, mask in zip(da_list, mask_list)])

    # Determine number of non-masked values:
    mask_sum = sum([mask.fillna(False).astype(int) for mask in mask_list])
    mask_sum = xr.where(mask_sum, mask_sum, np.nan)

    # Compute masked average:
    result = data_sum / mask_sum

    return result


def interpolate_grid(
    da: xr.DataArray,
    mask: xr.DataArray,
    source_grid: str,
    target_grid: str,
    iperio: bool = False,
    ijk_names: tuple = ("i", "j", "k"),
) -> xr.DataArray:
    """
    Interpolate variable onto neighbouring NEMO horizontal grid.

    Parameters:
    -----------
    da : xr.DataArray
        Variable defined on source grid.
    mask : xr.DataArray
        Mask defined on source grid.
    source_grid : str
        Source grid type ('T', 'U', 'V', or 'F').
    target_grid : str
        Target grid type ('T', 'U', 'V', or 'F').
    iperio : bool, optional
        Zonal periodicity of the domain. Default is False.
    ijk_names : tuple, optional
        Names of the i, j, k dimensions of NEMO source grid.
        Default is ('i', 'j', 'k').

    Returns:
    --------
    xr.DataArray
        Variable interpolated onto target grid.
    """
    # -- Get dimension names -- #
    i_name, j_name, _ = ijk_names

    match source_grid:
        # -- Linearly interpolate T-grid scalar variables -- #
        case "T":
            if target_grid == "U":
                # T -> U grid:
                if iperio:
                    # Zonally periodic domain:
                    result = masked_average(
                        da_list=[da.roll({i_name: 1}), da],
                        mask_list=[mask.roll({i_name: 1}), mask],
                    )
                else:
                    # Non-periodic domain:
                    result = masked_average(
                        da_list=[da, da.shift({i_name: -1})],
                        mask_list=[mask, mask.shift({i_name: -1})],
                    )
            elif target_grid == "V":
                # T -> V grid:
                result = masked_average(
                    da_list=[da, da.shift({j_name: -1})],
                    mask_list=[mask, mask.shift({j_name: -1})],
                )
            elif target_grid == "F":
                # T -> F grid:
                if iperio:
                    # Zonally periodic domain:
                    result = masked_average(
                        da_list=[
                            da.roll({i_name: 1}).shift({j_name: -1}),
                            da.roll({i_name: 1}),
                            da.shift({j_name: -1}),
                            da,
                        ],
                        mask_list=[
                            mask.roll({i_name: 1}).shift({j_name: -1}),
                            mask.roll({i_name: 1}),
                            mask.shift({j_name: -1}),
                            mask,
                        ],
                    )
                else:
                    # Non-periodic domain:
                    result = masked_average(
                        da_list=[
                            da.shift({i_name: -1}).shift({j_name: -1}),
                            da.shift({i_name: -1}),
                            da.shift({j_name: -1}),
                            da,
                        ],
                        mask_list=[
                            mask.shift({i_name: -1}).shift({j_name: -1}),
                            mask.shift({i_name: -1}),
                            mask.shift({j_name: -1}),
                            mask,
                        ],
                    )
            else:
                raise ValueError(
                    f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'."
                )

        # -- Linearly interpolate U-grid flux variables -- #
        case "U":
            if target_grid == "T":
                # U -> T grid:
                if iperio:
                    # Zonally periodic domain:
                    result = masked_average(
                        da_list=[da.roll({i_name: 1}), da],
                        mask_list=[mask.roll({i_name: 1}), mask],
                    )
                else:
                    # Non-periodic domain:
                    result = masked_average(
                        da_list=[da, da.shift({i_name: -1})],
                        mask_list=[mask, mask.shift({i_name: -1})],
                    )
            elif target_grid == "V":
                # U -> V grid:
                if iperio:
                    # Zonally periodic domain:
                    result = masked_average(
                        da_list=[
                            da.roll({i_name: 1}).shift({j_name: 1}),
                            da.roll({i_name: 1}),
                            da.shift({j_name: 1}),
                            da,
                        ],
                        mask_list=[
                            mask.roll({i_name: 1}).shift({j_name: 1}),
                            mask.roll({i_name: 1}),
                            mask.shift({j_name: 1}),
                            mask,
                        ],
                    )
                else:
                    # Non-periodic domain:
                    result = masked_average(
                        da_list=[
                            da.shift({i_name: 1}).shift({j_name: 1}),
                            da.shift({i_name: 1}),
                            da.shift({j_name: 1}),
                            da,
                        ],
                        mask_list=[
                            mask.shift({i_name: 1}).shift({j_name: 1}),
                            mask.shift({i_name: 1}),
                            mask.shift({j_name: 1}),
                            mask,
                        ],
                    )
            else:
                raise ValueError(
                    f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'."
                )

        # -- Linearly interpolate V-grid flux variables -- #
        case "V":
            if target_grid == "T":
                # V -> T grid:
                result = masked_average(
                    da_list=[da, da.shift({j_name: 1})],
                    mask_list=[mask, mask.shift({j_name: 1})],
                )
            elif target_grid == "U":
                # V -> U grid:
                result = masked_average(
                    da_list=[
                        da.shift({i_name: 1}).shift({j_name: 1}),
                        da.shift({i_name: 1}),
                        da.shift({j_name: 1}),
                        da,
                    ],
                    mask_list=[
                        mask.shift({i_name: 1}).shift({j_name: 1}),
                        mask.shift({i_name: 1}),
                        mask.shift({j_name: 1}),
                        mask,
                    ],
                )
            else:
                raise ValueError(
                    f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'."
                )

        case _:
            raise ValueError(f"Unsupported source grid '{source_grid}'.")

    return result

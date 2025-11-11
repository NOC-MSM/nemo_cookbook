"""
interpolate.py

Description:
This module includes functions to linearly interpolate between
horizontal grids of NEMO ocean general circulation model domains.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import xarray as xr


def interpolate_grid(
    da: xr.DataArray,
    source_grid: str,
    target_grid: str,
    iperio: bool = False,
    ijk_names: tuple = ('i', 'j', 'k')
) -> xr.DataArray:
    """
    Interpolate variable between NEMO horizontal grids.

    Parameters:
    -----------
    da : xr.DataArray
        DataArray defined on source grid.
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
        Variable linearly interpolated onto NEMO target grid.
    """
    # -- Get dimension names -- #
    i_name, j_name, k_name = ijk_names

    match source_grid:
        # -- Linearly interpolate T-grid scalar variables -- #
        case "T":
            if target_grid == "U":
                # T -> U grid:
                if iperio:
                    # Zonally periodic domain:
                    da_interp = 0.5 * (da.roll({i_name: 1}) + da)
                else:
                    # Non-periodic domain:
                    da_interp = 0.5 * (da + da.shift({i_name: -1}))
            elif target_grid == "V":
                # T -> V grid:
                da_interp = 0.5 * (da + da.shift({j_name: -1}))
            elif target_grid == "F":
                # T -> F grid:
                if iperio:
                    # Zonally periodic domain:
                    da_interp = 0.25 * (da.roll({i_name: 1}).shift({j_name: -1}) +
                                        da.roll({i_name: 1}) +
                                        da.shift({j_name: -1}) +
                                        da
                                        )
                else:
                    # Non-periodic domain:
                    da_interp = 0.25 * (da.shift({i_name: -1}).shift({j_name: -1}) +
                                        da.shift({i_name: -1}) +
                                        da.shift({j_name: -1}) +
                                        da
                                        )
            else:
                raise ValueError(f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'.")
    
        # -- Linearly interpolate U-grid flux variables -- #
        case "U":
            if target_grid == "T":
                # U -> T grid:
                if iperio:
                    # Zonally periodic domain:
                    da_interp = 0.5 * (da + da.roll({i_name: 1}))
                else:
                    # Non-periodic domain:
                    da_interp = 0.5 * (da + da.shift({i_name: 1}))
            elif target_grid == "V":
                # U -> V grid:
                if iperio:
                    # Zonally periodic domain:
                    da_interp = 0.25 * (da.roll({i_name: 1}).shift({j_name: 1}) +
                                        da.roll({i_name: 1}) +
                                        da.shift({j_name: 1}) +
                                        da
                                        )
                else:
                    # Non-periodic domain:
                    da_interp = 0.25 * (da.shift({i_name: 1}).shift({j_name: 1}) +
                                        da.shift({i_name: 1}) +
                                        da.shift({j_name: 1}) +
                                        da
                                        )
            else:
                raise ValueError(f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'.")
    
        # -- Linearly interpolate V-grid flux variables -- #
        case "V":
            if target_grid == "T":
                # V -> T grid:
                da_interp = 0.5 * (da + da.shift({j_name: 1}))
            elif target_grid == "U":
                # V -> U grid:
                da_interp = 0.25 * (da.shift({i_name: 1}).shift({j_name: 1}) +
                                    da.shift({i_name: 1}) +
                                    da.shift({j_name: 1}) +
                                    da
                                    )
            else:
                raise ValueError(f"Unsupported grid transformation from '{source_grid}' to '{target_grid}'.")

        case _:
            raise ValueError(f"Unsupported source grid '{source_grid}'.")

    return da_interp

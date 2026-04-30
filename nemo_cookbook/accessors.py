"""
accesors.py

Description:
This module includes accessor functions to transform core data structures
used in the NEMO Cookbook library into expected formats of third-party
libraries.


Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    # Avoid circular import at runtime:
    from nemo_cookbook.nemodataarray import NEMODataArray

def create_xesmf_dataset(
    nda: NEMODataArray,
    mask: bool = True
    ) -> xr.Dataset:
    """
    Create an xESMF-compatible dataset from a variable stored in
    a NEMODataArray.

    The resulting dataset includes the following expected coordinates:
    - lon (longitudes of grid cell centers)
    - lat (latitudes of grid cell centers)
    - lon_b (longitudes of grid cell boundaries)
    - lat_b (latitudes of grid cell boundaries)
    - mask (boolean land-sea mask identifying valid grid cells) [ Optional ]

    Parameters
    ----------
    nda : NEMODataArray
        NEMODataArray containing the variable to transform to an xESMF-compatible dataset.
    mask : bool, optional
        Whether to include a 2-dimensional land-sea mask in the output dataset.
        Default is True.

    Returns
    -------
    xr.Dataset
        Dataset containing variable and associated geographical coordinates
        formatted for use with xESMF.
     """
    # Define xarray.Dataset from NEMODataArray:
    ds = nda.to_dataset()

    # Update coordinate names to be xesmf-compliant:
    ds = ds.rename({f"glam{nda._grid_suffix}": "lon",
                    f"gphi{nda._grid_suffix}": "lat"
                    })

    if mask:
        # Optionally add 2-dimensional land-sea mask:
        mask_name = f"{nda._grid_suffix}maskutil"
        mask_data = nda._tree[f"{nda._grid}/{mask_name}"].sel_like(nda).data
        mask_data = mask_data.rename({f"glam{nda._grid_suffix}": "lon",
                                      f"gphi{nda._grid_suffix}": "lat"
                                      })
        ds["mask"] = mask_data
        ds["mask"].attrs.update({"standard_name": "land_sea_mask",
                                "long_name": "Land-Sea Mask",
                                "units": "1"
                                })

    if nda.iperio:
        # === Zonal Periodic === #
        # Clipping NEMO model j-dimension to exclude undefined grid cell corners:
        ds = ds.isel(j=slice(1, None))
        # Create grid cell corner coordinates:
        glamf = nda._tree[f"{nda._grid.replace('T', 'F')}/glamf"].data
        gphif = nda._tree[f"{nda._grid.replace('T', 'F')}/gphif"].data
        # Wrap grid cell corner coordinates along i-dimension:
        glamf = glamf.pad(i=(1, 0), mode='wrap')
        gphif = gphif.pad(i=(1, 0), mode='wrap')
    else:
        # === Non-Zonally Periodic === #
        # Clipping NEMO model (i, j)-dimensions to exclude undefined grid cell corners:
        ds = ds.isel(j=slice(1, None), i=slice(1, None))
        # Create grid cell corner coordinates:
        glamf = nda._tree[f"{nda._grid.replace('T', 'F')}/glamf"].data
        gphif = nda._tree[f"{nda._grid.replace('T', 'F')}/gphif"].data
        # Select grid cell corners to match dimensions of clipped dataset:
        glamf = glamf.sel(i=nda["i"].data + 0.5, j=nda["j"].data + 0.5)
        gphif = gphif.sel(i=nda["i"].data + 0.5, j=nda["j"].data + 0.5)

    # === Add grid cell corner geographical coordinates === #
    lon_b=glamf.rename({"j": "j_b", "i": "i_b"})
    lat_b=gphif.rename({"j": "j_b", "i": "i_b"})

    # Remove any non-standard coordinate variables:
    if ("glamf" in lon_b.coords) & ("gphif" in lat_b.coords):
        lon_b = lon_b.drop_vars(["glamf", "gphif"])
    if ("glamf" in lat_b.coords) & ("gphif" in lat_b.coords):
        lat_b = lat_b.drop_vars(["glamf", "gphif"])

    ds = ds.assign_coords(lon_b=lon_b,
                          lat_b=lat_b
                          )

    # === Update coordinate CF-metadata === #
    # Longitude, Latitude coordinates:
    ds["lon"].attrs.update({"standard_name": "longitude",
                            "long_name": "Longitude",
                            "units": "degrees_east"
                            })

    ds["lat"].attrs.update({"standard_name": "latitude",
                            "long_name": "Latitude",
                            "units": "degrees_north"
                            })

    # Longitude and Latitude of Grid Cell Corners coordinates:
    ds["lon_b"].attrs.update({"standard_name": "longitude",
                              "long_name": "Longitude of Grid Cell Corners",
                              "units": "degrees_east"
                              })

    ds["lat_b"].attrs.update({"standard_name": "latitude",
                              "long_name": "Latitude of Grid Cell Corners",
                              "units": "degrees_north"
                              })
    
    return ds
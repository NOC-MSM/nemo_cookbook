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
    ds = ds.rename({f"{nda._dom_prefix}glam{nda._grid_suffix}": "lon",
                    f"{nda._dom_prefix}gphi{nda._grid_suffix}": "lat"
                    })

    if mask:
        # Optionally add 2-dimensional land-sea mask:
        mask_name = f"{nda._grid_suffix}maskutil"
        mask_data = nda._tree[f"{nda._grid}/{mask_name}"].sel_like(nda).data
        mask_data = mask_data.rename({f"{nda._dom_prefix}glam{nda._grid_suffix}": "lon",
                                      f"{nda._dom_prefix}gphi{nda._grid_suffix}": "lat"
                                      })
        ds["mask"] = mask_data
        ds["mask"].attrs.update({"standard_name": "land_sea_mask",
                                "long_name": "Land-Sea Mask",
                                "units": "1"
                                })

    gridF = nda._tree._get_grid_paths(dom=nda._dom)["gridF"]
    glamf_name = f"{nda._dom_prefix}glamf"
    gphif_name = f"{nda._dom_prefix}gphif"

    if nda.iperio:
        # === Zonal Periodic === #
        # Clipping NEMO model j-dimension to exclude undefined grid cell corners:
        ds = ds.isel({nda.j_name: slice(1, None)})
        # Create grid cell corner coordinates:
        glamf = nda._tree[f"{gridF}/{glamf_name}"].data
        gphif = nda._tree[f"{gridF}/{gphif_name}"].data
        # Wrap grid cell corner coordinates along i-dimension:
        glamf = glamf.pad({nda.i_name: (1, 0)}, mode='wrap')
        gphif = gphif.pad({nda.i_name: (1, 0)}, mode='wrap')
    else:
        # === Non-Zonally Periodic === #
        # Clipping NEMO model (i, j)-dimensions to exclude undefined grid cell corners:
        ds = ds.isel({nda.j_name: slice(1, None), nda.i_name: slice(1, None)})
        # Create grid cell corner coordinates:
        glamf = nda._tree[f"{gridF}/{glamf_name}"].data
        gphif = nda._tree[f"{gridF}/{gphif_name}"].data
        # Select grid cell corners to match dimensions of clipped dataset:
        glamf = glamf.sel({nda.i_name: nda[nda.i_name].data + 0.5, nda.j_name: nda[nda.j_name].data + 0.5})
        gphif = gphif.sel({nda.i_name: nda[nda.i_name].data + 0.5, nda.j_name: nda[nda.j_name].data + 0.5})

    # === Add grid cell corner geographical coordinates === #
    lon_b = glamf.rename({nda.j_name: "j_b", nda.i_name: "i_b"})
    lat_b = gphif.rename({nda.j_name: "j_b", nda.i_name: "i_b"})

    # Remove any non-standard coordinate variables:
    if (glamf_name in lon_b.coords) & (gphif_name in lat_b.coords):
        lon_b = lon_b.drop_vars([glamf_name, gphif_name])
    if (glamf_name in lat_b.coords) & (gphif_name in lat_b.coords):
        lat_b = lat_b.drop_vars([glamf_name, gphif_name])

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
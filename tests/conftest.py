"""
conftest.py

Description:
Defines NEMO Cookbook configuration and fixtures.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import pytest
import numpy as np
import xarray as xr
from nemo_cookbook import NEMODataTree


@pytest.fixture
def example_global_nemodatatree() -> NEMODataTree:
    """
    Fixture to create an example, idealised global NEMODataTree.
    The global model domain is zonally periodic (i.e., iperio = True).

    Returns
    -------
    NEMODataTree
        Example NEMODataTree with idealised grid and variable data.
    """
    # -- Define grid dimensions -- #
    nt, nk, nj, ni = 3, 5, 10, 10

    # -- Vertical grid scale factors -- #
    # Time-dependent for QCO case:
    e3_data = 50 * np.ones((nt, nk, nj, ni))
    e3_dims = ("time_counter", "k", "j", "i")

    # -- Create Example NEMODataTree -- #
    nemo = NEMODataTree()

    # Add NEMODataTree Attributes:
    nemo.attrs = {"nftype": None, "iperio": False}

    # T-points:
    nemo["gridT"] = xr.Dataset(data_vars={
        'e1t': (("j", "i"), np.ones((nj, ni))),
        'e2t': (("j", "i"), np.ones((nj, ni))),
        'e3t': (e3_dims, e3_data),
        'tmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridT"] = (nemo["gridT"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1),
                                    gphit=(("j"), np.linspace(-90, 90, 2*nj)[::2]),
                                    glamt=(("i"), np.linspace(-180, 180, 2*ni)[::2]),
                                    deptht=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as T-points:
    nemo["gridT"]["tmask"][:, [0, -1], :] = False
    nemo["gridT"]["tmask"][:, 1, 4:6] = False
    # Add tmaskutil:
    nemo["gridT"]["tmaskutil"] = nemo["gridT"]["tmask"].isel(k=0).copy()

    # U-points:
    nemo["gridU"] = xr.Dataset(data_vars={
        'e1u': (("j", "i"), np.ones((nj, ni))),
        'e2u': (("j", "i"), np.ones((nj, ni))),
        'e3u': (e3_dims, e3_data),
        'umask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridU"] = (nemo["gridU"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1) + 0.5,
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1),
                                    gphiu=(("j"), np.linspace(-90, 90, 2*nj)[::2]),
                                    glamu=(("i"), np.linspace(-180, 180, 2*ni)[1::2]),
                                    depthu=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as U-points:
    nemo["gridU"]["umask"][:, [0, -1], :] = False
    nemo["gridU"]["umask"][:, 1, 3:6] = False
    # Add umaskutil:
    nemo["gridU"]["umaskutil"] = nemo["gridU"]["umask"].isel(k=0).copy()

    # V-points:
    nemo["gridV"] = xr.Dataset(data_vars={
        'e1v': (("j", "i"), np.ones((nj, ni))),
        'e2v': (("j", "i"), np.ones((nj, ni))),
        'e3v': (e3_dims, e3_data),
        'vmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridV"] = (nemo["gridV"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1) + 0.5,
                                    k=np.arange(1, nk + 1),
                                    gphiv=(("j"), np.linspace(-90, 90, 2*nj)[1::2]),
                                    glamv=(("i"), np.linspace(-180, 180, 2*ni)[::2]),
                                    depthv=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as V-points:
    nemo["gridV"]["vmask"][:, 0, :] = False
    nemo["gridV"]["vmask"][:, -2:, :] = False
    nemo["gridV"]["vmask"][:, 1, 4:6] = False
    # Add vmaskutil:
    nemo["gridV"]["vmaskutil"] = nemo["gridV"]["vmask"].isel(k=0).copy()

    # W-points:
    nemo["gridW"] = xr.Dataset(data_vars={
        'e1w': (("j", "i"), np.ones((nj, ni))),
        'e2w': (("j", "i"), np.ones((nj, ni))),
        'e3w': (e3_dims, e3_data),
        'wmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridW"] = (nemo["gridW"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1) - 0.5,
                                    gphit=(("j"), np.linspace(-90, 90, 2*nj)[::2]),
                                    glamt=(("i"), np.linspace(-180, 180, 2*ni)[::2]),
                                    depthw=(("k"), np.arange(0, 250, 50))
                                    )
                    )
    # Define land-sea mask as W-points:
    nemo["gridW"]["wmask"][:, [0, -1], :] = False
    nemo["gridW"]["wmask"][:, 1, 4:6] = False
    # Add wmaskutil:
    nemo["gridW"]["wmaskutil"] = nemo["gridW"]["wmask"].isel(k=0).copy()

    # F-points:
    nemo["gridF"] = xr.Dataset(data_vars={
        'e1f': (("j", "i"), np.ones((nj, ni))),
        'e2f': (("j", "i"), np.ones((nj, ni))),
        'e3f': (e3_dims, e3_data),
        'fmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridF"] = (nemo["gridF"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1) + 0.5,
                                    j=np.arange(1, nj + 1) + 0.5,
                                    k=np.arange(1, nk + 1),
                                    gphif=(("j"), np.linspace(-90, 90, 2*nj)[1::2]),
                                    glamf=(("i"), np.linspace(-180, 180, 2*ni)[1::2]),
                                    depthf=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as F-points:
    nemo["gridF"]["fmask"][:, 0, :] = False
    nemo["gridF"]["fmask"][:, -2:, :] = False
    nemo["gridF"]["fmask"][:, 1, 3:6] = False
    # Add fmaskutil:
    nemo["gridF"]["fmaskutil"] = nemo["gridF"]["fmask"].isel(k=0).copy()

    # -- Add Example Scalar Variable -- #
    time_data = np.arange(np.datetime64('2000-01'), np.datetime64('2000-04'))
    # Add conservative temperature (10 degC) field to T-grid:
    nemo['gridT']['thetao_con'] = xr.DataArray(data=10 * np.ones((nt, nk, nj, ni)),
                                               dims=('time_counter', 'k', 'j', 'i'),
                                               coords={'time_counter': time_data,
                                                       'k': nemo['gridT']['k'],
                                                       'j': nemo['gridT']['j'],
                                                       'i': nemo['gridT']['i']
                                                       }
                                                    )
    # Apply tmask to thetao_con:
    nemo['gridT']['thetao_con'] = nemo['gridT']['thetao_con'].where(nemo['gridT']['tmask'])

    # Add sea surface temperature (10 degC) field to T-grid:
    nemo['gridT']['tos_con'] = xr.DataArray(data=10 * np.ones((nt, nj, ni)),
                                               dims=('time_counter', 'j', 'i'),
                                               coords={'time_counter': time_data,
                                                       'j': nemo['gridT']['j'],
                                                       'i': nemo['gridT']['i']
                                                       }
                                                    )
    # Apply tmaskutil to tos_con:
    nemo['gridT']['tos_con'] = nemo['gridT']['tos_con'].where(nemo['gridT']['tmaskutil'])

    # -- Add Example Vector Variables -- #
    # Add zonal velocity (2 m/s) field to U-grid:
    nemo['gridU']['uo'] = xr.DataArray(data=2 * np.ones((nt, nk, nj, ni)),
                                       dims=('time_counter', 'k', 'j', 'i'),
                                       coords={'time_counter': time_data,
                                               'k': nemo['gridU']['k'],
                                               'j': nemo['gridU']['j'],
                                               'i': nemo['gridU']['i']
                                               }
                                            )
    # Apply umask to vo:
    nemo['gridU']['uo'] = nemo['gridU']['uo'].where(nemo['gridU']['umask'])

    # Add meridional velocity (2 m/s) field to V-grid:
    nemo['gridV']['vo'] = xr.DataArray(data=2 * np.ones((nt, nk, nj, ni)),
                                       dims=('time_counter', 'k', 'j', 'i'),
                                       coords={'time_counter': time_data,
                                               'k': nemo['gridV']['k'],
                                               'j': nemo['gridV']['j'],
                                               'i': nemo['gridV']['i']
                                               }
                                            )
    # Apply vmask to vo:
    nemo['gridV']['vo'] = nemo['gridV']['vo'].where(nemo['gridV']['vmask'])

    return nemo

@pytest.fixture
def example_regional_nemodatatree() -> NEMODataTree:
    """
    Fixture to create an example, idealised regional NEMODataTree.
    The regional model domain is not zonally periodic (i.e., iperio=False).

    Returns
    -------
    NEMODataTree
        Example regional NEMODataTree with idealised grid and variable data.
    """
    # -- Define grid dimensions -- #
    nt, nk, nj, ni = 3, 5, 10, 10

    # -- Vertical grid scale factors -- #
    # Time-dependent for QCO case:
    e3_data = 50 * np.ones((nt, nk, nj, ni))
    e3_dims = ("time_counter", "k", "j", "i")

    # -- Create Example NEMODataTree -- #
    nemo = NEMODataTree()

    # Add NEMODataTree Attributes:
    nemo.attrs = {"nftype": "T", "iperio": True}

    # T-points:
    nemo["gridT"] = xr.Dataset(data_vars={
        'e1t': (("j", "i"), np.ones((nj, ni))),
        'e2t': (("j", "i"), np.ones((nj, ni))),
        'e3t': (e3_dims, e3_data),
        'tmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridT"] = (nemo["gridT"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1),
                                    gphit=(("j"), np.linspace(-60, 0, 2*nj)[::2]),
                                    glamt=(("i"), np.linspace(25, 80, 2*ni)[::2]),
                                    deptht=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as T-points:
    nemo["gridT"]["tmask"][:, [0, -1], :] = False
    nemo["gridT"]["tmask"][:, :, [0, -1]] = False
    nemo["gridT"]["tmask"][:, 1, 4:6] = False
    # Add tmaskutil:
    nemo["gridT"]["tmaskutil"] = nemo["gridT"]["tmask"].isel(k=0).copy()

    # U-points:
    nemo["gridU"] = xr.Dataset(data_vars={
        'e1u': (("j", "i"), np.ones((nj, ni))),
        'e2u': (("j", "i"), np.ones((nj, ni))),
        'e3u': (e3_dims, e3_data),
        'umask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridU"] = (nemo["gridU"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1) + 0.5,
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1),
                                    gphiu=(("j"), np.linspace(-60, 0, 2*nj)[::2]),
                                    glamu=(("i"), np.linspace(25, 80, 2*ni)[1::2]),
                                    depthu=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as U-points:
    nemo["gridU"]["umask"][:, [0, -1], :] = False
    nemo["gridU"]["umask"][:, :, [0, -1]] = False
    nemo["gridU"]["umask"][:, 1, 3:6] = False
    # Add umaskutil:
    nemo["gridU"]["umaskutil"] = nemo["gridU"]["umask"].isel(k=0).copy()

    # V-points:
    nemo["gridV"] = xr.Dataset(data_vars={
        'e1v': (("j", "i"), np.ones((nj, ni))),
        'e2v': (("j", "i"), np.ones((nj, ni))),
        'e3v': (e3_dims, e3_data),
        'vmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridV"] = (nemo["gridV"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1) + 0.5,
                                    k=np.arange(1, nk + 1),
                                    gphiv=(("j"), np.linspace(-60, 0, 2*nj)[1::2]),
                                    glamv=(("i"), np.linspace(25, 80, 2*ni)[::2]),
                                    depthv=(("k"), np.arange(25, 250, 50))
                                    )
                    )
    # Define land-sea mask as V-points:
    nemo["gridV"]["vmask"][:, 0, :] = False
    nemo["gridV"]["vmask"][:, -2:, :] = False
    nemo["gridV"]["vmask"][:, :, [0, -1]] = False
    nemo["gridV"]["vmask"][:, 1, 4:6] = False
    # Add vmaskutil:
    nemo["gridV"]["vmaskutil"] = nemo["gridV"]["vmask"].isel(k=0).copy()

    # W-points:
    nemo["gridW"] = xr.Dataset(data_vars={
        'e1w': (("j", "i"), np.ones((nj, ni))),
        'e2w': (("j", "i"), np.ones((nj, ni))),
        'e3w': (e3_dims, e3_data),
        'wmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridW"] = (nemo["gridW"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1),
                                    j=np.arange(1, nj + 1),
                                    k=np.arange(1, nk + 1) - 0.5,
                                    gphit=(("j"), np.linspace(-60, 0, 2*nj)[1::2]),
                                    glamt=(("i"), np.linspace(25, 80, 2*ni)[::2]),
                                    depthw=(("k"), np.arange(0, 250, 50))
                                    )
                    )
    # Define land-sea mask as W-points:
    nemo["gridW"]["wmask"][:, [0, -1], :] = False
    nemo["gridW"]["wmask"][:, :, [0, -1]] = False
    nemo["gridW"]["wmask"][:, 1, 4:6] = False
    # Add wmaskutil:
    nemo["gridW"]["wmaskutil"] = nemo["gridW"]["wmask"].isel(k=0).copy()

    # F-points:
    nemo["gridF"] = xr.Dataset(data_vars={
        'e1f': (("j", "i"), np.ones((nj, ni))),
        'e2f': (("j", "i"), np.ones((nj, ni))),
        'e3f': (e3_dims, e3_data),
        'depthf': (("k"), np.linspace(0, 500, nk)),
        'fmask': (("k", "j", "i"), np.ones((nk, nj, ni)).astype(bool)),
    })
    nemo["gridF"] = (nemo["gridF"]
                     .dataset
                     .assign_coords(i=np.arange(1, ni + 1) + 0.5,
                                    j=np.arange(1, nj + 1) + 0.5,
                                    k=np.arange(1, nk + 1),
                                    gphif=(("j"), np.linspace(-60, 0, 2*nj)[1::2]),
                                    glamf=(("i"), np.linspace(25, 80, 2*ni)[1::2]),
                                    depthf=(("k"), np.arange(0, 250, 50))
                                    )
                    )
    # Define land-sea mask as F-points:
    nemo["gridF"]["fmask"][:, 0, :] = False
    nemo["gridF"]["fmask"][:, -2:, :] = False
    nemo["gridF"]["fmask"][:, :, [0, -1]] = False
    nemo["gridF"]["fmask"][:, 1, 3:6] = False
    # Add fmaskutil:
    nemo["gridF"]["fmaskutil"] = nemo["gridF"]["fmask"].isel(k=0).copy()

    # -- Add Example Scalar Variable -- #
    time_data = np.arange(np.datetime64('2000-01'), np.datetime64('2000-04'))
    # Add conservative temperature (10 degC) field to T-grid:
    nemo['gridT']['thetao_con'] = xr.DataArray(data=10 * np.ones((nt, nk, nj, ni)),
                                               dims=('time_counter', 'k', 'j', 'i'),
                                               coords={'time_counter': time_data,
                                                       'k': nemo['gridT']['k'],
                                                       'j': nemo['gridT']['j'],
                                                       'i': nemo['gridT']['i']
                                                       }
                                                    )
    # Apply tmask to thetao_con:
    nemo['gridT']['thetao_con'] = nemo['gridT']['thetao_con'].where(nemo['gridT']['tmask'])

    # Add sea surface temperature (10 degC) field to T-grid:
    nemo['gridT']['tos_con'] = xr.DataArray(data=10 * np.ones((nt, nj, ni)),
                                               dims=('time_counter', 'j', 'i'),
                                               coords={'time_counter': time_data,
                                                       'j': nemo['gridT']['j'],
                                                       'i': nemo['gridT']['i']
                                                       }
                                                    )
    # Apply tmaskutil to tos_con:
    nemo['gridT']['tos_con'] = nemo['gridT']['tos_con'].where(nemo['gridT']['tmaskutil'])

    # -- Add Example Vector Variables -- #
    # Add zonal velocity (2 m/s) field to U-grid:
    nemo['gridU']['uo'] = xr.DataArray(data=2 * np.ones((nt, nk, nj, ni)),
                                       dims=('time_counter', 'k', 'j', 'i'),
                                       coords={'time_counter': time_data,
                                               'k': nemo['gridU']['k'],
                                               'j': nemo['gridU']['j'],
                                               'i': nemo['gridU']['i']
                                               }
                                            )
    # Apply umask to vo:
    nemo['gridU']['uo'] = nemo['gridU']['uo'].where(nemo['gridU']['umask'])

    # Add meridional velocity (2 m/s) field to V-grid:
    nemo['gridV']['vo'] = xr.DataArray(data=2 * np.ones((nt, nk, nj, ni)),
                                       dims=('time_counter', 'k', 'j', 'i'),
                                       coords={'time_counter': time_data,
                                               'k': nemo['gridV']['k'],
                                               'j': nemo['gridV']['j'],
                                               'i': nemo['gridV']['i']
                                               }
                                            )
    # Apply vmask to vo:
    nemo['gridV']['vo'] = nemo['gridV']['vo'].where(nemo['gridV']['vmask'])

    return nemo

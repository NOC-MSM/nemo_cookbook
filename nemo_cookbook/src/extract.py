"""
extract.py

Description: Functions to extract observational sections from
NEMO ocean general circulation model output.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 26/04/2025
"""

# -- Import dependencies -- #
import sys
import logging
import numpy as np
import xarray as xr
from functools import partial
from .extract_utils import _nearest_ji_coords, _get_section_coords

# -- Internal Preprocessing Functions -- #
def process_Um(ds: xr.Dataset,
               vars: str | list[str],
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess zonal velocities (m/s), zonal eddy-induced velocities (m/s)
    and vertical grid cell thicknesses (m) on U- grid cell faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    vars : str | list[str]
        Variable names to be extracted from the dataset. Options are 'uo',
        'uo_eiv', 'e3u'.
    var_map : dict
        Dictionary mapping variable names to their corresponding dataset keys.
    x_index : xarray.DataArray
        x indexes of U- grid cell faces defining section on NEMO model grid.
    y_index : xarray.DataArray
        y indexes of U- grid cell faces defining section on NEMO model grid.
    stations : xarray.DataArray
        Station indexes of U- grid cell faces defining section on NEMO model grid.
    longitudes : xarray.DataArray
        Longitude coordinates of U- grid cell faces defining section on NEMO model grid.
    latitudes : xarray.DataArray
        Latitude coordinates of U- grid cell faces defining section on NEMO model grid.

    Returns
    -------
    ds_Um : xarray.Dataset
        Dataset including zonal velocities (m/s), eddy-induced velocities (m/s)
        and vertical grid cell thicknesses (m) on U- grid cell faces defining
        section on NEMO model grid.
    """
    # Define empty list to store variables:
    var_list = []

    if 'uo' in vars:
        # Extract U- velocities (m/s):
        if 'uo' in var_map:
            uo = ds[var_map['uo']]
        else:
            uo = ds['uo']

        uo = uo.isel(x=x_index, y=y_index)
        uo.name = 'velocity'
        uo = uo.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})
        var_list.append(uo)

    if 'uo_eiv' in vars:
        # Extract eddy-induced U- velocities (m/s):
        if 'uo_eiv' in var_map:
            uo_eiv = ds[var_map['uo_eiv']]
        else:
            uo_eiv = ds['uo_eiv']

        uo_eiv = uo_eiv.isel(x=x_index, y=y_index)
        uo_eiv.name = 'eddy_induced_velocity'
        uo_eiv = uo_eiv.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})
        var_list.append(uo_eiv)

    if 'e3u' in vars:
        # Extract vertical grid cell thickness (m):
        if 'e3u' in var_map:
            e3u = ds[var_map['e3u']]
        else:
            e3u = ds['e3u']

        e3u = e3u.isel(x=x_index, y=y_index)
        e3u.name = 'dz'
        e3u = e3u.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})
        var_list.append(e3u)

    # Return variable or combine into single Dataset:
    if len(var_list) == 1:
        ds_Um = var_list[0]
    else:
        ds_Um = xr.merge(var_list)

    # Assign coordinates to the dataset:
    ds_Um = (ds_Um.
            assign_coords({'station': stations,
                            'longitude': longitudes,
                            'latitude': latitudes
                            })
            .rename({'depthu':'depth'})
            )

    return ds_Um


def process_Up(ds: xr.Dataset,
               vars: str | list[str],
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess zonal velocities (m/s) & vertical grid cell thickness (m)
    on U+ faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    vars : str | list[str]
        Variable names to be extracted from the dataset. Options are 'uo',
        'uo_eiv', 'e3u'.
    var_map : dict
        Dictionary mapping variable names to their corresponding dataset keys.
    x_index : xarray.DataArray
        x indexes of U+ grid cell faces defining section on NEMO model grid.
    y_index : xarray.DataArray
        y indexes of U+ grid cell faces defining section on NEMO model grid.
    stations : xarray.DataArray
        Station indexes of U+ grid cell faces defining section on NEMO model grid.
    longitudes : xarray.DataArray
        Longitude coordinates of U+ grid cell faces defining section on NEMO model grid.
    latitudes : xarray.DataArray
        Latitude coordinates of U+ grid cell faces defining section on NEMO model grid.

    Returns
    -------
    ds_Up : xarray.Dataset
        Dataset including zonal velocities (m/s) and vertical grid cell thickness (m)
        on U+ grid cell faces defining section on NEMO model grid.
    """
    # Define empty list to store variables:
    var_list = []

    if 'uo' in vars:
        # Extract U+ velocities (m/s):
        if 'uo' in var_map:
            up = -ds[var_map['uo']]
        else:
            up = -ds['uo']

        up = up.isel(x=x_index, y=y_index)
        up.name = 'velocity'
        up = up.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})
        var_list.append(up)

    if 'uo_eiv' in vars:
        # Extract eddy-induced U+ velocities (m/s):
        if 'uo_eiv' in var_map:
            up_eiv = -ds[var_map['uo_eiv']]
        else:
            up_eiv = -ds['uo_eiv']

        up_eiv = up_eiv.isel(x=x_index, y=y_index)
        up_eiv.name = 'eddy_induced_velocity'
        up_eiv = up_eiv.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})
        var_list.append(up_eiv)

    if 'e3u' in vars:
        # Extract vertical grid cell thickness (m):
        if 'e3u' in var_map:
            e3up = ds[var_map['e3u']]
        else:
            e3up = ds['e3u']
        
        e3up = e3up.isel(x=x_index, y=y_index)
        e3up.name = 'dz'
        e3up = e3up.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})
        var_list.append(e3up)

    # Return variable or combine into single Dataset:
    if len(var_list) == 1:
        ds_Up = var_list[0]
    else:
        ds_Up = xr.merge(var_list)

    # Assign coordinates to the dataset:
    ds_Up = (ds_Up.
             assign_coords({'station': stations,
                            'longitude': longitudes,
                            'latitude': latitudes
                            })
            .rename({'depthu':'depth'})
            )

    return ds_Up


def process_Vp(ds: xr.Dataset,
               vars: str | list[str],
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess zonal velocities (m/s) & vertical grid cell thickness (m)
    on V+ faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    vars : str | list[str]
        Variable names to be extracted from the dataset. Options are 'vo',
        'vo_eiv', 'e3v'.
    var_map : dict
        Dictionary mapping variable names to their corresponding dataset keys.
    x_index : xarray.DataArray
        x indexes of V+ grid cell faces defining section on NEMO model grid.
    y_index : xarray.DataArray
        y indexes of V+ grid cell faces defining section on NEMO model grid.
    stations : xarray.DataArray
        Station indexes of V+ grid cell faces defining section on NEMO model grid.
    longitudes : xarray.DataArray
        Longitude coordinates of V+ grid cell faces defining section on NEMO model grid.
    latitudes : xarray.DataArray
        Latitude coordinates of V+ grid cell faces defining section on NEMO model grid.

    Returns
    -------
    ds_Vp : xarray.Dataset
        Dataset including meridional velocities (m/s) and vertical grid cell thickness (m)
        on V+ grid cell faces defining section on NEMO model grid.
    """
    # Define empty list to store variables:
    var_list = []

    # Extract V+ velocities (m/s):
    if 'vo' in vars:
        if 'vo' in var_map:
            vp = ds[var_map['vo']]
        else:
            vp = ds['vo']

        vp = vp.isel(x=x_index, y=y_index)
        vp.name = 'velocity'
        vp = vp.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})
        var_list.append(vp)

    # Extract eddy-induced V+ velocities (m/s):
    if 'vo_eiv' in vars:
        if 'vo_eiv' in var_map:
            vp_eiv = ds[var_map['vo_eiv']]
        else:
            vp_eiv = ds['vo_eiv']

        vp_eiv = vp_eiv.isel(x=x_index, y=y_index)
        vp_eiv.name = 'eddy_induced_velocity'
        vp_eiv = vp_eiv.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})
        var_list.append(vp_eiv)

    # Extract vertical grid cell thickness (m):
    if 'e3v' in vars:
        if 'e3v' in var_map:
            e3vp = ds[var_map['e3v']]
        else:
            e3vp = ds['e3v']

        e3vp = e3vp.isel(x=x_index, y=y_index)
        e3vp.name = 'dz'
        e3vp = e3vp.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})
        var_list.append(e3vp)

    # Return variable or combine into single Dataset:
    if len(var_list) == 1:
        ds_Vp = var_list[0]
    else:
        ds_Vp = xr.merge(var_list)

    # Assign coordinates to the dataset:
    ds_Vp = (ds_Vp.
             assign_coords({'station': stations,
                            'longitude': longitudes,
                            'latitude': latitudes
                            })
            .rename({'depthv':'depth'})
            )

    return ds_Vp


def process_Tu(ds: xr.Dataset,
               vars: str | list[str],
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess temperature & salinity interpolated on U grid cell faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    vars : str | list[str]
        Variable names to be extracted from the dataset. Options are 'temp',
        'sal'.
    var_map : dict
        Dictionary mapping variable names to their corresponding dataset keys.
    x_index : xarray.DataArray
        x indexes of U grid cell faces defining section on NEMO model grid.
    y_index : xarray.DataArray
        y indexes of U grid cell faces defining section on NEMO model grid.
    stations : xarray.DataArray
        Station indexes of U grid cell faces defining section on NEMO model grid.
    longitudes : xarray.DataArray
        Longitude coordinates of U grid cell faces defining section on NEMO model grid.
    latitudes : xarray.DataArray
        Latitude coordinates of U grid cell faces defining section on NEMO model grid.

    Returns
    -------
    ds_Tu : xarray.Dataset
        Dataset including temperature (C) and salinity (psu | g/kg) on U grid cell
        faces defining section on NEMO model grid.
    """
    # Define empty list to store variables:
    var_list = []

    # Extract U temperature (C):
    if 'temp' in vars:
        if 'temp' in var_map:
            temp = ds[var_map['temp']]
        else:
            temp = ds['temp']

        temp_u = 0.5 * (temp.isel(x=x_index, y=y_index) + temp.isel(x=x_index+1, y=y_index))
        temp_u.name = 'temp'
        temp_u = temp_u.assign_attrs({'standard_name': 'sea_water_temperature'})
        var_list.append(temp_u)

    # Extract U salinity (psu | g/kg):
    if 'sal' in vars:
        if 'sal' in var_map:
            sal = ds[var_map['sal']]
        else:
            sal = ds['sal']

        sal_u = 0.5 * (sal.isel(x=x_index, y=y_index) + sal.isel(x=x_index+1, y=y_index))
        sal_u.name = 'sal'
        sal_u = sal_u.assign_attrs({'standard_name': 'sea_water_salinity'})
        var_list.append(sal_u)

    # Return variable or combine into single Dataset:
    if len(var_list) == 1:
        ds_Tu = var_list[0]
    else:
        ds_Tu = xr.merge(var_list)

    # Assign coordinates to the dataset:
    ds_Tu = (ds_Tu.
             assign_coords({'station': stations,
                            'longitude': longitudes,
                            'latitude': latitudes
                            })
            .rename({'deptht':'depth'})
            )

    return ds_Tu


def process_Tv(ds: xr.Dataset,
                vars: str | list[str],
                var_map: dict,
                x_index: xr.DataArray,
                y_index: xr.DataArray,
                stations: xr.DataArray,
                longitudes: xr.DataArray,
                latitudes: xr.DataArray,
                ) -> xr.Dataset:
    """
    Preprocess temperature & salinity interpolated on V grid cell faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    vars : str | list[str]
        Variable names to be extracted from the dataset. Options are 'temp',
        'sal'.
    var_map : dict
        Dictionary mapping variable names to their corresponding dataset keys.
    x_index : xarray.DataArray
        x indexes of V grid cell faces defining section on NEMO model grid.
    y_index : xarray.DataArray
        y indexes of V grid cell faces defining section on NEMO model grid.
    stations : xarray.DataArray
        Station indexes of V grid cell faces defining section on NEMO model grid.
    longitudes : xarray.DataArray
        Longitude coordinates of V grid cell faces defining section on NEMO model grid.
    latitudes : xarray.DataArray
        Latitude coordinates of V grid cell faces defining section on NEMO model grid.

    Returns
    -------
    ds_Tv : xarray.Dataset
        Dataset including temperature (C) and salinity (psu | g/kg) on V grid cell
        faces defining section on NEMO model grid.
    """
    # Define empty list to store variables:
    var_list = []

    # Extract V temperature (C):
    if 'temp' in vars:
        if 'temp' in var_map:
            temp = ds[var_map['temp']]
        else:
            temp = ds['temp']

        temp_v = 0.5 * (temp.isel(x=x_index, y=y_index) + temp.isel(x=x_index, y=y_index+1))
        temp_v.name = 'temp'
        temp_v = temp_v.assign_attrs({'standard_name': 'sea_water_temperature'})
        var_list.append(temp_v)

    # Extract V salinity (psu | g/kg):
    if 'sal' in vars:
        if 'sal' in var_map:
            sal = ds[var_map['sal']]
        else:
            sal = ds['sal']

        sal_v = 0.5 * (sal.isel(x=x_index, y=y_index) + sal.isel(x=x_index, y=y_index+1))
        sal_v.name = 'sal'
        sal_v = sal_v.assign_attrs({'standard_name': 'sea_water_salinity'})
        var_list.append(sal_v)

    # Return variable or combine into single Dataset:
    if len(var_list) == 1:
        ds_Tv = var_list[0]
    else:
        ds_Tv = xr.merge(var_list)

    # Assign coordinates to the dataset:
    ds_Tv = (ds_Tv.
             assign_coords({'station': stations,
                            'longitude': longitudes,
                            'latitude': latitudes
                            })
            .rename({'deptht':'depth'})
            )

    return ds_Tv


# -- External Functions -- #
def extract_section(section_lon: np.ndarray,
                    section_lat: np.ndarray,
                    domain_path: str,
                    T_paths: list[str] | dict[str, list[str]],
                    U_paths: list[str] | dict[str, list[str]],
                    V_paths: list[str] | dict[str, list[str]],
                    var_map: dict = {},
                    uv_eiv: bool = False,
                    log: bool = False
                    ) -> xr.Dataset:
    """
    Extract velocity, volume transport & tracer variables along a continuous
    hydrographic section connecting a specified collection of geographic coordinates
    in the NEMO ocean model.

    Parameters
    ----------
    section_lon : numpy.ndarray
        Longitude coordinates defining the hydrographic section.
    section_lat : numpy.ndarray
        Latitude coordinates defining the hydrographic section.
    domain_path : str
        Path to the NEMO model domain configuration file. This file must contain
        the expected variables 'glamt' and 'gphit' model grid coordinates.
    T_paths : list[str] | dict[str, list[str]]
        Path to the NEMO model output T files. If variables are stored in separate
        output files, a dictionary of file paths must be provided with the variable
        names as keys and the file paths as values.
    U_paths : list[str] | dict[str, list[str]]
        Path to the NEMO model output U files. If variables are stored in separate
        output files, a dictionary of file paths must be provided with the variable
        names as keys and the file paths as values.
    V_paths : list[str] | dict[str, list[str]]
        Path to the NEMO model output V files. If variables are stored in separate
        output files, a dictionary of file paths must be provided with the variable
        names as keys and the file paths as values.
    var_map : dict, default={}
        Dictionary mapping expected variable names to their corresponding names in
        the given NEMO output files. Expected keys are 'uo', 'uo_eiv', 'vo', 'vo_eiv',
        'temp', 'sal', 'e1v', 'e2u', 'e3u', 'e3v'.
    uv_eiv : bool, default=False
        If True, eddy-induced zonal ('uo_eiv') and meridional velocities ('vo_eiv') are
        extracted to return the total velocity normal to the section (i.e., u = uo + uo_eiv).
    log : bool, default=False
        Whether to output logging information during the extraction process.

    Returns
    -------
    xarray.Dataset
        Dataset including velocity, volume transport, temperature & salinity variables along
        the continuous hydrographic section in the NEMO ocean model.

    Raises
    ------
    TypeError
    FileNotFoundError

    Example
    -------
    >>> from nemo_cookbook import extract_section

    >>> # Define section coordinates:
    >>> section_lon = np.array([-80, -45, -12])
    >>> section_lat = np.array([26.5, 26.5, 27])

    >>> # Define NEMO model domain configuration file path:
    >>> domain_path = '/path/to/nemo/domain_cfg.nc'
    >>> # Define NEMO model output file paths, where variables are stored in shared files:
    >>> T_paths = ['/path/to/nemo/output_gridT_1.nc', '/path/to/nemo/output_gridT_2.nc']
    >>> U_paths = ['/path/to/nemo/output_gridU_1.nc', '/path/to/nemo/output_gridU_2.nc']
    >>> V_paths = ['/path/to/nemo/output_gridV_1.nc', '/path/to/nemo/output_gridV_2.nc']

    >>> # Extract hydrographic section from NEMO model output with eddy-induced velocities:
    >>> ds_section = extract_section(section_lon=section_lon,
    ...                              section_lat=section_lat,
    ...                              domain_path=domain_path,
    ...                              T_paths=T_paths,
    ...                              U_paths=U_paths,
    ...                              V_paths=V_paths,
    ...                              var_map={},
    ...                              uv_eiv=True
    ...                              )
    <xarray.Dataset> Size: 715kB
    Dimensions:                (depth: 75, time_counter: 4, station: 66)
    Coordinates:
    * depth                  (depth) float32 300B 0.5058 1.556 ... 5.902e+03
    * time_counter           (time_counter) datetime64[ns] 32B 1976-07-02 ... 1...
    * station                (station) int64 528B 0 1 2 3 4 5 ... 61 62 63 64 65
        longitude              (station) float64 528B -56.8 -55.78 ... -9.802 -8.703
        latitude               (station) float64 528B 52.19 52.26 ... 56.76 56.69
    Data variables:
        velocity               (time_counter, depth, station) float32 79kB nan .....
        eddy_induced_velocity  (time_counter, depth, station) float32 79kB nan .....
        dz                     (time_counter, depth, station) float32 79kB nan .....
        dx                     (station) float64 528B 6.946e+04 ... 6.715e+04
        volume_transport       (time_counter, depth, station) float64 158kB nan ....
        temp                   (time_counter, depth, station) float32 79kB nan .....
        sal                    (time_counter, depth, station) float32 79kB nan .....
    """
    # -- Verify Inputs -- #
    if not isinstance(section_lon, np.ndarray):
        raise TypeError("section_lon must be an xarray DataArray.")
    if not isinstance(section_lat, np.ndarray):
        raise TypeError("section_lat must be an xarray DataArray.")
    if not isinstance(domain_path, str):
        raise TypeError("domain_path must be a string.")
    if not isinstance(T_paths, (list, dict)):
        raise TypeError("T_paths must be either a list or a dictionary.")
    if not isinstance(U_paths, (list, dict)):
        raise TypeError("U_paths must be either a list or a dictionary.")
    if not isinstance(V_paths, (list, dict)):
        raise TypeError("V_paths must be either a list or a dictionary.")
    if not isinstance(var_map, dict):
        raise TypeError("var_map must be a dictionary.")
    if not isinstance(uv_eiv, bool):
        raise TypeError("uv_eiv must be a boolean.")
    if not isinstance(log, bool):
        raise TypeError("log must be a boolean.")
    
    # -- Logging -- #
    if log:
        logging.basicConfig(
            stream=sys.stdout,
            format="nemo_cookbook | %(levelname)10s | %(asctime)s | %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # -- Load NEMO domain configuration -- #
    try:
        ds_domain_cfg = xr.open_dataset(domain_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"NEMO domain_cfg file not found at: {domain_path}.")

    try:
        glamt = ds_domain_cfg['glamt'].squeeze()
        gphit = ds_domain_cfg['gphit'].squeeze()
    except KeyError:
        raise KeyError("NEMO domain_cfg file does not contain 'glamt' or 'gphit' variables.")

    # -- Extract section coords from NEMO model grid -- #
    # Ensure section longitudes are increasing, otherwise reverse coords:
    if section_lon[-1] < section_lon[0]:
        section_lon = section_lon[::-1]
        section_lat = section_lat[::-1]

    # Find the nearest NEMO model grid cells to the section-defining coords:
    target_coords = _nearest_ji_coords(glamt=glamt,
                                       gphit=gphit,
                                       target_lon=section_lon,
                                       target_lat=section_lat
                                       )

    # Define full section in NEMO model grid:
    section_coords = _get_section_coords(glamt=glamt,
                                         gphit=gphit,
                                         target_coords=target_coords
                                         )

    # Add station dimension to section model grid coordinates:
    x_index = xr.DataArray([p[1] for p in section_coords], dims='station')
    y_index = xr.DataArray([p[0] for p in section_coords], dims='station')
    flux_dir = xr.DataArray(np.array([p[2] for p in section_coords]), dims='station')

    # Update model coordinates according to grid cell face:
    x_index = xr.where(cond=flux_dir == 'U+', x=x_index-1, y=x_index)
    y_index = xr.where(cond=flux_dir == 'U+', x=y_index+1, y=y_index)
    x_index = xr.where(cond=flux_dir == 'U-', x=x_index-1, y=x_index)

    # Extract T-point geographic coordinates along the section array:
    longitudes = glamt.isel(x=x_index, y=y_index)
    latitudes = gphit.isel(x=x_index, y=y_index)

    # Define zonal and meridional grid cell face masks:
    umask = (flux_dir == 'U-') | (flux_dir == 'U+')
    ummask = flux_dir == 'U-'
    upmask = flux_dir == 'U+'
    vmask = flux_dir == 'V+'

    # Extract station indices for each grid cell face:
    station_umask = x_index.station[umask]
    station_ummask = x_index.station[ummask]
    station_upmask = x_index.station[upmask]
    station_vmask = x_index.station[vmask]

    if log:
        logging.info("Completed Extracted section coordinates from NEMO model grid.")

    # -- Process NEMO domain -- #
    # U+ zonal grid cell width (m):
    if 'e2u' in var_map:
        e2u = ds_domain_cfg[var_map['e2u']]
    else:
        e2u = ds_domain_cfg['e2u']
    e2um = e2u.squeeze().isel(x=x_index[ummask], y=y_index[ummask])
    e2um = e2um.assign_coords({'station': station_ummask,
                               'longitude': longitudes[ummask],
                               'latitude': latitudes[ummask]}
                               )
    e2um.name = 'dx'

    # U- zonal grid cell width (m):
    e2up = e2u.squeeze().isel(x=x_index[upmask], y=y_index[upmask])
    e2up = e2up.assign_coords({'station': station_upmask,
                               'longitude': longitudes[upmask],
                               'latitude': latitudes[upmask]}
                               )
    e2up.name = 'dx'

    # V+ zonal grid cell width (m):
    if 'e1v' in var_map:
        e1v = ds_domain_cfg[var_map['e1v']]
    else:
        e1v = ds_domain_cfg['e1v']
    e1v = e1v.squeeze().isel(x=x_index[vmask], y=y_index[vmask])
    e1v = e1v.assign_coords({'station': station_vmask,
                             'longitude': longitudes[vmask],
                             'latitude': latitudes[vmask]}
                             )
    e1v.name = 'dx'

    if log:
        logging.info("Completed: Extracted variable dx from NEMO model grid.")

    # -- Process NEMO U,V outputs -- #
    # All U variables are stored in common output files:
    if isinstance(U_paths, list):
        if uv_eiv:
            vars = ['uo', 'uo_eiv', 'e3u']
        else:
            vars = ['uo', 'e3u']

        # Extract U- velocities & vertical grid cell thickness:
        _process_Um = partial(process_Um,
                              vars=vars,
                              var_map=var_map,
                              x_index=x_index[ummask],
                              y_index=y_index[ummask],
                              stations=station_ummask,
                              longitudes=longitudes[ummask],
                              latitudes=latitudes[ummask],
                              )
        try:
            ds_Um = xr.open_mfdataset(U_paths, preprocess=_process_Um).load()
            if log:
                logging.info(f"Completed: Extracted U- variables -> {vars}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths}.")

        # Extract U+ velocities & vertical grid cell thickness:
        _process_Up = partial(process_Up,
                              vars=vars,
                              var_map=var_map,
                              x_index=x_index[upmask],
                              y_index=y_index[upmask],
                              stations=station_upmask,
                              longitudes=longitudes[upmask],
                              latitudes=latitudes[upmask],
                              )
        try:
            ds_Up = xr.open_mfdataset(U_paths, preprocess=_process_Up).load()
            if log:
                logging.info(f"Completed: Extracted U+ variables -> {vars}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths}.")

    # All U variables are stored in separate output files:
    else:
        Um_list = []
        Up_list = []
        for var in U_paths.keys():
            # Extract U- velocities & vertical grid cell thickness:
            _process_Um = partial(process_Um,
                                  vars=[var],
                                  var_map=var_map,
                                  x_index=x_index[ummask],
                                  y_index=y_index[ummask],
                                  stations=station_ummask,
                                  longitudes=longitudes[ummask],
                                  latitudes=latitudes[ummask],
                                  )
            try:
                Um_list.append(xr.open_mfdataset(U_paths[var], preprocess=_process_Um).load())
                if log:
                    logging.info(f"Completed: Extracted U- variable -> {var}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths[var]}.")

            # Extract U+ velocities & vertical grid cell thickness:
            _process_Up = partial(process_Up,
                                  vars=[var],
                                  var_map=var_map,
                                  x_index=x_index[upmask],
                                  y_index=y_index[upmask],
                                  stations=station_upmask,
                                  longitudes=longitudes[upmask],
                                  latitudes=latitudes[upmask],
                                  )
            try:
                Up_list.append(xr.open_mfdataset(U_paths[var], preprocess=_process_Up).load())
                if log:
                    logging.info(f"Completed: Extracted U+ variable -> {var}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths[var]}.")

        # Merge U- and U+ variables into single datasets:
        ds_Um = xr.merge(Um_list)
        ds_Up = xr.merge(Up_list)

    # All V variables are stored in common output files:
    if isinstance(V_paths, list):
        if uv_eiv:
            vars = ['vo', 'vo_eiv', 'e3v']
        else:
            vars = ['vo', 'e3v']

        # Extract V+ velocities & vertical grid cell thickness:
        _process_Vp = partial(process_Vp,
                              vars=vars,
                              var_map=var_map,
                              x_index=x_index[vmask],
                              y_index=y_index[vmask],
                              stations=station_vmask,
                              longitudes=longitudes[vmask],
                              latitudes=latitudes[vmask],
                              )
        try:
            ds_V = xr.open_mfdataset(V_paths, preprocess=_process_Vp).load()
            if log:
                logging.info(f"Completed: Extracted V variables -> {vars}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO V-grid file not found at: {V_paths}.")

    # All V variables are stored in separate output files:
    else:
        V_list = []
        for var in V_paths.keys():
            _process_Vm = partial(process_Vp,
                                  vars=[var],
                                  var_map=var_map,
                                  x_index=x_index[vmask],
                                  y_index=y_index[vmask],
                                  stations=station_vmask,
                                  longitudes=longitudes[vmask],
                                  latitudes=latitudes[vmask],
                                  )
            try:
                V_list.append(xr.open_mfdataset(V_paths[var], preprocess=_process_Vm).load())
                if log:
                    logging.info(f"Completed: Extracted V variable -> {var}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"NEMO V-grid file not found at: {V_paths[var]}.")
        
        ds_V = xr.merge(V_list)

    # Merge zonal, meridional velocities & grid cell properties into a single Dataset:
    ds_uv = xr.merge([ds_Um, ds_Up, ds_V, e2um, e2up, e1v], combine_attrs='drop_conflicts')

    # Calculate total volume transport normal to the section:
    if uv_eiv:
        ds_uv['volume_transport'] = (ds_uv['velocity'] + ds_uv['eddy_induced_velocity']) * ds_uv['dx'] * ds_uv['dz']
    else:
        ds_uv['volume_transport'] = ds_uv['velocity'] * ds_uv['dx'] * ds_uv['dz']
    if log:
        logging.info("Completed: Calculated volume transport normal to section.")

    # -- Process NEMO T outputs -- #
    # All T variables are stored in common output files:
    if isinstance(T_paths, list):
        # Extract U-point temperature and salinity:
        _process_Tu = partial(process_Tu,
                              vars=['temp', 'sal'],
                              var_map=var_map,
                              x_index=x_index[umask],
                              y_index=y_index[umask],
                              stations=station_umask,
                              longitudes=longitudes[umask],
                              latitudes=latitudes[umask]
                              )
        try:
            ds_Tu = xr.open_mfdataset(T_paths, preprocess=_process_Tu).load()
            if log:
                logging.info("Completed: Extracted Tu variables -> temp, sal.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths}.")

        # Extract V-point temperature and salinity:
        _process_Tv = partial(process_Tv,
                              vars=['temp', 'sal'],
                              var_map=var_map,
                              x_index=x_index[vmask],
                              y_index=y_index[vmask],
                              stations=station_vmask,
                              longitudes=longitudes[vmask],
                              latitudes=latitudes[vmask]
                              )
        try:
            ds_Tv = xr.open_mfdataset(T_paths, preprocess=_process_Tv).load()
            if log:
                logging.info("Completed: Extracted Tv variables -> temp, sal.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths}.")

    # All T variables are stored in separate output files:
    else:
        # Extract U-point temperature and salinity:
        Tu_list = []
        for var in T_paths.keys():
            _process_Tu = partial(process_Tu,
                                  vars=[var],
                                  var_map=var_map,
                                  x_index=x_index[umask],
                                  y_index=y_index[umask],
                                  stations=station_umask,
                                  longitudes=longitudes[umask],
                                  latitudes=latitudes[umask]
                                  )
            try:
                Tu_list.append(xr.open_mfdataset(T_paths[var], preprocess=_process_Tu).load())
                if log:
                    logging.info(f"Completed: Extracted Tu variable -> {var}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths[var]}.")
            
        # Extract V-point temperature and salinity:
        Tv_list = []
        for var in T_paths.keys():
            _process_Tv = partial(process_Tv,
                                  vars=[var],
                                  var_map=var_map,
                                  x_index=x_index[vmask],
                                  y_index=y_index[vmask],
                                  stations=station_vmask,
                                  longitudes=longitudes[vmask],
                                  latitudes=latitudes[vmask]
                                  )
            try:
                Tv_list.append(xr.open_mfdataset(T_paths[var], preprocess=_process_Tv).load())
                if log:
                    logging.info(f"Completed: Extracted Tv variable -> {var}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths[var]}.")
        
        ds_Tu = xr.merge(Tu_list)
        ds_Tv = xr.merge(Tv_list)

    # Merge temperature and salinity into a single dataset:
    ds_ts = xr.merge([ds_Tu, ds_Tv], combine_attrs='drop_conflicts')

    # Merge velocity and tracers variables into a single Dataset:
    ds_section = xr.merge([ds_uv, ds_ts], combine_attrs='drop_conflicts')

    # Remove unpermitted coordinates:
    permitted_coords = ['time_counter', 'depth', 'station', 'longitude', 'latitude']
    ds_section = ds_section.drop_vars([coord for coord in ds_section.coords if coord not in permitted_coords])

    if log:
        logging.info("Completed: Combined variables along the section.")

    return ds_section

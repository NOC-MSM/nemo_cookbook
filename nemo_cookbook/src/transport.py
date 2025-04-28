"""
transports.py

Description: Functions to calculate the meridional volume, heat, salt and freshwater
transports as a function of latitude and sigma-z volume transport diagrams.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 22/10/2024
"""

# -- Import dependencies -- #
import xarray as xr

# -- Exernal Functions -- #
def compute_mht(vo: xr.DataArray,
                thetao: xr.DataArray,
                e1v: xr.DataArray,
                e3v: xr.DataArray,
                mask: xr.DataArray | None = None
                ) -> xr.DataArray:
    """
    Compute the meridional heat transport in latitude coordinates using 
    output stored on the NEMO ORCA grid. The vertical grid cell thickness
    on V-points is assumed to be time-dependent (i.e., e3v(t,z,y,x)).

    Parameters
    ----------
    vo: xarray.DataArray
        Meridional velocity component stored at V-points.
    thetao: xarray.DataArray
        Potential temperature stored at T-points.
    e1v: xarray.DataArray
        Grid cell width in the zonal direction at V-points.
    e3v: xarray.DataArray
        Grid cell thickness in the vertical direction at V-points.
    mask: xarray.DataArray, default=None
        Ocean basin mask where 1 = included and 0 = excluded values.

    Returns
    -------
    xarray.DataArray
        Meridional heat transport [1 PW == 10^15 Watts] as a function of latitude.

    Raises
    ------
    TypeError
    ValueError

    Examples
    --------
    >>> from nemo_cookbook.recipes import compute_mht

    >>> # Calculate meridional heat transport as a function of latitude:
    >>> compute_mht(vo=ds_gridV['vo'],
                    thetao=ds_gridT['thetao'],
                    e1v=ds_gridV['e1v'],
                    e3v=ds_gridV['e3v'],
                    mask=ds_subbasins['atlmsk'],
                    )

    """
    # -- Verify Inputs --#
    # Types:
    if not isinstance(vo, xr.DataArray):
        raise TypeError("vo must be an xarray.DataArray.")
    if not isinstance(thetao, xr.DataArray):
        raise TypeError("thetao must be an xarray.DataArray.")
    if not isinstance(e1v, xr.DataArray):
        raise TypeError("e1v must be an xarray.DataArray.")
    if not isinstance(e3v, xr.DataArray):
        raise TypeError("e3v must be an xarray.DataArray.")
    if mask is not None:
        if not isinstance(mask, xr.DataArray):
            raise TypeError("mask must be an xarray.DataArray.")
    # Dimension names:
    if vo.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("vo must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if thetao.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("thetao must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if e3v.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("e3v must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if e1v.dims != ('y', 'x'):
        raise ValueError("e1v must have dimensions ('y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")

    # -- Define parameters -- #
    # Conversion from heat flux in [Jm^2/s to PW]:
    Jm2s_to_PW = 1E-15
    # Reference density for seawater [kg/m^3]:
    rho0 = 1026.0
    # Specific heat capacity of seawater [J/kg/K]:
    cp0 = 3991.86795711963

    # -- Apply Atlantic Ocean mask to variables --
    if mask is not None:
        # Time-evolving meridional volume transport:
        vo = vo.where(mask == 1)
        # Time-evolving potential temperature:
        thetao = thetao.where(mask == 1)
        # Time-dependent grid cell thickness, dz(t):
        dzt = e3v.where(mask == 1)
        # Time-independent grid cell width, dx:
        dx = e1v.where(mask == 1)

    # -- Interpolate potential temperature onto V-points -- #
    # Average adjacent T-points to interpolate, excluding last T-point:
    thetao_interpv = 0.5 * (thetao.isel(y=slice(None,-1)) + thetao.isel(y=slice(1,None)))
    # Rename vertical coordinate to match V-points:
    thetao_interpv = thetao_interpv.rename({'deptht':'depthv'})

    # -- Calculate MHT -- #
    # Compute product of meridional volume transport and potential temperature:
    # NOTE: Exclude last y-point to match interpolated V-point dimensions.
    # See CDFTOOLS: https://github.com/meom-group/CDFTOOLS/blob/master/src/cdfmhst.f90
    subvol_thetao = (vo * dx * dzt).isel(y=slice(None,-1)) * thetao_interpv
    # Compute meridional heat transport [PW]:
    mht = (Jm2s_to_PW * rho0 * cp0 *subvol_thetao).sum(dim=['x', 'depthv'])

    # Update DataArray attributes:
    mht.name = 'mht'
    mht.attrs['units'] = 'PW'
    mht.attrs['long_name'] = 'meridional heat transport'
    mht.attrs['standard_name'] = 'mht'

    return mht


def compute_mst(vo: xr.DataArray,
                so: xr.DataArray,
                e1v: xr.DataArray,
                e3v: xr.DataArray,
                mask: xr.DataArray | None = None
                ) -> xr.DataArray:
    """
    Compute the meridional salt transport in latitude coordinates using 
    output stored on the NEMO ORCA grid. The vertical grid cell thickness
    on V-points is assumed to be time-dependent (i.e., e3v(t,z,y,x)).

    Parameters
    ----------
    vo: xarray.DataArray
        Meridional velocity component stored at V-points.
    so: xarray.DataArray
        Practical salinity stored at T-points.
    e1v: xarray.DataArray
        Grid cell width in the zonal direction at V-points.
    e3v: xarray.DataArray
        Grid cell thickness in the vertical direction at V-points.
    mask: xarray.DataArray, default=None
        Ocean basin mask where 1 = included and 0 = excluded values.

    Returns
    -------
    xarray.DataArray
        Meridional salt transport [Sv.PSU] as a function of latitude.

    Raises
    ------
    TypeError
    ValueError

    Examples
    --------
    >>> from nemo_cookbook.recipes import compute_mst

    >>> # Calculate meridional salt transport as a function of latitude:
    >>> compute_mst(vo=ds_gridV['vo'],
                    so=ds_gridT['so'],
                    e1v=ds_gridV['e1v'],
                    e3v=ds_gridV['e3v'],
                    mask=ds_subbasins['atlmsk'],
                    )

    """
    # -- Verify Inputs --#
    # Types:
    if not isinstance(vo, xr.DataArray):
        raise TypeError("vo must be an xarray.DataArray.")
    if not isinstance(so, xr.DataArray):
        raise TypeError("so must be an xarray.DataArray.")
    if not isinstance(e1v, xr.DataArray):
        raise TypeError("e1v must be an xarray.DataArray.")
    if not isinstance(e3v, xr.DataArray):
        raise TypeError("e3v must be an xarray.DataArray.")
    if mask is not None:
        if not isinstance(mask, xr.DataArray):
            raise TypeError("mask must be an xarray.DataArray.")
    # Dimension names:
    if vo.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("vo must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if so.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("so must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if e3v.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("e3v must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if e1v.dims != ('y', 'x'):
        raise ValueError("e1v must have dimensions ('y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")

    # -- Define parameters -- #
    # Conversion for salt transport in [Sv.PSU]:
    m3s_to_Sv = 1E-6

    # -- Apply Atlantic Ocean mask to variables -- #
    if mask is not None:
        # Time-evolving meridional volume transport:
        vo = vo.where(mask == 1)
        # Time-evolving practical salinity:
        so = so.where(mask == 1)
        # Time-dependent grid cell thickness, dz(t):
        dzt = e3v.where(mask == 1)
        # Time-independent grid cell width, dx:
        dx = e1v.where(mask == 1)

    # -- Interpolate practical salinity onto V-points -- #
    # Average adjacent T-points to interpolate, excluding last T-point:
    so_interpv = 0.5 * (so.isel(y=slice(None,-1)) + so.isel(y=slice(1,None)))
    # Rename vertical coordinate to match V-points:
    so_interpv = so_interpv.rename({'deptht':'depthv'})

    # -- Calculate MST --
    # Compute product of meridional volume transport and practical salinity:
    # NOTE: Exclude last y-point to match interpolated V-point dimensions.
    # See CDFTOOLS: https://github.com/meom-group/CDFTOOLS/blob/master/src/cdfmhst.f90
    subvol_so = (vo * dx * dzt).isel(y=slice(None,-1)) * so_interpv
    # Compute meridional salt transport [Sv.PSU]:
    mst = (m3s_to_Sv * subvol_so).sum(dim=['x', 'depthv'])

    # Update DataArray attributes:
    mst.name = 'mst'
    mst.attrs['units'] = 'Sv.PSU'
    mst.attrs['long_name'] = 'meridional salt transport'
    mst.attrs['standard_name'] = 'mst'

    return mst

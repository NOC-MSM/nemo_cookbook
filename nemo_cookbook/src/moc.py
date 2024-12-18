"""
moc.py

Description: Functions to calculate the meridional overturning stream function
as a function of latitude and geopotential- or tracer-coordinates [Sv]

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 22/10/2024
"""

# -- Import required packages -- #
import numpy as np
import xarray as xr
from flox.xarray import xarray_reduce

# -- Define function to compute depth-space AMOC as a function of latitude -- #
def compute_moc_z(vo:xr.DataArray, e1v:xr.DataArray, e3v:xr.DataArray, mask:xr.DataArray | None = None) -> xr.DataArray:
    """
    Compute the Meridional Overturning Stream Function in depth-coordinates
    using output stored on the NEMO ORCA grid. The vertical grid cell thickness
    on V-points is assumed to be time-dependent (i.e., e3v(t,z,y,x)).

    Parameters
    ----------
    vo: xarray.DataArray
        Meridional velocity component stored at V-points.
    e1v: xarray.DataArray
        Grid cell width in the zonal direction at V-points.
    e3v: xarray.DataArray
        Grid cell thickness in the vertical direction at V-points.
    mask: xarray.DataArray
        Ocean basin mask where 1 = included and 0 = excluded values.
        Default value is None.

    Returns
    -------
    DataArray
        Meridional overturning stream function in latitude-depth coordinates.

    Raises
    ------
    TypeError
    ValueError

    Examples
    --------
    >>> from nemo_cookbook.recipes import compute_moc_z

    >>> # Calculate meridional overturning stream function in depth coordinates:
    >>> from nemo_cookbook.recipes import compute_moc_z
    >>> compute_moc_z(vo=ds_gridV['vo'],
                      e1v=ds_gridV['e1v'],
                      e3v=ds_gridV['e3v'],
                      mask=ds_subbasins['atlmsk'],
                      )

    <xarray.DataArray 'moc_z' (time_counter: 120, depthv: 75, y: 1206)> Size: 87MB
    array([[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            3.11791952e-05, -3.21932477e-03,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            1.77901547e-03, -5.02720583e-03,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            5.40307820e-03, -5.27595005e-03,  0.00000000e+00],
            ...,
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            -1.27486725e+00, -1.27782904e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            -1.27486725e+00, -1.27782904e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            -1.27486725e+00, -1.27782904e+00,  0.00000000e+00]]])
    Coordinates:
    * depthv         (depthv) float32 300B 0.5058 1.556 ... 5.698e+03 5.902e+03
    * time_counter   (time_counter) datetime64[ns] 960B 1976-01-16T12:00:00 ......
    Dimensions without coordinates: y

    """
    # -- Verify input arguments --#
    # Dimension names:
    if vo.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("vo must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if e3v.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("e3v must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if e1v.dims != ('y', 'x'):
        raise ValueError("e1v must have dimensions ('y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")
    # Number of dimensions:
    if vo.ndim != 4:
        raise ValueError("vo must be a 4D array.")
    if e3v.ndim != 4:
        raise ValueError("e3v must be a 4D array.")
    if e1v.ndim != 2:
        raise ValueError("e1v must be a 2D array.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    # -- Define parameters --
    # Conversion from volume flux in m^3/s to Sv:
    m3s_to_Sv = 1E-6

    # -- Apply Atlantic Ocean mask to variables --
    if mask is not None:
        # Time-evolving meridional volume transport:
        vo = vo.where(mask == 1)
        # Time-dependent grid cell thickness, dz(t):
        dzt = e3v.where(mask == 1)
        # Time-independent grid cell width, dx:
        dx = e1v.where(mask == 1)

    # -- Calculate AMOC in depth-coordinates --
    # Compute meridional volume transport (Sv):
    subvol = m3s_to_Sv * (vo * dzt * dx)
    # Compute vertical overturning stream function (Sv):
    moc_z = subvol.sum(dim='x').cumsum(dim='depthv')

    # Update DataArray attributes:
    moc_z.name = f'moc_z'
    moc_z.attrs['units'] = 'Sv'
    moc_z.attrs['long_name'] = 'meridional overturning stream function in depth coordinates'
    moc_z.attrs['standard_name'] = 'moc_z'

    return moc_z

# -- Define function to compute tracer-space AMOC as a function of latitude -- #
def compute_moc_tracer(vo:xr.DataArray, e1v:xr.DataArray, e3v:xr.DataArray, tracer:xr.DataArray, tracer_bins:np.ndarray, dir:str = '+1', mask:xr.DataArray | None = None) -> xr.DataArray:
    """
    Compute the Meridional Overturning Stream Function in latitude-tracer coordinates
    using output stored on the NEMO ORCA grid. The vertical grid cell thickness
    on V-points is assumed to be time-dependent (i.e., e3v(t,z,y,x)).

    Note in the current implementation adjacent T-point tracer values are used rather
    than interpolating onto V-points as in CDFTOOLS cdfmocsig.f90.

    Parameters
    ----------
    v0: xarray.DataArray
        Meridional velocity component stored at V-points.
    e1v: xarray.DataArray
        Grid cell width in the zonal direction at V-points.
    e3v: xarray.DataArray
        Grid cell thickness in the vertical direction at V-points.
    tracer: xarray.DataArray
        Tracer values used to bin the meridional volume transport.
    tracer_bins: ndarray
        Monotonically increasing array of bin edges closed on the rightmost edge
        (i.e., bin_{n} <= x < bin_{n+1}).
    dir: str
        Direction of cumulative integration along the discrete tracer coordinate axis.
        Default value is '+1' resulting in accumulation from the smallest to largest
        tracer value.
    mask: xarray.DataArray
        Ocean basin mask where 1 = included and 0 = excluded values.
        Default value is None.
    
    Returns
    -------
    DataArray
        Meridional overturning stream function in latitude-tracer coordinates.

    Raises
    ------
    TypeError
    ValueError

    Examples
    --------
    >>> import gsw
    >>> from nemo_cookbook.recipes import compute_moc_tracer

    >>> # Calculate potential density anomaly from conservative temp. and abs. salinity:
    >>> sigma0 = gsw.density.sigma0(CT=ds_gridT['thetao_con'], SA=ds_gridT['so_abs'])
    >>> # Re-define DataArray name - potential density anomaly referenced to sea surface:
    >>> sigma0.name = 'sigma0'

    >>> # Calculate meridional overturning stream function in potential density coordinates:
    >>> compute_moc_tracer(vo=ds_gridV['vo'],
                           e1v=ds_gridV['e1v'],
                           e3v=ds_gridV['e3v'],
                           tracer=sigma0,
                           tracer_bins=np.arange(21, 28.2, 0.01),
                           dir = '+1',
                           mask=ds_subbasins['atlmsk'],
                           )
    <xarray.DataArray 'moc' (time_counter: 120, y: 1206, sigma0_bins: 719)> Size: 832MB
    array([[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            ...,
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            -1.27486725e+00, -1.27486725e+00, -1.27486725e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            -1.27699252e+00, -1.27699252e+00, -1.27699252e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])
    Coordinates:
    * time_counter  (time_counter) datetime64[ns] 960B 1976-01-16T12:00:00 ... ...
    * y             (y) int64 10kB 0 1 2 3 4 5 6 ... 1200 1201 1202 1203 1204 1205
    * sigma0_bins   (sigma0_bins) object 6kB (21.0, 21.01] ... (28.180000000001...
    """

    # -- Verify input arguments -- #
    # Dimension names:
    if vo.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("vo must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if e3v.dims != ('time_counter', 'depthv', 'y', 'x'):
        raise ValueError("e3v must have dimensions ('time_counter', 'depthv', 'y', 'x').")
    if tracer.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("tracer must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if e1v.dims != ('y', 'x'):
        raise ValueError("e1v must have dimensions ('y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")
    # Number of dimensions:
    if vo.ndim != 4:
        raise ValueError("vo must be a 4D array.")
    if e3v.ndim != 4:
        raise ValueError("e3v must be a 4D array.")
    if tracer.ndim != 4:
        raise ValueError("tracer must be a 4D array.")
    if e1v.ndim != 2:
        raise ValueError("e1v must be a 2D array.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    # Type of tracer bins:
    if not isinstance(tracer_bins, np.ndarray):
        raise TypeError('tracer_bins must be specified as an ndarray')
    # Type and value of cumulative integration direction:
    if not isinstance(dir, str):
        raise TypeError('dir must be specified as a str')
    if dir not in ['+1', '-1']:
        raise ValueError('direction of cumulative integration must be given as either \'+1\' or \'-1\'')

    # -- Define parameters & tracer properties -- #
    # Conversion from volume flux in m^3/s to Sv:
    m3s_to_Sv = 1E-6
    # Extract tracer variable name:
    tracer_name = tracer.name
    # Update tracer depth-dimension label:
    tracer = tracer.rename({'deptht':'depthv'})

    # -- Apply ocean mask to variables -- #
    if mask is not None:
        # Time-evolving meridional volume transport:
        vo = vo.where(mask == 1)
        # Time-dependent tracer field:
        tracer = tracer.where(mask == 1)
        # Time-dependent grid cell thickness, dz(t):
        dzt = e3v.where(mask == 1)
        # Time-independent grid cell width, dx:
        dx = e1v.where(mask == 1)

    # -- Calculate meridional volume transport in tracer-coordinates -- #
    # Compute meridional volume transport (Sv):
    subvol = m3s_to_Sv * (vo * dzt * dx)
    subvol.name = 'subvol'

    # Define combined volume transport and tracer DataSet to input to xarray_reduce:
    subvol = subvol.to_dataset(name=subvol.name)
    subvol[tracer_name] = tracer

    # Compute binned sum of meridional volume transport (Sv) in discrete tracer bins.
    subvol_tracer = xarray_reduce(
            subvol['subvol'], # Meridional volume transport DataArray to bin.
            subvol['time_counter'], # Coordinate DataArrays to retain - 1. time_counter
            subvol[tracer_name], # Tracer variable used to bin meridional volume transport.
            subvol['y'], # Coordinate DataArrays to retain - 2. y coordinate index.
            func="nansum", # Summary operation within bins - no need to include x coordinate since sum is over x-tracer.
            expected_groups=(None, tracer_bins, None), # Bins specified for each group - None for existing labels.
            isbin=(False, True, False),
            method="map-reduce",
            reindex=False, # Do not reindex during block aggregations to reduce memory at cost of performance.
            engine='numbagg' # Use numbagg grouped aggregations.
        )

    # -- Compute meridional overturning stream function in tracer-coordinates -- #
    # Accumulating from smallest to largest tracer bin:
    if dir == '+1':
        moc_tracer = subvol_tracer.cumsum(dim=f'{tracer_name}_bins')
    # Accumulating from largest to smallest tracer bin:
    else:
        moc_tracer = subvol_tracer.reindex(tracer_bins=subvol_tracer[f'{tracer_name}_bins'][::-1]).cumsum(dim='tracer_bins')
        moc_tracer.rename({'tracer_bins':f'{tracer_name}_bins'})

    # Update DataArray attributes:
    moc_tracer.name = f'moc_{tracer_name}'
    moc_tracer.attrs['units'] = 'Sv'
    moc_tracer.attrs['long_name'] = f'meridional overturning stream function in {tracer_name} coordinates'
    moc_tracer.attrs['standard_name'] = f'moc_{tracer_name}'
    # Update coordinate labels:
    moc_tracer.sigma0_bins.attrs['long_name'] = f'{tracer_name} bins'
    moc_tracer.sigma0_bins.attrs['standard_name'] = tracer_name

    return moc_tracer

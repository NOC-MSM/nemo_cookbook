"""
wmt.py

Description: Functions to calculate the water mass transformation diagnostics
as a function of tracer-coordinates [Sv].

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 30/10/2024
"""

# -- Import required packages -- #
import gsw
import numpy as np
import xarray as xr
from flox.xarray import xarray_reduce

# -- Define function to compute surface-forced overturning component in density-coordinates -- #
def compute_sfoc_sigma0(sst:xr.DataArray, sss:xr.DataArray, qhf:xr.DataArray, qfw:xr.DataArray, e1t:xr.DataArray, e2t:xr.DataArray, sigma0_bins:np.ndarray, mask:xr.DataArray | None = None) -> xr.DataArray:
    """
    Compute Surface-Forced Overturning Component (SFOC) in
    density-coordinates from sea surface properties and surface
    buoyancy flux into the ocean.

    SFOC represents the accumulated volume flux across a given
    outcropping density surface (isopycnal) owing to surface buoyancy
    forcing alone.

    Parameters
    ----------
    sst : xr.DataArray
        Sea surface temperature (C) with dimensions (j x i).
    sss : xr.DataArray
        Sea surface salinity (g kg-1) with dimensions (j x i).
    qhf : xr.DataArray
        Net surface heat flux (W m-2) directed downwards into
        the ocean with dimensions (j x i).
    qfw : xr.DataArray
        Net surface freshwater flux (kg m-2) directed upwards
        out of the ocean with dimensions (j x i).
    e1t : xr.DataArray
        Zonal width of model grid cell at T-points (m).
    e2t : xr.DataArray
        Meridional width of model grid cell at T-points (m).
    sigma0_bins: ndarray
        Monotonically increasing array of sea surface density bin edges
        closed on the rightmost edge (i.e., bin_{n} <= x < bin_{n+1}).
    mask: xarray.DataArray
        Ocean basin mask where 1 = included and 0 = excluded values.
        Default value is None.

    Returns
    -------
    xr.DataArray
        Surface-forced overturning component (Sv) in
        potential density coordinates.

    Raises
    ------
    TypeError
    ValueError

    Example
    -------
    >>> # Calculate surface-forced overturning component in density coordinates:
    >>> from nemo_cookbook.recipes import compute_sfoc_sigma0
    >>> compute_sfoc_sigma0(sst=ds_sst.sst,
                            sss=ds_sss.sss,
                            qhf=ds_shf.heatflux,
                            qfw=-ds_swf.waterflux,
                            e1t=ds_grid.e1t,
                            e2t=ds_grid.e2t,
                            sigma0_bins=np.arange(20, 28, 0.01)
                            mask=ds_mask.atlmask
                            )

    <xarray.DataArray 'sfoc_sigma0' (time_counter: 120, sigma0_bins: 599)> Size: 5MB
    array([[-1.96161061e-01,  4.76969465e-01,  9.19266571e-01,
            1.19149290e+00,  7.98253722e-01,  1.09950030e+00,
            1.41423059e+00,  2.51569701e+00, -7.33021441e-02,
            2.69024176e+00,  1.75924921e+00,  2.31847949e+00,
            4.47225205e-01,  1.07407598e+00,  1.91941170e+00,
            1.55073911e+00,  1.04647197e+00,  2.16705865e+00,
            5.73937088e-01,  2.03200180e+00, -3.88888598e-01,
            8.60289416e-01,  9.86043671e-01,  4.59567545e-01,
            4.33853362e-01,  9.50074676e-01,  2.88884070e-04,
            2.63865890e-01,  9.35489566e-01, -3.02711124e-01,
            -1.58833632e+00,  2.50333021e+00,  8.34612170e-01,
            8.33165344e-02, -1.07395164e+00, -1.29790172e+00,
            -1.04865377e+00, -1.26126730e+00, -8.66729430e-01,
            6.37792117e-01, -5.93714393e-01, -4.44344778e-01,
            4.43514677e-01, -1.76845444e-02, -3.66341066e-01,
            9.18922421e-01, -1.08504129e+00,  3.87270948e-01,
            4.96308965e-01,  3.26862112e+00,  1.16670095e+00,
            2.99221858e+00,  1.15222430e+00,  2.47429514e+00,
            8.49286632e-01,  3.30913930e+00,  1.42410406e+00,
            9.04368036e-01,  1.29169471e+00,  1.59489560e+00,
    ...
            1.90115562e+01,  1.73090398e+01,  1.78236435e+01,
            1.72344492e+01,  1.62369284e+01,  1.49556745e+01,
            1.59870484e+01,  1.17489057e+01,  8.91749027e+00,
            9.23528699e+00,  1.04526995e+01,  1.10700443e+01,
            1.18813182e+01,  1.05129082e+01,  1.12257544e+01,
            1.14991275e+01,  1.31751295e+01,  1.21212983e+01,
            1.34041625e+01,  1.35781846e+01,  1.41035749e+01,
            1.53561977e+01,  1.55170851e+01,  1.16976984e+01,
            9.50422030e+00,  8.54824223e+00,  8.55486055e+00,
            7.72613950e+00,  8.09130497e+00,  8.12188994e+00,
            7.89566337e+00,  7.08595404e+00,  6.90844599e+00,
            4.86636871e+00,  4.50318486e+00,  3.17532626e+00,
            2.46093827e+00,  1.99055373e+00,  1.83348636e+00,
            1.52457452e+00,  1.32904106e+00,  1.47134385e+00,
            1.08356976e+00,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan]])
    Coordinates:
    * time_counter  (time_counter) datetime64[ns] 1kB 2023-07-02T12:00:00....
    * sigma0_bins   (sigma0_bins) object 5kB (22.0, 22.01] ... (27.980000000000...
    """
    # -- Verify input arguments -- #
    # Dimension names:
    if sst.dims != ('time_counter', 'y', 'x'):
        raise ValueError("sst must have dimensions ('time_counter', 'y', 'x').")
    if sss.dims != ('time_counter', 'y', 'x'):
        raise ValueError("sss must have dimensions ('time_counter', 'y', 'x').")
    if e1t.dims != ('y', 'x'):
        raise ValueError("e1t must have dimensions ('y', 'x').")
    if e2t.dims != ('y', 'x'):
        raise ValueError("e2t must have dimensions ('y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")
    # Number of dimensions:
    if sst.ndim != 3:
        raise ValueError("sst must be a 3D array.")
    if sss.ndim != 3:
        raise ValueError("sss must be a 3D array.")
    if e1t.ndim != 2:
        raise ValueError("tracer must be a 2D array.")
    if e2t.ndim != 2:
        raise ValueError("e1v must be a 2D array.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    # Type of tracer bins:
    if not isinstance(sigma0_bins, np.ndarray):
        raise TypeError('sigma0_bins must be specified as an ndarray')

    # -- Defining Physical Parameters -- #
    # Specific heat capacity of sea water J kg m-3 from GSW.
    cp = 3991.86795711963
    # Conversion factor m3/s to Sv.
    m3s_to_Sv = 1E-6

    #  -- Computing Surface Area Model Grid Cells -- #
    # Compute area of model grid cells at T-Grid points.
    dxdy = e1t.squeeze()*e2t.squeeze()

    # -- Apply mask to variables --
    if mask is not None:
        # Sea surface temperature:
        sst = sst.where(mask == 1)
        # Sea surface salinity:
        sss = sss.where(mask == 1)
        # Net surface heat flux:
        qhf = qhf.where(mask == 1)
        # Net surface freshwater flux:
        qfw = qfw.where(mask == 1)

    # -- Compute Surface Properies -- #
    # Thermal expansion coefficient at sea surface - sea pressure = 0 dBar.
    alpha = gsw.density.alpha(SA=sss, CT=sst, p=0)
    # Haline Contraction Coefficient at sea surface - sea pressure = 0 dBar.
    beta = gsw.density.beta(SA=sss, CT=sst, p=0)

    # Compute sea surface density anomaly using TEOS-10:
    ssd = gsw.sigma0(CT=sst, SA=sss)

    # -- Compute SFWMT -- #
    # Computing surface density flux, f.
    f = -(alpha/cp)*qhf + beta*(sss/(1 - sss))*qfw
    # Compute product of surface density flux x grid cell area.
    f_area = f*dxdy
    # Compute SFWMT across each surface grid cell (Sv).
    dsigma0 = sigma0_bins[1] - sigma0_bins[0]
    sfwmt = (1/dsigma0)*f_area*m3s_to_Sv

    # -- Compute Surface Forced Overturning Component (SFOC) -- #
    # Binned summation of SFWMT in density-coordinates:
    sfoc_sigma0 = xarray_reduce(
            sfwmt,               # SFWMT to bin.
            sfwmt.time_counter,  # Coordinate to retain - time_counter.
            ssd,                 # Sea surface density to bin SFWMT.
            func="nansum",
            expected_groups=(None, sigma0_bins),
            isbin=[False, True],
            method="map-reduce",
            reindex=False,       # Do not reindex during block aggregations.
            engine='numbagg',    # Use numbagg grouped aggregations.
        )

    # Update DataArray attributes:
    sfoc_sigma0.name =  'sfoc_sigma0'
    sfoc_sigma0.attrs['units'] = 'Sv'
    sfoc_sigma0.attrs['long_name'] = 'surface-forced overturning component'
    sfoc_sigma0.attrs['standard_name'] = 'sfoc_sigma0'

    return sfoc_sigma0

# -- Define function to compute area of sea surface density outcrops -- #
def compute_ssd_area(sst:xr.DataArray, sss:xr.DataArray, e1t:xr.DataArray, e2t:xr.DataArray, sigma0_bins:np.ndarray, mask:xr.DataArray | None = None) -> xr.DataArray:
    """
    Compute area of sea surface density outcrops in potential
    density-coordinates from sea surface properties.

    Parameters
    ----------
    sst : xr.DataArray
        Sea surface temperature (C) stored at T-points.
    sss : xr.DataArray
        Sea surface salinity (g kg-1) stored at T-points.
    e1t : xr.DataArray
        Zonal width of model grid cell (m) on T-points.
    e2t : xr.DataArray
        Meridional width of model grid cell (m) at T-points.
    sigma0_bins: ndarray
        Monotonically increasing array of sea surface density bin edges
        closed on the rightmost edge (i.e., bin_{n} <= x < bin_{n+1}).
    mask: xarray.DataArray
        Ocean basin mask where 1 = included and 0 = excluded values.
        Default value is None.

    Returns
    -------
    xr.DataArray
        Area of sea surface density outcrops (m2) in potential density
        coordinates.

    Raises
    ------
    TypeError
    ValueError

    Example
    -------
    >>> # Calculate sea surface density outcrop area in the Atlantic Ocean:
    >>> from nemo_cookbook.recipes import compute_ssd_area
    >>> compute_ssd_area(sst=ds.sst,
                         sss=ds.sss,
                         e1t=ds_grid.e1t,
                         e2t=ds_grid.e2t,
                         sigma0_bins=np.arange(20, 28, 0.01)
                         mask=ds_mask.atlmask
                        )

    <xarray.DataArray 'ssd_area' (time_counter: 120, sigma0_bins: 799)> Size: 8MB
    array([[3.57555452e+09, 2.21622448e+09, 3.34888771e+09, 1.38283901e+09,
            4.86835886e+09, 8.46157343e+08, 3.70659207e+09, 8.36524334e+08,
            2.66636468e+09, 2.60323641e+09, 4.57815810e+09, 5.17835383e+09,
            8.01829870e+09, 3.45057600e+09, 1.80175423e+09, 2.17075050e+09,
            3.40465577e+09, 3.31977868e+09, 3.54069076e+09, 2.47288824e+09,
            5.36004988e+09, 5.19691005e+09, 2.40229209e+09, 5.19394409e+09,
            3.13370330e+09, 2.14988707e+09, 2.88201200e+09, 3.50093107e+09,
            3.00996140e+09, 2.40870638e+09, 2.72366766e+09, 8.80733855e+08,
            2.97325012e+09, 1.39653705e+09, 1.61162528e+09, 4.83819189e+09,
            3.32059103e+09, 3.71463245e+09, 2.20308363e+09, 4.05380839e+09,
            3.81368256e+09, 5.49909505e+09, 4.13823777e+09, 3.41343622e+09,
            9.57634453e+08, 3.43560242e+09, 3.03655539e+09, 2.67471345e+09,
            3.69084687e+09, 1.77584517e+09, 2.83483855e+09, 3.17036997e+09,
            2.49394045e+09, 4.14793380e+09, 3.85394838e+09, 2.75639621e+09,
            5.80738041e+08, 5.77273125e+09, 2.71313308e+09, 2.62344045e+09,
            4.65330523e+09, 4.29521468e+09, 1.27773114e+09, 2.88041942e+09,
            3.89207762e+09, 3.12015170e+09, 2.39227456e+09, 3.85064909e+09,
            4.20151471e+09, 4.08740655e+09, 4.86366679e+09, 5.20529918e+09,
            4.13039526e+09, 5.86346623e+09, 2.58863887e+09, 3.56281415e+09,
            2.75793184e+09, 4.79402231e+09, 3.75691875e+09, 5.13230712e+09,
    ...
            9.39584102e+10, 9.15906262e+10, 7.28943001e+10, 7.27963511e+10,
            7.65013586e+10, 7.36437503e+10, 7.69967134e+10, 7.54173897e+10,
            7.68768806e+10, 7.03834496e+10, 6.47753448e+10, 6.90201775e+10,
            5.74852691e+10, 6.74179012e+10, 6.31992263e+10, 6.47814510e+10,
            6.68150722e+10, 7.85645636e+10, 7.89267388e+10, 9.53029146e+10,
            1.01120323e+11, 8.99655798e+10, 8.88752848e+10, 8.78996470e+10,
            8.38625045e+10, 7.85996323e+10, 8.00118596e+10, 5.76872938e+10,
            3.86449723e+10, 3.62496637e+10, 4.12773171e+10, 4.15997266e+10,
            4.55421350e+10, 4.04002106e+10, 4.19861690e+10, 4.32212460e+10,
            4.97004436e+10, 4.55771755e+10, 5.11980633e+10, 5.31093112e+10,
            5.37400142e+10, 6.10194245e+10, 6.81813077e+10, 4.67175669e+10,
            3.99396504e+10, 3.79386554e+10, 3.98683622e+10, 3.87923576e+10,
            4.21181432e+10, 4.29533673e+10, 4.16647328e+10, 3.52454113e+10,
            3.38349246e+10, 2.44630974e+10, 2.44118485e+10, 1.73817172e+10,
            1.38657402e+10, 1.16823728e+10, 1.08859919e+10, 9.99545437e+09,
            7.87763597e+09, 9.46026688e+09, 7.08956542e+09,            nan,
                    nan,            nan,            nan,            nan,
                    nan,            nan,            nan,            nan,
                    nan,            nan,            nan,            nan,
                    nan,            nan,            nan]])
    Coordinates:
    * time_counter  (time_counter) datetime64[ns] 8B 2023-07-02T12:00:00...
    * sigma0_bins   (sigma0_bins) object 6kB (20.0, 20.01] ... (27.980000000001...
    """
    #  -- Computing Surface Area Model Grid Cells -- #
    # Compute area of model grid cells located about T-Grid points.
    dxdy = e1t.squeeze()*e2t.squeeze()

    # -- Apply mask to variables --
    if mask is not None:
        # Sea surface temperature:
        sst = sst.where(mask == 1)
        # Sea surface salinity:
        sss = sss.where(mask == 1)
        # Area of model grid cells:
        dxdy = dxdy.where(mask == 1)

    # -- Compute Surface Properies -- #
    # Compute sea surface density anomaly using TEOS-10:
    ssd = gsw.sigma0(CT=sst, SA=sss)
    ssd.name = 'sigma0'

    # Expand dimensions of grid cell area to match sea surface density:
    dxdy = dxdy.expand_dims({'time_counter':ssd.time_counter})

    # -- Compute Sea Surface Density Outcrop Area -- #
    # Binned summation of surface density outcrop area in density coordinates.
    ssd_area = xarray_reduce(
            dxdy,
            dxdy.time_counter,
            ssd,
            func="nansum",
            expected_groups=(None, sigma0_bins),
            isbin=[False, True],
            method="map-reduce",
            reindex=False,
            engine='numbagg',
        )

    # Update attributes of DataArray:
    ssd_area.name =  'ssd_area'
    ssd_area.attrs['units'] = 'm2'
    ssd_area.attrs['long_name'] = 'area of sea surface density outcrops'
    ssd_area.attrs['standard_name'] = 'ssd_area'

    return ssd_area

# -- Define function to compute sea water volume census in T-S -- #
def compute_volume_census(thetao:xr.DataArray, so:xr.DataArray, e1t:xr.DataArray, e2t:xr.DataArray, e3t:xr.DataArray, thetao_bins:np.ndarray, so_bins:np.ndarray, mask:xr.DataArray | None = None) -> xr.DataArray:
    """
    Compute sea water volume census in temperature and salinity coordinates.

    The volume of model grid cells is binned according to the temperature and
    salinity of the water mass contained within the grid cell.

    Parameters
    ----------
    thetao : xr.DataArray
        Sea water temperature (C) stored at T-points.
    so : xr.DataArray
        Salinity (g kg-1) stored at T-points.
    e1t : xr.DataArray
        Zonal width of model grid cell (m) on T-points.
    e2t : xr.DataArray
        Meridional width of model grid cell (m) on T-points.
    e3t : xr.DataArray
        Vertical thickness of model grid cell (m) on T-points.
    thetao_bins: ndarray
        Monotonically increasing array of temperature bin edges
        closed on the rightmost edge (i.e., bin_{n} <= x < bin_{n+1}).
    so_bins: ndarray
        Monotonically increasing array of salinity bin edges
        closed on the rightmost edge (i.e., bin_{n} <= x < bin_{n+1}).
    mask: xarray.DataArray
        Ocean basin mask where 1 = included and 0 = excluded values.
        Default value is None.

    Returns
    -------
    xr.DataArray
        Volume of seawater (m3) in temperature and salinity coordinates.

    Raises
    ------
    TypeError
    ValueError

    Example
    -------
    >>> # Calculate sea water volume census in conservative temperature and absolute salinity coordinates:
    >>> compute_volume_census(thetao=ds.thetao,
                                so=ds.so,
                                e1t=ds_grid.e1t,
                                e2t=ds_grid.e2t,
                                e3t=ds_grid.e3t,
                                thetao_bins=np.arange(-2, 30, 0.1),
                                so_bins=np.arange(32, 38, 0.1),
                                mask=ds_mask.atlmask
                                 )

    <xarray.DataArray 'ts_volume' (time_counter: 120, thetao_con_bins: 63, so_abs_bins: 49)> Size: 1MB
    array([[[3.00741814e+12, 3.01658688e+12, 2.82652018e+12, ...,
                        nan,            nan,            nan],
            [4.52284855e+12, 3.97087651e+12, 3.61937799e+12, ...,
                        nan,            nan,            nan],
            [4.04677527e+12, 4.54529717e+12, 5.19145875e+12, ...,
                        nan,            nan,            nan],
            ...,
            [2.22046241e+10, 9.85733650e+09, 2.19657090e+10, ...,
                        nan,            nan,            nan],
            [           nan,            nan,            nan, ...,
                        nan,            nan,            nan],
            [4.39457123e+09, 3.72875289e+09,            nan, ...,
                        nan,            nan,            nan]]])
    Coordinates:
    * time_counter     (time_counter) datetime64[ns] 8B 2023-07-02T12:00:00...
    * thetao_con_bins  (thetao_con_bins) object 504B (-2.0, -1.5] ... (29.0, 29.5]
    * so_abs_bins      (so_abs_bins) object 392B (33.0, 33.1] ... (37.800000000...
    """
    # -- Verify input arguments -- #
    # Dimension names:
    if thetao.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("thetao must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if so.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("so must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if e1t.dims != ('y', 'x'):
        raise ValueError("e1t must have dimensions ('y', 'x').")
    if e2t.dims != ('y', 'x'):
        raise ValueError("e2t must have dimensions ('y', 'x').")
    if e3t.dims != ('time_counter', 'deptht', 'y', 'x'):
        raise ValueError("tracer must have dimensions ('time_counter', 'deptht', 'y', 'x').")
    if mask.dims != ('y', 'x'):
        raise ValueError("mask must have dimensions ('y', 'x').")
    # Number of dimensions:
    if thetao.ndim != 4:
        raise ValueError("thetao must be a 4D array.")
    if so.ndim != 4:
        raise ValueError("so must be a 4D array.")
    if e1t.ndim != 2:
        raise ValueError("e1t must be a 2D array.")
    if e2t.ndim != 2:
        raise ValueError("e2t must be a 2D array.")
    if e3t.ndim != 4:
        raise ValueError("e3t must be a 4D array.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    # Type of tracer bins:
    if not isinstance(thetao_bins, np.ndarray):
        raise TypeError('thetao_bins must be specified as an ndarray')
    if not isinstance(so_bins, np.ndarray):
        raise TypeError('so_bins must be specified as an ndarray')

    # -- Apply ocean mask to variables -- #
    if mask is not None:
        # Time-dependent temperature field, thetao(t):
        thetao = thetao.where(mask == 1)
        # Time-dependent salinity field, so(t):
        so = so.where(mask == 1)
        # Time-dependent grid cell thickness, dz(t):
        dzt = e3t.where(mask == 1)
        # Time-independent grid cell width, dx:
        dx = e1t.where(mask == 1)
        # Time-independent grid cell width, dy:
        dy = e2t.where(mask == 1)

    # -- Calculate meridional volume transport -- #
    # Compute grid cell volume (m3):
    vol = (dx * dy * dzt)
    vol.name = 'volume'

    # Compute binned sum of grid cell volume (m3) in discrete T-S bins.
    vol_census = xarray_reduce(
            vol,                  # Meridional volume transport DataArray to bin.
            vol['time_counter'],  # Coordinate DataArrays to retain - 1. time_counter
            thetao,               # Variable used to bin volume - 2. temperature.
            so,                   # Variable used to bin volume - 3. salinity.
            func="nansum",        # Summary operation within bins - no need to include x coordinate since sum is over x-tracer.
            expected_groups=(None, thetao_bins, so_bins), # Bins specified for each group - None for existing labels.
            isbin=(False, True, True),
            method="map-reduce",
            reindex=False, # Do not reindex during block aggregations to reduce memory at cost of performance.
            engine='numbagg' # Use numbagg grouped aggregations.
        )

    # Update DataArray attributes:
    vol_census.name = 'ts_volume'
    vol_census.attrs['units'] = 'm3'
    vol_census.attrs['long_name'] = 'volume of seawater in temperature and salinity coordinates'
    vol_census.attrs['standard_name'] = 'ts_volume'

    return vol_census

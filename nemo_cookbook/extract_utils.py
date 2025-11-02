"""
extract_utils.py

Description: Functions to extract NEMO ocean model coordinates
cooresponding to a user-defined hydrographic sections.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 20/04/2025
"""

# -- Import dependencies -- #
import sys
import heapq
import logging
import numpy as np
import xarray as xr
from math import sqrt
from functools import partial
from xarray.indexes import NDPointIndex
from xarray.indexes.nd_point_index import ScipyKDTreeAdapter

# -- Define function to find nearest NEMO eORCA grid coords to observed coords -- #
def _nearest_ji_coords(glamt:xr.DataArray,
                       gphit:xr.DataArray,
                       target_lon:np.ndarray,
                       target_lat:np.ndarray
                       ) -> list[tuple[int, int]]:
    """
    Define (j,i) tuples of the NEMO eORCA grid points closest
    to a given collection of latitudes and longitudes constituting
    an observational array.

    Parameters
    ----------
    glamt : xr.DataArray
        Latitude coordinates of the NEMO model grid on T-points.
    gphit : xr.DataArray
        Longitude coordinates of the NEMO model grid on T-points.
    target_lon : np.ndarray
        Longitude coordinates defining the hydrographic section.
    target_lat : np.ndarray
        Latitude coordinates defining the hydrographic section.

    Returns
    -------
    ji_coords : List[Tuple[int, int]]
        List of (j,i) coordinate tuples identifying the NEMO grid points
        closest to the observation points.
    """
    # Create a dataset with the NEMO model grid coordinates:
    ds = xr.Dataset(coords={"lon": glamt,
                            "lat": gphit})
    ds = ds.squeeze(drop=True)

    # Set the index for BallTree search:
    ds = ds.set_xindex(("lat", "lon"), NDPointIndex, tree_adapter_cls=ScipyKDTreeAdapter)
    # Select the nearest grid points to obs points:
    ds_mdl = ds.sel(lat=target_lat, lon=target_lon, method="nearest")

    # Find the (j,i) coordinates of the nearest grid points:
    n_obs_points = len(target_lat)
    ji_coords = []
    grid_shape = ds.lon.shape
    # Iterate over obs coordinates:
    for n in range(n_obs_points):
        ji_n = np.unravel_index((abs(ds.lon.values - ds_mdl.lon.values[n]) + abs(ds.lat.values - ds_mdl.lat.values[n])).argmin(), grid_shape)
        ji_coords.append((ji_n[0], ji_n[1]))

    return ji_coords


# -- Define A* algorithm to find shortest path on NEMO eORCA grid -- #
def _create_node(position: tuple[int, int],
                 g: float = float('inf'), 
                 h: float = 0.0,
                 parent: dict = None
                 ) -> dict:
    """
    Create a node for the A* algorithm.
    
    Parameters
    ----------
    position: Tuple[int, int]
        The (x, y) coordinates of the node
    g: float
        Cost from start to this node (default: infinity)
    h: float
        Estimated cost from this node to goal (default: 0)
    parent: Dict
    
    Returns
    --------
    node: Dict
        A dictionary containing the node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }


def _compute_heuristic(pos1: tuple[int, int],
                       pos2: tuple[int, int]
                       ) -> float:
    """
    Calculate the estimated distance between two points using Euclidean distance.

    Parameters
    ----------
    pos1: Tuple[int, int]
        The first position (x1, y1).
    pos2: Tuple[int, int]
        The second position (x2, y2).
    
    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    x1, y1 = pos1
    x2, y2 = pos2

    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def _get_valid_neighbors(grid: np.ndarray,
                         position: tuple[int, int]
                         ) -> list[tuple[int, int]]:
    """
    Get all valid neighboring positions in the grid.
    
    Parameters
    ----------
    grid: np.ndarray
        2D numpy array where 0 represents walkable cells and 1 represents obstacles
    
    Returns
    -------
    List[Tuple[int, int]]
        List of valid neighboring positions
    """
    x, y = position
    rows, cols = grid.shape

    # All possible moves (excluding diagonals)
    possible_moves = [
        (x+1, y), (x-1, y),    # Right, Left
        (x, y+1), (x, y-1),    # Up, Down
    ]

    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
        and grid[nx, ny] == 0                # Not an obstacle
    ]


def _reconstruct_path(goal_node: dict) -> list[tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.

    Parameters
    ----------
    goal_node: Dict
        The goal node from which to reconstruct the path.
    
    Returns
    -------
    list[tuple[int, int]]
        The path from start to goal as a list of positions.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]  # Reverse to get path from start to goal


def _find_path(grid: np.ndarray,
               start: tuple[int, int],
               goal: tuple[int, int]
               ) -> list[tuple[int, int]]:
    """
    Find the optimal path from a start coordinate to an end
    coordinate on a grid using A* algorithm.

    Parameters
    ----------
    grid: np.ndarray
        2-dimensional model grid where 0 represents walkable cells
        and 1 represents obstacles.

    Returns
    -------
    list[ruple[int, int]]
        The path from start to goal as a list of positions.
    """
    # Define start node:
    start_node = _create_node(
        position=start,
        g=0,
        h=_compute_heuristic(start, goal)
    )

    # Define open and closed sets:
    open_list = [(start_node['f'], start)]  # Priority queue.
    open_dict = {start: start_node}         # For quick node lookup.
    closed_set = set()                      # Explored nodes.

    while open_list:
        # Get node with lowest f value:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # Check if we've reached the goal:
        if current_pos == goal:
            return _reconstruct_path(current_node)
        closed_set.add(current_pos)

        # Explore neighbors:
        for neighbor_pos in _get_valid_neighbors(grid, current_pos):
            # Skip if already explored:
            if neighbor_pos in closed_set:
                continue

            # Calculate new path cost:
            tentative_g = current_node['g'] + _compute_heuristic(current_pos, neighbor_pos)

            # Create or update neighbor:
            if neighbor_pos not in open_dict:
                neighbor = _create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=_compute_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node

    return []  # No path found.


# -- Define function to construct hydrographic section along NEMO eORCA model grid -- #
def _get_section_coords(glamt: xr.DataArray,
                        gphit: xr.DataArray,
                        target_coords: list[tuple[float, float]],
                        ) -> list[tuple[int, int, str]]:
    """
    Construct hydrographic section along NEMO eORCA model grid.

    Returns the (j, i, flux) tuple coordinates of a continuous zig-zag line on
    the NEMO eORCA grid that minimises the distance to the geographical
    coordinates defining an observational section. The flux parameter defines
    the model grid cell face in terms of U+ (eastern), U- (western) and V+
    (northern) directions.

    Parameters:
    -----------
    glamt : xr.DataArray
        Longitude coordinates of the NEMO model grid on T-points.
    gphit : xr.DataArray
        Latitude coordinates of the NEMO model grid on T-points.
    target_coords : list(tuple[int, int])
        List of model grid coordinates sparsely defining the hydrographic section 
        on the NEMO model grid. Each tuple is of the form (j,i).

    Returns:
    --------
    section_coords : list(tuple[int, int, str])
        List of model grid coordinates and flux directions defining the full
        hydrographic section on the NEMO model grid. Each tuple is of the form
        (j, i, flux), where flux is one of 'U+', 'U-', or 'V+'.
    """
    # -- Verify Inputs -- #
    if not isinstance(glamt, xr.DataArray):
        raise TypeError("glamt must be an xarray DataArray.")
    if not isinstance(gphit, xr.DataArray):
        raise TypeError("gphit must be an xarray DataArray.")
    if not isinstance(target_coords, list):
        raise TypeError("target_coords must be a list of tuples.")
    if not all(isinstance(coord, tuple) and len(coord) == 2 for coord in target_coords):
        raise ValueError("Each element in obs_coords must be a tuple of (j, i).")

    # -- Constructing hydrographic section in NEMO model -- #
    # Define the model grid:
    glamt = glamt.squeeze()
    gphit.squeeze()
    grid = np.zeros_like(glamt)

    # Define empty list to store (j,i) coords:
    complete_path = []

    # Iterate over model grid coords:
    for i in range(len(target_coords) - 1):
        # Define start and goal positions:
        start_pos = target_coords[i]
        goal_pos = target_coords[i+1]

        # Find the optimal path from start to goal position using A* algorithm:
        path = _find_path(grid=grid, start=start_pos, goal=goal_pos)
        # Drop last point of the previous path (repeated):
        complete_path.extend(path[:-1])

    # -- Classify flux direction -- #
    # Define empty list to store section coords:
    section_coords = []

    # Iterate over hydrographic section coords:
    for n in range(len(complete_path) - 1):
        # Get the (j,i) model coords of the n and n+1 points:
        y_n, x_n = complete_path[n]
        y_n1, x_n1 = complete_path[n+1]

        # Determine the flux direction:
        if x_n1 > x_n:
            # Northward [meridional] cell face:
            section_coords.append((y_n, x_n, 'V+'))
        elif y_n1 > y_n:
            # Eastward [zonal] cell face:
            section_coords.append((y_n, x_n, 'U+'))
        elif y_n1 < y_n:
            # Westward [zonal] cell face:
            section_coords.append((y_n, x_n, 'U-'))
        else:
            raise ValueError(f"Invalid flux direction for coords: {(x_n, y_n)} -> {(x_n1, y_n1)}")
        
    return section_coords


# -- Internal Preprocessing Functions -- #
def process_Um(ds: xr.Dataset,
               var: str,
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
    var : str
        Variable name to extract from the dataset. Default options are 'uo',
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
    xarray.Dataset
        Dataset including zonal velocities (m/s), eddy-induced velocities (m/s)
        and vertical grid cell thicknesses (m) on U- grid cell faces defining
        section on NEMO model grid.
    """
    match var:
        case 'uo':
            # Extract U- velocities (m/s):
            variable = ds[var_map.get('uo', 'uo')].isel(x=x_index, y=y_index)
            variable.name = 'velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})

        case 'uo_eiv':
            # Extract eddy-induced U- velocities (m/s):
            variable = ds[var_map.get('uo_eiv', 'uo_eiv')].isel(x=x_index, y=y_index)
            variable.name = 'eddy_induced_velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})

        case 'e3u':
            # Extract vertical grid cell thickness (m):
            variable = ds[var_map.get('e3u', 'e3u')].isel(x=x_index, y=y_index)
            variable.name = 'dz'
            variable = variable.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})

        case _:
            # Extract specified variable without assigned attributes:
            variable = ds[var].isel(x=x_index, y=y_index)
            variable.name = var

    # Assign coordinates to the dataset:
    variable = (variable.
                assign_coords({'station': stations,
                               'longitude': longitudes,
                               'latitude': latitudes
                               })
                               .rename({'depthu':'depth'})
                               )

    return variable


def process_Up(ds: xr.Dataset,
               var: str,
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
    var : str
        Variable name to extract from the dataset. Default options are 'uo',
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
    xarray.Dataset
        Dataset including zonal velocities (m/s) and vertical grid cell thickness (m)
        on U+ grid cell faces defining section on NEMO model grid.
    """
    match var:
        case 'uo':
            # Extract U+ velocities (m/s):
            variable = -ds[var_map.get('uo', 'uo')].isel(x=x_index, y=y_index)
            variable.name = 'velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})

        case 'uo_eiv':
            # Extract eddy-induced U+ velocities (m/s):
            variable = -ds[var_map.get('uo_eiv', 'uo_eiv')].isel(x=x_index, y=y_index)
            variable.name = 'eddy_induced_velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})

        case 'e3u':
            # Extract vertical grid cell thickness (m):
            variable = ds[var_map.get('e3u', 'e3u')].isel(x=x_index, y=y_index)
            variable.name = 'dz'
            variable = variable.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})

        case _:
            # Extract specified variable without assigned attributes:
            variable = ds[var].isel(x=x_index, y=y_index)
            variable.name = var

    # Assign coordinates to the dataset:
    variable = (variable.
                assign_coords({'station': stations,
                               'longitude': longitudes,
                               'latitude': latitudes
                               })
                               .rename({'depthu':'depth'})
                               )

    return variable

def process_Vp(ds: xr.Dataset,
               var: str,
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
    var : str
        Variable name to extract from the dataset. Default options are 'vo',
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
    xarray.Dataset
        Dataset including meridional velocities (m/s) and vertical grid cell thickness (m)
        on V+ grid cell faces defining section on NEMO model grid.
    """
    match var:
        case 'vo':
            # Extract V+ velocities (m/s):
            variable = ds[var_map.get('vo', 'vo')].isel(x=x_index, y=y_index)
            variable.name = 'velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_velocity', 'units': 'm/s'})

        case 'vo_eiv':
            # Extract eddy-induced V+ velocities (m/s):
            variable = ds[var_map.get('vo_eiv', 'vo_eiv')].isel(x=x_index, y=y_index)
            variable.name = 'eddy_induced_velocity'
            variable = variable.assign_attrs({'standard_name': 'sea_water_bolus_velocity', 'units': 'm/s'})
        
        case 'e3v':
            # Extract vertical grid cell thickness (m):
            variable = ds[var_map.get('e3v', 'e3v')].isel(x=x_index, y=y_index)
            variable.name = 'dz'
            variable = variable.assign_attrs({'standard_name': 'cell_thickness', 'units': 'm'})

        case _:
            # Extract specified variable without assigned attributes:
            variable = ds[vars].isel(x=x_index, y=y_index)
            variable.name = vars

    # Assign coordinates to the dataset:
    variable = (variable.
                assign_coords({'station': stations,
                               'longitude': longitudes,
                               'latitude': latitudes
                               })
                               .rename({'depthv':'depth'})
                               )
    
    return variable


def process_Tu(ds: xr.Dataset,
               var: str,
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess tracer variables interpolated on U grid cell faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    var : str
        Variable name to be extracted from the dataset. Default options are 'temp',
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
    xarray.Dataset
        Dataset including temperature (C) and salinity (psu | g/kg) on U grid cell
        faces defining section on NEMO model grid.
    """
    match var:
        case 'temp':
            # Extract U temperature (C):
            variable = ds[var_map.get('temp', 'temp')]
            variable = 0.5 * (variable.isel(x=x_index, y=y_index) + variable.isel(x=x_index+1, y=y_index))
            variable.name = 'temp'
            variable = variable.assign_attrs({'standard_name': 'sea_water_temperature'})

        case 'sal':
            # Extract U salinity (psu | g/kg):
            variable = ds[var_map.get('sal', 'sal')]
            variable = 0.5 * (variable.isel(x=x_index, y=y_index) + variable.isel(x=x_index+1, y=y_index))
            variable.name = 'sal'
            variable = variable.assign_attrs({'standard_name': 'sea_water_salinity'})

        case _:
            # Extract specified T variable without assigned attributes:
            variable = 0.5 * (ds[var].isel(x=x_index, y=y_index) + ds[var].isel(x=x_index+1, y=y_index))
            variable.name = var

    # Assign coordinates to the dataset:
    variable = (variable
                .assign_coords({'station': stations,
                               'longitude': longitudes,
                               'latitude': latitudes
                               })
                .rename({'deptht':'depth'})
                )

    return variable


def process_Tv(ds: xr.Dataset,
               var: str,
               var_map: dict,
               x_index: xr.DataArray,
               y_index: xr.DataArray,
               stations: xr.DataArray,
               longitudes: xr.DataArray,
               latitudes: xr.DataArray,
               ) -> xr.Dataset:
    """
    Preprocess tracer variables interpolated on V grid cell faces.

    Parameters
    ----------
    ds : xarray.Dataset
        NEMO model output dataset.
    var : str
        Variable name to be extracted from the dataset. Default options are 'temp',
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
    xarray.Dataset
        Dataset including temperature (C) and salinity (psu | g/kg) on V grid cell
        faces defining section on NEMO model grid.
    """
    match var:
        case 'temp':
            # Extract V temperature (C):
            variable = ds[var_map.get('temp', 'temp')]
            variable = 0.5 * (variable.isel(x=x_index, y=y_index) + variable.isel(x=x_index, y=y_index+1))
            variable.name = 'temp'
            variable = variable.assign_attrs({'standard_name': 'sea_water_temperature'})
    
        case 'sal':
            # Extract V salinity (psu | g/kg):
            variable = ds[var_map.get('sal', 'sal')]
            variable = 0.5 * (variable.isel(x=x_index, y=y_index) + variable.isel(x=x_index, y=y_index+1))
            variable.name = 'sal'
            variable = variable.assign_attrs({'standard_name': 'sea_water_salinity'})

        case _:
            # Extract specified T variable without assigned attributes:
            variable = 0.5 * (ds[var].isel(x=x_index, y=y_index) + ds[var].isel(x=x_index, y=y_index+1))
            variable.name = var

    # Assign coordinates to the dataset:
    variable = (variable
                .assign_coords({'station': stations,
                               'longitude': longitudes,
                               'latitude': latitudes
                               })
                .rename({'deptht':'depth'})
                )
    
    return variable


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
    T_paths : dict[str, list[str]]
        Paths to the NEMO model output T variables. A dictionary of file paths must
        be provided with the variable names as keys and the file paths as values.
    U_paths : dict[str, list[str]]
        Paths to the NEMO model output U variables. A dictionary of file paths must
        be provided with the variable names as keys and the file paths as values.
    V_paths : dict[str, list[str]]
        Paths to the NEMO model output V variables. A dictionary of file paths must
        be provided with the variable names as keys and the file paths as values.
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
    >>> # Define NEMO model output file paths:
    >>> T_paths = {'temp': ['/path/to/nemo/output_gridT_1.nc', '/path/to/nemo/output_gridT_2.nc'],
    >>>            'sal': ['/path/to/nemo/output_gridT_1.nc', '/path/to/nemo/output_gridT_2.nc'],
    >>>            }
    >>> U_paths = {'uo': ['/path/to/nemo/output_gridU_1.nc', '/path/to/nemo/output_gridU_2.nc'],
    >>>            'uo_eiv': ['/path/to/nemo/output_gridU_1.nc', '/path/to/nemo/output_gridU_2.nc'],
    >>>            'e3u': ['/path/to/nemo/output_gridU_1.nc', '/path/to/nemo/output_gridU_2.nc']
    >>>            }
    >>> V_paths = {'vo': ['/path/to/nemo/output_gridV_1.nc', '/path/to/nemo/output_gridV_2.nc'],
    >>>            'vo_eiv': ['/path/to/nemo/output_gridV_1.nc', '/path/to/nemo/output_gridV_2.nc'],
    >>>            'e3v': ['/path/to/nemo/output_gridV_1.nc', '/path/to/nemo/output_gridV_2.nc']
    >>>            }

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
        raise TypeError("section_lon must be a numpy ndarray.")
    if not isinstance(section_lat, np.ndarray):
        raise TypeError("section_lat must be a numpy ndarray.")
    if not isinstance(domain_path, str):
        raise TypeError("domain_path must be a string.")
    if not isinstance(T_paths, dict):
        raise TypeError("T_paths must be either a dictionary.")
    if not isinstance(U_paths, dict):
        raise TypeError("U_paths must be either a dictionary.")
    if not isinstance(V_paths, dict):
        raise TypeError("V_paths must be either a dictionary.")
    if not isinstance(var_map, dict):
        raise TypeError("var_map must be a dictionary.")
    if not isinstance(uv_eiv, bool):
        raise TypeError("uv_eiv must be a boolean.")
    if uv_eiv:
        if var_map.get('uo_eiv', 'uo_eiv') not in U_paths.keys():
            raise KeyError(f"U_paths must contain '{var_map.get('uo_eiv', 'uo_eiv')}' key.")
        if var_map.get('vo_eiv', 'vo_eiv') not in V_paths.keys():
            raise KeyError(f"V_paths must contain '{var_map.get('uo_eiv', 'uo_eiv')}' key.")
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

    # -- Process NEMO domain -- #
    # U+ zonal grid cell width (m):
    e2u = ds_domain_cfg[var_map.get('e2u', 'e2u')]
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
    e1v = ds_domain_cfg[var_map.get('e1v', 'e1v')]
    e1v = e1v.squeeze().isel(x=x_index[vmask], y=y_index[vmask])
    e1v = e1v.assign_coords({'station': station_vmask,
                             'longitude': longitudes[vmask],
                             'latitude': latitudes[vmask]}
                             )
    e1v.name = 'dx'

    if log:
        logging.info("Completed: Extracted variable dx from NEMO model grid.")

    # -- Process NEMO U,V outputs -- #
    # Store U- and U+ variable Datasets in a list:
    UV_list = []
    for var in U_paths.keys():
        # Extract U- velocities & vertical grid cell thicknesses:
        _process_Um = partial(process_Um,
                              var=var,
                              var_map=var_map,
                              x_index=x_index[ummask],
                              y_index=y_index[ummask],
                              stations=station_ummask,
                              longitudes=longitudes[ummask],
                              latitudes=latitudes[ummask],
                              )
        try:
            with xr.open_mfdataset(U_paths[var], preprocess=_process_Um) as ds:
                UV_list.append(ds.load())
            if log:
                logging.info(f"Completed: Extracted U- variable -> {var}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths[var]}.")

        # Extract U+ velocities & vertical grid cell thicknesses:
        _process_Up = partial(process_Up,
                              var=var,
                              var_map=var_map,
                              x_index=x_index[upmask],
                              y_index=y_index[upmask],
                              stations=station_upmask,
                              longitudes=longitudes[upmask],
                              latitudes=latitudes[upmask],
                              )
        try:
            with xr.open_mfdataset(U_paths[var], preprocess=_process_Up) as ds:
                UV_list.append(ds.load())
            if log:
                logging.info(f"Completed: Extracted U+ variable -> {var}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO U-grid file not found at: {U_paths[var]}.")

    # Extract V+ velocities & vertical grid cell thicknesses:
    for var in V_paths.keys():
        _process_Vm = partial(process_Vp,
                                var=var,
                                var_map=var_map,
                                x_index=x_index[vmask],
                                y_index=y_index[vmask],
                                stations=station_vmask,
                                longitudes=longitudes[vmask],
                                latitudes=latitudes[vmask],
                                )
        try:
            with xr.open_mfdataset(V_paths[var], preprocess=_process_Vm) as ds:
                UV_list.append(ds.load())
            if log:
                logging.info(f"Completed: Extracted V variable -> {var}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO V-grid file not found at: {V_paths[var]}.")
    
    # Merge U-, U+ & V variables into single dataset:
    ds_UV = xr.merge(UV_list)

    # Merge zonal, meridional velocities & grid cell properties into a single Dataset:
    ds_uv = xr.merge([ds_UV, e2um, e2up, e1v], combine_attrs='drop_conflicts')

    # Calculate total volume transport normal to the section:
    if uv_eiv:
        ds_uv['volume_transport'] = (ds_uv['velocity'] + ds_uv['eddy_induced_velocity']) * ds_uv['dx'] * ds_uv['dz']
    else:
        ds_uv['volume_transport'] = ds_uv['velocity'] * ds_uv['dx'] * ds_uv['dz']
    if log:
        logging.info("Completed: Calculated volume transport normal to section.")

    # -- Process NEMO T outputs -- #
    # Store T-point variable Datasets in a list:
    T_list = []
    for var in T_paths.keys():
        # Extract U-point temperature and salinity:
        _process_Tu = partial(process_Tu,
                              var=var,
                              var_map=var_map,
                              x_index=x_index[umask],
                              y_index=y_index[umask],
                              stations=station_umask,
                              longitudes=longitudes[umask],
                              latitudes=latitudes[umask]
                              )
        try:
            with xr.open_mfdataset(T_paths[var], preprocess=_process_Tu) as ds:
                T_list.append(ds.load())
            if log:
                logging.info(f"Completed: Extracted T variable on U-face -> {var}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths[var]}.")
        
        # Extract V-point temperature and salinity:
        _process_Tv = partial(process_Tv,
                              var=var,
                              var_map=var_map,
                              x_index=x_index[vmask],
                              y_index=y_index[vmask],
                              stations=station_vmask,
                              longitudes=longitudes[vmask],
                              latitudes=latitudes[vmask]
                              )
        try:
            with xr.open_mfdataset(T_paths[var], preprocess=_process_Tv) as ds:
                T_list.append(ds.load())
            if log:
                logging.info(f"Completed: Extracted T variable on V-face -> {var}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"NEMO T-grid file not found at: {T_paths[var]}.")
        
    # Merge temperature and salinity into a single dataset:
    ds_ts = xr.merge(T_list, combine_attrs='drop_conflicts')

    # Merge velocity and tracers variables into a single Dataset:
    ds_section = xr.merge([ds_uv, ds_ts], combine_attrs='drop_conflicts')

    # Remove unpermitted coordinates:
    permitted_coords = ['time_counter', 'depth', 'station', 'longitude', 'latitude']
    ds_section = ds_section.drop_vars([coord for coord in ds_section.coords if coord not in permitted_coords])

    if log:
        logging.info("Completed: Combined variables along the section.")

    return ds_section

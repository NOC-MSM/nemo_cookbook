"""
extract_utils.py

Description: Functions to extract NEMO ocean model coordinates
cooresponding to a user-defined hydrographic sections.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
Date Created: 20/04/2025
"""

# -- Import dependencies -- #
import heapq
import numpy as np
import xarray as xr
from math import sqrt
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

"""
utils.py

Description:
This module includes utility functions utilised by
the NEMODataTree structure.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import numpy as np
from sklearn.neighbors import BallTree
from xarray.indexes.nd_point_index import TreeAdapter


class SklearnGeoBallTreeAdapter(TreeAdapter):
    """
    Light-weight adapter for BallTree using Haversine distance metric
    to query latitude-longitude coordinates in degrees.

    See:
    https://xarray-indexes.readthedocs.io/blocks/ndpoint.html
    """
    def __init__(self, points: np.ndarray, options: dict):
        """
        Initialize the BallTree with Haversine metric.

        Parameters
        ----------
        points : ndarray
            Array of shape (n_points, 2) containing latitude-longitude
            coordinates in degrees.
        options : dict
            Additional options to pass to BallTree constructor.
        """
        options.update({"metric": "haversine"})
        self._balltree = BallTree(np.deg2rad(points), **options)


    def query(
        self,
        points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Query the BallTree for nearest neighbors.

        Parameters
        ----------
        points : ndarray
            Array of shape (n_query, 2) containing latitude-longitude
            coordinates in degrees.
        
        Returns
        -------
        tuple of ndarrays
            Distances and indices of nearest neighbors.
        """
        return self._balltree.query(np.deg2rad(points))


    def equals(
        self,
        other: "SklearnGeoBallTreeAdapter"
    ) -> bool:
        """
        Check equality with alternative xarray TreeAdapter.

        Parameters
        ----------
        other : TreeAdapter
            Another TreeAdapter instance to compare against.
        """
        return np.array_equal(
            self._balltree.data, other._balltree.data
        )
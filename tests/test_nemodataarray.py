"""
test_nemodataarray.py

Description:
This module includes unit tests for the NEMODataArray class, verifying input
validation and public properties using the idealised global and regional
NEMODataTree fixtures from conftest.py.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import re

import numpy as np
import pytest
import xarray as xr

from nemo_cookbook.nemodataarray import NEMODataArray


# Define utility function to select the appropriate NEMODataTree fixture:
def _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree):
    match dom_type:
        case "global":
            return example_global_nemodatatree
        case "regional":
            return example_regional_nemodatatree
        case _:
            raise ValueError("dom_type must be 'global' or 'regional'")


class TestNEMODataArrayInit:
    """
    Test NEMODataArray.__init__() Input Validation.
    """
    @pytest.mark.parametrize("da_error", [np.ones((3, 5, 10, 10)), 10, "data", None, [1, 2, 3]])
    def test_da_type_error(self, da_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        # Invalid da type (not xarray.DataArray):
        with pytest.raises(TypeError, match=re.escape("da must be specified as an xarray.DataArray.")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    @pytest.mark.parametrize("tree_error", [42, "tree", None, xr.Dataset()])
    def test_tree_type_error(self, tree_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da = nemo["gridT"]["tos_con"]
        # Invalid tree type (not NEMODataTree):
        with pytest.raises(TypeError, match=re.escape("tree must be specified as a NEMODataTree.")):
            NEMODataArray(da=da, tree=tree_error, grid="gridT")

    @pytest.mark.parametrize("grid_error", [1, None, ["gridT"], ("gridT",)])
    def test_grid_type_error(self, grid_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da = nemo["gridT"]["tos_con"]
        # Invalid grid type (not a string):
        with pytest.raises(TypeError, match=re.escape("grid must be specified as a string.")):
            NEMODataArray(da=da, tree=nemo, grid=grid_error)

    def test_grid_key_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da = nemo["gridT"]["tos_con"]
        # Grid name not found in NEMODataTree grids:
        with pytest.raises(KeyError, match=re.escape("gridX not found in available NEMODataTree grids")):
            NEMODataArray(da=da, tree=nemo, grid="gridX")

    def test_da_name_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        # DataArray with dimensions not found in NEMO model grid:
        da_error = xr.DataArray(
            data=np.ones((3, 10, 10)),
            dims=("time_counter", "y", "x"),
        )
        with pytest.raises(ValueError, match=re.escape("not all in NEMO model 'gridT' dimensions")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    def test_da_dims_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        # DataArray with dimension sizes exceeding NEMO model grid:
        da_error = nemo["gridT"]["tos_con"].copy()
        da_error['j'] = np.arange(5, 15, dtype=int)
        with pytest.raises(ValueError, match=re.escape("not all within NEMO model 'gridT' dimension values")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    def test_da_dim_values_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da_error = nemo["gridT"]["tos_con"].copy()
        da_error = da_error.assign_coords(gphit=da_error["gphit"] + 100)
        # DataArray with coordinate values outside of NEMO model grid coordinates:
        with pytest.raises(ValueError, match=re.escape("not all within NEMO model 'gridT' coordinate values")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_init_returns_nemodataarray(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        assert isinstance(nda, NEMODataArray)


class TestNEMODataArrayProperties:
    """
    Test NEMODataArray Class Properties.
    """
    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_data_property(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        assert isinstance(nda.data, xr.DataArray)
        assert nda.data.equals(nemo["gridT"]["tos_con"])

    @pytest.mark.parametrize("dom_type,grid,var", [
        ("global", "gridT", "thetao_con"),
        ("regional", "gridT", "thetao_con"),
        ("global", "gridU", "uo"),
        ("regional", "gridV", "vo"),
    ])
    def test_grid_property(self, dom_type, grid, var, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo[f"{grid}/{var}"]
        assert nda.grid == grid

    @pytest.mark.parametrize("dom_type,grid,var,expected_suffix", [
        ("global", "gridT", "thetao_con", "t"),
        ("global", "gridU", "uo", "u"),
        ("global", "gridV", "vo", "v"),
        ("global", "gridW", "wo", "w"),
        ("global", "gridF", "fo", "f"),
        ("regional", "gridT", "thetao_con", "t"),
        ("regional", "gridU", "uo", "u"),
    ])
    def test_grid_type_property(
        self, dom_type, grid, var, expected_suffix, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo[f"{grid}/{var}"]
        assert nda.grid_type == expected_suffix

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_metrics_2d_variable(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 2-D NEMODataArray metrics contain only (e1, e2) not (e3):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        metrics = nda.metrics
        assert "e1" in metrics
        assert "e2" in metrics
        assert "e3" not in metrics
        assert isinstance(metrics["e1"], NEMODataArray)
        assert isinstance(metrics["e2"], NEMODataArray)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_metrics_3d_variable(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 3-D NEMODataArray metrics contain (e1, e2, e3):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        metrics = nda.metrics
        assert "e1" in metrics
        assert "e2" in metrics
        assert "e3" in metrics
        assert isinstance(metrics["e3"], NEMODataArray)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_mask_2d_variable(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 2-D surface variable returns appropriate {}maskutil:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        mask = nda.mask
        assert isinstance(mask, xr.DataArray)
        assert mask.equals(nemo["gridT"]["tmaskutil"])

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_mask_3d_variable(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 3-D variable returns appropriate {}mask:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        mask = nda.mask
        assert isinstance(mask, xr.DataArray)
        assert mask.equals(nemo["gridT"]["tmask"])

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_2d_masked_property(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 2-dimensional masking behaves equivalent to da.where({}maskutil).
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        masked_nda = nda.masked
        assert isinstance(masked_nda, NEMODataArray)
        expected = nemo["gridT"]["tos_con"].where(nemo["gridT"]["tmaskutil"])
        assert masked_nda.data.equals(expected)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_3d_masked_property(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test 3-dimensional masking behaves equivalent to da.where({}mask).
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridU/uo"]
        masked_nda = nda.masked
        assert isinstance(masked_nda, NEMODataArray)
        expected = nemo["gridU"]["uo"].where(nemo["gridU"]["umask"])
        assert masked_nda.data.equals(expected)

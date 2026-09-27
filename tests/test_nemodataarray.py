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

    def test_da_dims_name_error(self, example_global_nemodatatree):
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
        da_error = da_error.pad(pad_width={"i": (0, 5)})
        with pytest.raises(ValueError, match=re.escape("not all less than or equal to NEMO model 'gridT' dimension sizes")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    def test_da_coords_name_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        # DataArray with coordinates not found in NEMO model grid:
        da_error = nemo["gridT"]["tos_con"].copy()
        da_error = da_error.assign_coords(glam=(["j", "i"], np.ones((10, 10))))

        with pytest.raises(ValueError, match=re.escape("not all in NEMO model 'gridT' coordinates")):
            NEMODataArray(da=da_error, tree=nemo, grid="gridT")

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_init_returns_nemodataarray(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        assert isinstance(nda, NEMODataArray)

class TestNEMODataArrayFromXESMF:
    """
    Test NEMODataArray.from_xesmf() Input Validation and Functionality.
    """
    @pytest.mark.parametrize("da_error", [np.ones((3, 5, 10, 10)), 10, "data", None, [1, 2, 3]])
    def test_da_type_error(self, da_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        # Invalid da type (not xarray.DataArray):
        with pytest.raises(TypeError, match="da must be specified as an xarray.DataArray."):
            NEMODataArray.from_xesmf(da=da_error, tree=nemo, grid="gridT")

    @pytest.mark.parametrize("tree_error", [42, "tree", None, xr.Dataset()])
    def test_tree_type_error(self, tree_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da = nemo["gridT"]["tos_con"]
        # Invalid tree type (not NEMODataTree):
        with pytest.raises(TypeError, match="tree must be specified as a NEMODataTree."):
            NEMODataArray.from_xesmf(da=da, tree=tree_error, grid="gridT")

    @pytest.mark.parametrize("grid_error", [1, None, ["gridT"], ("gridT",)])
    def test_grid_type_error(self, grid_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        da = nemo["gridT"]["tos_con"]
        # Invalid grid type (not a string):
        with pytest.raises(TypeError, match="grid must be specified as a string."):
            NEMODataArray.from_xesmf(da=da, tree=nemo, grid=grid_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_from_xesmf_returns_nemodataarray(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        # Export NEMO variable to xESMF-compatible DataSet:
        ds_xesmf = nemo["gridT/tos_con"].to_xesmf(mask=True)
        nda = NEMODataArray.from_xesmf(da=ds_xesmf['tos_con'], tree=nemo, grid="gridT")
        assert isinstance(nda, NEMODataArray)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_from_xesmf_preserves_data(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        # Export NEMO variable to xESMF-compatible DataSet:
        ds_xesmf = nemo["gridT/tos_con"].to_xesmf(mask=True)
        nda = NEMODataArray.from_xesmf(da=ds_xesmf['tos_con'], tree=nemo, grid="gridT")
        assert np.array_equal(nda.data, ds_xesmf['tos_con'].data, equal_nan=True)

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

class TestNEMODataArrayOperators:
    """
    Test NEMODataArray Class Operators.
    """
    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_unary_operators(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        # __neg__
        assert (-nda).data.equals(-nemo["gridT"]["tos_con"])
        # __pos__
        assert (+nda).data.equals(+nemo["gridT"]["tos_con"])
        # __abs__
        assert (abs(nda)).data.equals(abs(nemo["gridT"]["tos_con"]))

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_binary_operators(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        other = nemo["gridT/tos_con"]
        # __add__
        assert (nda + other).data.equals(nemo["gridT"]["tos_con"] + nemo["gridT"]["tos_con"])
        # __radd__
        assert (2 + nda).data.equals(2 + nemo["gridT"]["tos_con"])

        # __sub__
        assert (nda - other).data.equals(nemo["gridT"]["tos_con"] - nemo["gridT"]["tos_con"])
        # __rsub__
        assert (2 - nda).data.equals(2 - nemo["gridT"]["tos_con"])

        # __mul__
        assert (nda * other).data.equals(nemo["gridT"]["tos_con"] * nemo["gridT"]["tos_con"])
        # __rmul__
        assert (2 * nda).data.equals(2 * nemo["gridT"]["tos_con"])

        # __truediv__
        assert (nda / other).data.equals(nemo["gridT"]["tos_con"] / nemo["gridT"]["tos_con"])
        # __rtruediv__
        assert (2 / nda).data.equals(2 / nemo["gridT"]["tos_con"])

        # __floordiv__
        assert (nda // other).data.equals(nemo["gridT"]["tos_con"] // nemo["gridT"]["tos_con"])
        # __rfloordiv__
        assert (2 // nda).data.equals(2 // nemo["gridT"]["tos_con"])

        # __mod__
        assert (nda % other).data.equals(nemo["gridT"]["tos_con"] % nemo["gridT"]["tos_con"])
        # __rmod__
        assert (2 % nda).data.equals(2 % nemo["gridT"]["tos_con"])

        # __pow__
        assert (nda ** 2).data.equals(nemo["gridT"]["tos_con"] ** 2)
        # __rpow__
        assert (2 ** nda).data.equals(2 ** nemo["gridT"]["tos_con"])

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_comparison_operators(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        other = nemo["gridT/tos_con"]
        # __eq__
        assert (nda == other).data.equals(nemo["gridT"]["tos_con"] == nemo["gridT"]["tos_con"])
        # __ne__
        assert (nda != other).data.equals(nemo["gridT"]["tos_con"] != nemo["gridT"]["tos_con"])

        # __lt__
        assert (nda < other).data.equals(nemo["gridT"]["tos_con"] < nemo["gridT"]["tos_con"])
        # __le__
        assert (nda <= other).data.equals(nemo["gridT"]["tos_con"] <= nemo["gridT"]["tos_con"])

        # __gt__
        assert (nda > other).data.equals(nemo["gridT"]["tos_con"] > nemo["gridT"]["tos_con"])
        # __ge__
        assert (nda >= other).data.equals(nemo["gridT"]["tos_con"] >= nemo["gridT"]["tos_con"])

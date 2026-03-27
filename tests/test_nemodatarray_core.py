"""
test_nemodataarray_core.py

Description:
This module includes unit tests for the NEMODataArray class core
operation methods using the idealised global and regional NEMODataTree
fixtures from conftest.py.

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


class TestNEMODataArrayApplyMask:
    """
    Test NEMODataArray.apply_mask() Input Validation and Behaviour.
    """
    @pytest.mark.parametrize("mask_error", [[True, False], np.ones((10, 10), dtype=bool), 1])
    def test_mask_type_error(self, mask_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("mask must be specified as an xarray.DataArray or None.")):
            nda.apply_mask(mask=mask_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_apply_lsm_only(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test default behaviour of apply_mask(mask=None) -> applies land-sea mask only:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        result = nda.apply_mask()
        assert isinstance(result, NEMODataArray)
        expected = nemo["gridT"]["tos_con"].where(nemo["gridT"]["tmaskutil"])
        assert result.data.equals(expected)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_apply_lsm_subset_only(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test apply_mask(mask=None) for subset of NEMO model grid:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"].sel(i=slice(2, 6), j=slice(2, 6), k=1)
        result = nda.apply_mask()
        assert isinstance(result, NEMODataArray)
        assert ("i" in result.dims) and ("j" in result.dims)
        assert ("k" not in result.dims)
        assert (result.data['i'].size == 5) and (result.data['j'].size == 5)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_apply_custom_mask(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test custom mask is combined with land-sea mask.
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        # Define custom mask (e.g. mask where SST is greater than 5 degC):
        custom_mask = nemo['gridT']['tos_con'] > 5
        result = nda.apply_mask(mask=custom_mask)
        expected = nemo["gridT"]["tos_con"].where(nemo["gridT"]["tmaskutil"] & custom_mask)
        assert isinstance(result, NEMODataArray)
        assert result.data.equals(expected)


class TestNEMODataArraySelLike:
    """
    Test NEMODataArray.sel_like() Input Validation and Behaviour.
    """
    @pytest.mark.parametrize("other_error", [[1, 2], np.ones((10, 10)), 1])
    def test_other_type_error(self, other_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("other must be specified as a NEMODataArray or xarray.DataArray.")):
            nda.sel_like(other_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_nemodataarray_sel_like(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test default behaviour of sel_like(other) -> selects data based on another NEMODataArray:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        # Use NEMODataArray as 'other':
        expected = nda.isel(time_counter=0, k=0)
        result = nda.sel_like(nda.isel(time_counter=0, k=0))
        assert isinstance(result, NEMODataArray)
        assert ("i" in result.dims) and ("j" in result.dims)
        assert ("time_counter" not in result.dims) and ("k" not in result.dims)
        assert result.data.equals(expected.data)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_dataarray_sel_like(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test default behaviour of sel_like(other) -> selects data based on another NEMODataArray:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        # Use xarray.DataArray as 'other':
        expected = nda.isel(time_counter=0, k=0).data
        result = nda.sel_like(nda.isel(time_counter=0, k=0))
        assert isinstance(result, NEMODataArray)
        assert ("i" in result.dims) and ("j" in result.dims)
        assert ("time_counter" not in result.dims) and ("k" not in result.dims)
        assert result.data.equals(expected)

class TestNEMODataArrayWeightedMean:
    """
    Test NEMODataArray.weighted_mean() Input Validation and Behaviour.
    """
    @pytest.mark.parametrize("dims_error", ["j", ("j",), {"j"}, None])
    def test_dims_type_error(self, dims_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("dims must be specified as a list.")):
            nda.weighted_mean(dims=dims_error)

    @pytest.mark.parametrize("skipna_error", [1, 0, "True", 1.0])
    def test_skipna_type_error(self, skipna_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("skipna must be specified as a boolean or None.")):
            nda.weighted_mean(dims=["j"], skipna=skipna_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_weighted_mean_returns_nemodataarray(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        result = nda.weighted_mean(dims=["k"])
        assert isinstance(result, NEMODataArray)
        assert result.name == "wmean_k(thetao_con)"

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_weighted_mean_uniform_field(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Uniform field with uniform weights -> weighted mean equals original field:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        result = nda.weighted_mean(dims=["k"])
        # All non-NaN values are 10; uniform weights (e1t=e2t=1) -> weighted mean = 10
        assert float(result.data.mean()) == pytest.approx(10.0, rel=1e-5)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_weighted_mean_subset(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Subset of uniform field with uniform weights:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"].isel(i=slice(2, 6), j=slice(2, 6))
        result = nda.weighted_mean(dims=["k"])
        # Output Dimensions -> time_counter: 3, i: 4, j: 4
        assert result.data.shape == (3, 4, 4)
        assert ("time_counter" in result.dims) and ("i" in result.dims) and ("j" in result.dims)

class TestNEMODataArrayDiff:
    """
    Test NEMODataArray.diff() Input Validation and C-grid stagger logic.
    """
    @pytest.mark.parametrize("dim_error", [[True, False], None, 1])
    def test_dim_type_error(self, dim_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(ValueError, match="dim must be a string"):
            nda.diff(dim=dim_error)

    def test_dim_key_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(KeyError, match="dimension 'x' not found in"):
            nda.diff(dim="x")

    @pytest.mark.parametrize("fillna_error", ["True", None, 1])
    def test_fillna_type_error(self, fillna_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("`fillna` must be specified as a boolean. Default is False.")):
            nda.diff(dim="i", fillna=fillna_error)

    def test_dim_not_in_2d_var(self, example_global_nemodatatree):
        # Test dimension error -> k not a dimension of the 2-D variable:
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(KeyError, match="dimension 'k' not found in"):
            nda.diff(dim="k")

    @pytest.mark.parametrize("dom_type,grid,var,diff_dim,expected_grid_suffix", [
        # T + diff_i → U-grid:
        ("global", "gridT", "thetao_con", "i", "u"),
        ("regional", "gridT", "thetao_con", "i", "u"),
        # T + diff_j → V-grid:
        ("global", "gridT", "thetao_con", "j", "v"),
        ("regional", "gridT", "thetao_con", "j", "v"),
        # T + diff_k → W-grid:
        ("global", "gridT", "thetao_con", "k", "w"),
        # U + diff_j → F-grid:
        ("global", "gridU", "uo", "j", "f"),
        ("regional", "gridU", "uo", "j", "f"),
        # V + diff_i → F-grid:
        ("global", "gridV", "vo", "i", "f"),
        ("regional", "gridV", "vo", "i", "f"),
        # U + diff_i → T-grid:
        ("global", "gridU", "uo", "i", "t"),
        ("regional", "gridU", "uo", "i", "t"),
        # V + diff_j → T-grid:
        ("global", "gridV", "vo", "j", "t"),
        ("regional", "gridV", "vo", "j", "t"),
    ])
    def test_diff_output_grid_type(
        self, dom_type, grid, var, diff_dim, expected_grid_suffix,
        example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo[f"{grid}/{var}"]
        result = nda.diff(dim=diff_dim, fillna=False)
        assert isinstance(result, NEMODataArray)
        assert result.grid_type == expected_grid_suffix
        assert result.name == f"diff_{diff_dim}({var})"

    def test_diff_iperio_global_no_nans(self, example_global_nemodatatree):
        # Test Case: No NaNs at the wraparound column when differencing global (iperio=True) T-grid:
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        result = nda.diff(dim="i", fillna=False)
        interior_row = result.data.isel(time_counter=0, k=0, j=5)
        assert all(val == pytest.approx(0.0) for val in interior_row.values)

    def test_diff_iperio_regional_boundary_nan(self, example_regional_nemodatatree):
        # Test Case: NaN in (i_end-1) column when differencing regional (iperio=False, fillna=False) T-grid:
        nemo = example_regional_nemodatatree
        nda = nemo["gridT/tos_con"]
        result = nda.diff(dim="i", fillna=False)
        interior_row = result.data.sel(j=5)
        val = float(interior_row.isel(time_counter=0, i=-2))
        assert np.isnan(val)

    def test_diff_fillna_regional_boundary(self, example_regional_nemodatatree):
        # Test Case: No NaN in (i_end-1) column when differencing regional (iperio=False, fillna=True) T-grid:
        nemo = example_regional_nemodatatree
        nda = nemo["gridT/tos_con"]
        result = nda.diff(dim="i", fillna=True)
        interior_row = result.data.sel(j=5)
        val = float(interior_row.isel(time_counter=0, i=-2))
        assert not np.isnan(val)
        # For gradient field (10 degC + 1 degC along i-dim) -> diff_i(tos_con) = 0 - 18 = -18 since RHS boundary NaN -> 0:
        assert val == pytest.approx(-18.0, rel=1e-5)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_diff_uniform_field_zero(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Difference of a uniform interior field is zero at all sea points.
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        # Select 0.5m depth uniform (10 degC) field:
        nda = nemo["gridT/thetao_con"].isel(k=0)
        result = nda.diff(dim="j", fillna=False).data
        # Non-NaN values of a uniform (10 degC) field -> diff_j(thetao_con) -> 0:
        non_nan_vals = result.values[~np.isnan(result.values)]
        assert np.allclose(non_nan_vals, 0.0)

class TestNEMODataArrayDerivative:
    """
    Test NEMODataArray.derivative() Input Validation and Behaviour.
    """
    @pytest.mark.parametrize("dim_error", [[True, False], None, 1])
    def test_dim_type_error(self, dim_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(ValueError, match="dim must be a string"):
            nda.derivative(dim=dim_error)

    def test_dim_key_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(KeyError, match="dimension 'x' not found in"):
            nda.derivative(dim="x")

    @pytest.mark.parametrize("fillna_error", ["True", None, 1])
    def test_fillna_type_error(self, fillna_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("`fillna` must be specified as a boolean. Default is False.")):
            nda.derivative(dim="i", fillna=fillna_error)

    def test_dim_not_in_2d_var(self, example_global_nemodatatree):
        # Test dimension error -> k not a dimension of the 2-D variable:
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(KeyError, match="dimension 'k' not found in"):
            nda.derivative(dim="k")

    @pytest.mark.parametrize("dom_type,grid,var,diff_dim,expected_grid_suffix", [
        # T + diff_i → U-grid:
        ("global", "gridT", "thetao_con", "i", "u"),
        ("regional", "gridT", "thetao_con", "i", "u"),
        # T + diff_j → V-grid:
        ("global", "gridT", "thetao_con", "j", "v"),
        ("regional", "gridT", "thetao_con", "j", "v"),
        # T + diff_k → W-grid:
        ("global", "gridT", "thetao_con", "k", "w"),
        # U + diff_j → F-grid:
        ("global", "gridU", "uo", "j", "f"),
        ("regional", "gridU", "uo", "j", "f"),
        # V + diff_i → F-grid:
        ("global", "gridV", "vo", "i", "f"),
        ("regional", "gridV", "vo", "i", "f"),
        # U + diff_i → T-grid:
        ("global", "gridU", "uo", "i", "t"),
        ("regional", "gridU", "uo", "i", "t"),
        # V + diff_j → T-grid:
        ("global", "gridV", "vo", "j", "t"),
        ("regional", "gridV", "vo", "j", "t"),
    ])
    def test_derivative_output_grid_type(
        self, dom_type, grid, var, diff_dim, expected_grid_suffix,
        example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo[f"{grid}/{var}"]
        result = nda.derivative(dim=diff_dim, fillna=False)
        assert isinstance(result, NEMODataArray)
        assert result.grid_type == expected_grid_suffix
        assert result.name == f"d({var})/d{diff_dim}"

    def test_weights_key_error(self, example_global_nemodatatree):
        # Verify that KeyError is raised when required e3uw grid scale factors are missing from the NEMO model U-grid:
        nemo = example_global_nemodatatree
        nda = nemo["gridU/uo"]
        with pytest.raises(KeyError, match=re.escape("NEMO model grid: 'gridUW' does not contain grid scale factor 'e3uw' required to calculate derivatives along the k-dimension.")):
            nda.derivative(dim="k", fillna=False)


class TestNEMODataArrayIntegral:
    """
    Test NEMODataArray.integral() Input Validation and Behavior.
    """
    def test_cum_dim_not_in_dims_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(ValueError, match=re.escape("cumulative integration dimension 'k' not included in `dims`.")):
            nda.integral(dims=["j"], cum_dims=["k"], dir="+1")

    @pytest.mark.parametrize("dir_error", ["0", "up", None, 1])
    def test_invalid_dir_error(self, dir_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(ValueError, match="invalid direction of cumulative integration"):
            nda.integral(dims=["k"], cum_dims=["k"], dir=dir_error)

    @pytest.mark.parametrize("mask_error", [np.ones((10, 10), dtype=bool), [True, False], 5])
    def test_mask_type_error(self, mask_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(ValueError, match=re.escape("mask must be an xarray.DataArray.")):
            nda.integral(dims=["k"], mask=mask_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_integral_over_k_returns_nemodataarray(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        result = nda.integral(dims=["k"])
        assert isinstance(result, NEMODataArray)
        assert result.name == "integral_k(thetao_con)"

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_integral_over_k_dims(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test full-integration over (k) removes (k) from dimensions:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/thetao_con"].integral(dims=["k"])
        assert "k" not in result.dims
        assert ("j" in result.dims) and ("i" in result.dims)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_integral_over_k_values(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Uniform field (10 degC) and uniform vertical scale factors e3 (50 m)
        # -> k-integral of unmasked column equals 10 * 50 * 5 = 2500.
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/thetao_con"].integral(dims=["k"])
        # Unmasked T-point (j=5, i=5) in both global and regional fixtures:
        val = float(result.data.sel(j=5, i=5).isel(time_counter=0))
        assert val == pytest.approx(2500.0, rel=1e-5)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_integral_over_k_subset(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: k-integral of spatial subset of uniform field:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"].isel(i=slice(2, 6), j=slice(2, 6))
        result = nda.integral(dims=["k"])
        # Output Dimensions -> time_counter: 3, i: 4, j: 4
        assert result.data.shape == (3, 4, 4)
        assert ("time_counter" in result.dims) and ("i" in result.dims) and ("j" in result.dims)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_integral_over_k_i_returns_nemodataarray(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test full-integration over (k, i) returns a NEMODataArray with remaining dimension (j):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/thetao_con"].integral(dims=["i", "k"])
        assert isinstance(result, NEMODataArray)
        assert result.name == "integral_ik(thetao_con)"
        assert "j" in result.dims
        assert ("k" not in result.dims) and ("i" not in result.dims)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_cumulative_integral_preserves_k(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test cumulative integral over (k) retains the (k) dimension:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridU/uo"].integral(dims=["k"], cum_dims=["k"], dir="+1")
        assert isinstance(result, NEMODataArray)
        assert "k" in result.dims

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_cumulative_integral_monotonic(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test cumulative integral (dir=+1) at an unmasked interior column is monotonically increasing:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridU/uo"].integral(dims=["k"], cum_dims=["k"], dir="+1")
        col = result.data.sel(j=5, i=5.5).isel(time_counter=0).values
        assert np.all(np.diff(col[~np.isnan(col)]) >= 0)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_cumulative_integral_reversed(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test cumulative integral (dir=-1) at an unmasked interior column is monotonically decreasing:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridU/uo"].integral(dims=["k"], cum_dims=["k"], dir="-1")
        # k dimension indexes are reversed:
        assert np.all(np.diff(result.data["k"]) < 0)
        col = result.data.sel(j=5, i=5.5).isel(time_counter=0).values
        # Values still increase monotonically since k order is reversed:
        assert np.all(np.diff(col[~np.isnan(col)]) >= 0)


class TestNEMODataArrayDepthIntegral:
    """
    Test NEMODataArray.depth_integral() Input Validation and Behavior.
    """
    @pytest.mark.parametrize("limits", [[0, 100], "0, 100", {"lower": 0, "upper": 100}])
    def test_limits_type_error(self, limits, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(TypeError, match=re.escape("depth limits of integration should be given by a tuple")):
            nda.depth_integral(limits=limits)

    def test_limits_size_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(TypeError, match=re.escape("depth limits of integration should be given by a tuple")):
            nda.depth_integral(limits=(0, 100, 200))

    @pytest.mark.parametrize("limits", [(-1, 5), (0, -1), (-5, -1)])
    def test_limits_negative_error(self, limits, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(ValueError, match=re.escape("depth limits of integration must be non-negative.")):
            nda.depth_integral(limits=limits)

    def test_limits_order_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        with pytest.raises(ValueError, match=re.escape("lower depth limit must be less than upper depth limit.")):
            nda.depth_integral(limits=(100, 50))

    @pytest.mark.parametrize(
        "dom_type, limits",
        [
        ("global", (0, 100)),
        ("regional", (0, 100)),
        ("global", (25, 125)),
        ("regional", (25, 125)),
        ],
    )
    def test_depth_integral_value(
        self, dom_type, limits, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Uniform field (10 degC) & uniform vertical scale factors e3 (50 m) ->
        # Depth Integral = (upper_limit - lower_limit)*10:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        expected_data = (
            nemo["gridT/thetao_con"].data.isel(k=0).drop_vars(["k", "deptht"])
        )
        expected_data = (limits[1] - limits[0]) * expected_data
        result = nemo["gridT/thetao_con"].depth_integral(limits=limits)
        assert isinstance(result, NEMODataArray)
        assert result.name == "integral_z(thetao_con)"
        assert result.data.equals(expected_data)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_depth_integral_subset(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: depth-integral of spatial subset of uniform field:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"].isel(i=slice(2, 6), j=slice(2, 6))
        result = nda.depth_integral(limits=(0, 100))
        # Output Dimensions -> time_counter: 3, i: 4, j: 4
        assert result.data.shape == (3, 4, 4)
        assert ("time_counter" in result.dims) and ("i" in result.dims) and ("j" in result.dims)


class TestNEMODataArrayMaskedStatistic:
    """
    Test NEMODataArray.masked_statistic() Input Validation and Behavior.
    """
    def test_statistic_type_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("statistic must be specified as a string.")):
            nda.masked_statistic(lon_poly=[], lat_poly=[], statistic=None, dims=["j", "i"])

    def test_dims_type_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("dims must be specified as a list.")):
            nda.masked_statistic(lon_poly=[], lat_poly=[], statistic="mean", dims="j")

    @pytest.mark.parametrize("skipna_error", [1, 0, "True"])
    def test_skipna_type_error(self, skipna_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("skipna must be specified as a boolean or None.")):
            nda.masked_statistic(lon_poly=[], lat_poly=[], statistic="mean", dims=["j", "i"], skipna=skipna_error)


class TestNEMODataArrayTransformTo:
    """
    Test NEMODataArray.transform_to() Input Validation and Behavior.
    """
    @pytest.mark.parametrize("to_error", [1, None, ["V"], ("V",)])
    def test_to_type_error(self, to_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(TypeError, match=re.escape("'to' must be a string")):
            nda.transform_to(to=to_error)

    @pytest.mark.parametrize("to_value_error", ["W", "t", "u", "X", ""])
    def test_to_value_error(self, to_value_error, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        with pytest.raises(ValueError, match=re.escape("'to' must be one of ['T', 'U', 'V', 'F']")):
            nda.transform_to(to=to_value_error)

    @pytest.mark.parametrize("dom_type,source_grid,var,target_grid_suffix", [
        ("global", "gridT", "tos_con", "v"),
        ("regional", "gridT", "tos_con", "v"),
        ("global", "gridT", "thetao_con", "u"),
        ("regional", "gridT", "thetao_con", "u"),
        ("global", "gridU", "uo", "t"),
        ("regional", "gridV", "vo", "t"),
    ])
    def test_transform_to_grid_type(
        self, dom_type, source_grid, var, target_grid_suffix,
        example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo[f"{source_grid}/{var}"]
        result = nda.transform_to(to=target_grid_suffix.upper())
        assert isinstance(result, NEMODataArray)
        assert result.grid_type == target_grid_suffix

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_to_returns_correct_grid_path(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/tos_con"].transform_to(to="V")
        assert result.grid == "gridV"

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_to_returns_correct_coords(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/tos_con"].transform_to(to="V")
        assert 'gphiv' in result.data.coords
        assert 'glamv' in result.data.coords
        assert 'depthv' not in result.data.coords

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_to_returns_correct_mask(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        result = nemo["gridT/tos_con"].transform_to(to="V")
        assert result.mask.equals(nemo["gridV"]["vmaskutil"])

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_to_preserves_dims(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test transformed variable has same dimensions as the input:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        result = nda.transform_to(to="U")
        assert ("time_counter" in result.dims) and ("j" in result.dims) and ("i" in result.dims) and ("k" in result.dims)

class TestNEMODataArrayTransformVerticalGrid:
    """
    Test NEMODataArray.transform_vertical_grid() Input Validation and Behavior.
    """
    def test_e3_new_dims_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        # 1-D input DataArray with invalid dimension name:
        e3_error = xr.DataArray(np.ones(5) * 60, dims=["z_new"])
        with pytest.raises(ValueError, match=re.escape("e3_new must be a 1-dimensional xarray.DataArray with dimension 'k_new'.")):
            nda.transform_vertical_grid(e3_new=e3_error)

    def test_e3_new_ndim_error(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        # 2-D input DataArray with invalid dimension:
        e3_error = xr.DataArray(np.ones((3, 5)) * 60, dims=["extra", "k_new"])
        with pytest.raises(ValueError, match=re.escape("e3_new must be a 1-dimensional xarray.DataArray with dimension 'k_new'.")):
            nda.transform_vertical_grid(e3_new=e3_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_e3_new_sum_error(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test sum(e3_new) < max(depth) returns ValueError:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridV/vo"]
        e3_error = xr.DataArray(np.ones(5) * 10.0, dims=["k_new"]) # 50 m < max. depth of 225 m.
        with pytest.raises(ValueError, match=re.escape("e3_new must sum to at least the maximum depth")):
            nda.transform_vertical_grid(e3_new=e3_error)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_vertical_grid_output_type(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test transformation returns an xarray.Dataset:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        # Define sum(e3_new) = 300 m >= max. depth = 225 m:
        e3_new = xr.DataArray(np.ones(5) * 60.0, dims=["k_new"])
        result = nda.transform_vertical_grid(e3_new=e3_new)
        assert isinstance(result, xr.Dataset)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_vertical_grid_output_variables(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        """Output Dataset contains the transformed variable and the new e3 scale factor."""
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"]
        e3_new = xr.DataArray(np.ones(10) * 30.0, dims=["k_new"])
        result = nda.transform_vertical_grid(e3_new=e3_new)
        # Variables:
        assert ("thetao_con" in result) and ("e3t_new" in result)
        assert "deptht_new" in result.coords
        # Dimensions:
        assert result["thetao_con"].dims == ("time_counter", "k_new", "j", "i")
        assert result["e3t_new"].dims == ("time_counter", "k_new", "j", "i")
        assert result["deptht_new"].dims == ("k_new",)
        # Sizes:
        assert result["thetao_con"].shape == (3, 10, 10, 10)
        assert result["e3t_new"].shape == (3, 10, 10, 10)
        assert result["deptht_new"].shape == (10,)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_transform_vertical_grid_subset(
        self, dom_type, example_global_nemodatatree, example_regional_nemodatatree
    ):
        # Test Case: Vertical grid transformation of single profile of a uniform field:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/thetao_con"].isel(time_counter=0, i=5, j=5)
        e3_new = xr.DataArray(np.ones(10) * 30.0, dims=["k_new"])
        result = nda.transform_vertical_grid(e3_new=e3_new)
        # Variables:
        assert ("thetao_con" in result) and ("e3t_new" in result)
        assert "deptht_new" in result.coords
        # Dimensions:
        assert result["thetao_con"].dims == ("k_new",)
        assert result["e3t_new"].dims == ("k_new",)
        assert result["deptht_new"].dims == ("k_new",)
        # Sizes:
        assert result["thetao_con"].shape == (10,)
        assert result["e3t_new"].shape == (10,)
        assert result["deptht_new"].shape == (10,)


class TestNEMODataArrayReductions:
    """
    Test wrapped reduction methods delegating to xr.DataArray and returning NEMODataArray.
    """
    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "min", "max", "sum", "prod"])
    def test_reduction_returns_nemodataarray(self, method, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        result = getattr(nda, method)()
        assert isinstance(result, NEMODataArray)

    @pytest.mark.parametrize("method", ["mean", "std", "min", "max", "sum"])
    def test_reduction_along_dim(self, method, example_global_nemodatatree):
        # Test reduction along time_counter removes that dimension:
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        result = getattr(nda, method)(dim="time_counter")
        assert "time_counter" not in result.dims

    @pytest.mark.parametrize("method,expected_value", [
        ("mean", 10.0),
        ("min", 10.0),
        ("max", 10.0),
    ])
    def test_reduction_uniform_field_value(self, method, expected_value, example_global_nemodatatree):
        # Test Case: Uniform field (10 degC) reductions return original field value:
        nemo = example_global_nemodatatree
        nda = nemo["gridT/thetao_con"]
        result = getattr(nda, method)(skipna=True)
        assert float(result.data) == pytest.approx(expected_value, rel=1e-5)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_reduction_matches_xarray(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # Test reduction result is equivalent to calling xarray method directly:
        nemo = _get_nemodatatree(dom_type, example_global_nemodatatree, example_regional_nemodatatree)
        nda = nemo["gridT/tos_con"]
        result_nda = nda.mean(dim="time_counter")
        result_xr = nemo["gridT"]["tos_con"].mean(dim="time_counter")
        assert result_nda.data.equals(result_xr)


class TestNEMODataArrayGetattr:
    """
    Test __getattr__() delegation to the underlying xr.DataArray.
    """
    @pytest.mark.parametrize("attr", ["name", "dims", "shape", "ndim"])
    def test_attribute_delegation(self, attr, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        da = nemo["gridT"]["tos_con"]
        assert getattr(nda, attr) == getattr(da, attr)

    def test_values_delegation(self, example_global_nemodatatree):
        nemo = example_global_nemodatatree
        nda = nemo["gridT/tos_con"]
        da = nemo["gridT"]["tos_con"]
        np.testing.assert_array_equal(nda.values, da.values)

"""
test_processing.py

Description:
This module includes unit tests for processing utility functions.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import re
import pytest
import numpy as np
import xarray as xr
from nemo_cookbook import processing
from unittest.mock import MagicMock


@pytest.fixture
def create_child_dataset():
    """Fixture factory used to generate example NEMO child domain datasets."""
    def _create_child_dataset(grid: str):
        # Define coordinate names and values based on grid type:
        if ('gridT' in grid) or ('gridW' in grid):
            i_values = np.arange(1, 81)
            j_values = np.arange(1, 81)
        elif 'gridU' in grid:
            i_values = np.arange(1, 81) + 0.5
            j_values = np.arange(1, 81)
        elif 'gridV' in grid:
            i_values = np.arange(1, 81)
            j_values = np.arange(1, 81) + 0.5
        elif 'gridF' in grid:
            i_values = np.arange(1, 81) + 0.5
            j_values = np.arange(1, 81) + 0.5
        else:
            raise ValueError(f"Unrecognised grid type: {grid}")

        # Define example xr.Dataset:
        ds = xr.Dataset(
            coords={
                "i2": ("i2", i_values),
                "j2": ("j2", j_values),
            },
            attrs={
                'rx': 2,
                'ry': 2,
                'imin': 10,
                'imax': 50,
                'jmin': 10,
                'jmax': 50,
                }
        )
        return ds

    return _create_child_dataset


@pytest.mark.parametrize("grid", ['gridT', 'gridU', 'gridV', 'gridW', 'gridF'])
def test_add_parent_indices(create_child_dataset, grid):
    # -- Create example child dataset -- #
    ds = create_child_dataset(grid=grid)
    ds_out = processing._add_parent_indices(ds=ds, grid=grid, label='2')
    # -- Verify type -- #
    assert isinstance(ds_out, xr.Dataset)
    # -- Verify coords -- #
    assert 'i_i2' in ds_out.coords
    assert 'j_j2' in ds_out.coords
    # -- Verify coord values -- #
    if ('gridT' in grid) or ('gridW' in grid):
        assert all(ds_out['i_i2'] % 1 == 0)
        assert all(ds_out['j_j2'] % 1 == 0)
    elif 'gridU' in grid:
        assert all(ds_out['i_i2'] % 1 == 0.5)
        assert all(ds_out['j_j2'] % 1 == 0)
    elif 'gridV' in grid:
        assert all(ds_out['i_i2'] % 1 == 0)
        assert all(ds_out['j_j2'] % 1 == 0.5)
    elif 'gridF' in grid:
        assert all(ds_out['i_i2'] % 1 == 0.5)
        assert all(ds_out['j_j2'] % 1 == 0.5)


def test_get_child_indices():
    # -- Create example child indices -- #
    indices = processing._get_child_indices(imin=10, imax=50, jmin=10, jmax=50,
                                            rx=2, ry=2, nbghost_child=4
                                            )
    # -- Verify type -- #
    assert isinstance(indices, tuple)
    assert all(isinstance(i, int) for i in indices)
    # -- Verify values -- #
    assert len(indices) == 4
    assert indices == (4, 83, 4, 83)


class TestCheckGridDims():
    def test_check_domain_grid_dims(self):
        # -- Create example dataset -- #
        ds = xr.Dataset(coords={"z": ("z", np.arange(1, 51))})
        # -- Verify KeyError -- #
        core_dims = ['nav_lev', 'y', 'x']
        expected_str = f"is missing or exceeding required dimensions {tuple(core_dims)} expected for domain dataset."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._check_grid_dims(ds=ds, grid="domain")


    def test_check_2d_grid_dims(self):
        # -- Create example dataset -- #
        ds = xr.Dataset()
        # -- Verify KeyError -- #
        grid = "gridT"
        core_dims = ['time_counter', 'y', 'x']
        expected_str = f"missing one or more required dimensions {tuple(core_dims)} in {grid} dataset."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._check_grid_dims(ds=ds, grid=grid)


    def test_check_3d_grid_dims(self):
        # -- Create example dataset -- #
        ds = xr.Dataset(coords={"deptht": ("deptht", np.arange(1, 51))})
        # -- Verify KeyError -- #
        grid = "gridT"
        core_dims = ['time_counter', 'deptht', 'y', 'x']
        expected_str = f"missing one or more required dimensions {tuple(core_dims)} in {grid} dataset."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._check_grid_dims(ds=ds, grid=grid)


class TestCheckGridDatasets():
    def test_check_missing_domain_dataset(self):
        # -- Create example datatree dict -- #
        d_example = {
            "gridT": xr.Dataset(),
        }
        # -- Verify KeyError -- #
        expected_str = "missing 'domain': xarray Dataset in dictionary."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._check_grid_datasets(d=d_example)

    def test_check_incompatible_dataset(self):
        # -- Create example datatree dict -- #
        d_example = {
            "domain": xr.Dataset(),
            "gridX": xr.Dataset(),
        }
        # -- Verify ValueError -- #
        grid_keys = ['domain', 'gridT', 'gridU', 'gridV', 'gridW', 'icemod']
        expected_str = f"incompatible key in {d_example.keys()}. Expecting {grid_keys}."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._check_grid_datasets(d=d_example)

    def test_check_dataset_types(self):
        # -- Create example datatree dict -- #
        d_example = {
            "domain": xr.Dataset(),
            "gridT": "not a dataset",
        }
        # -- Verify TypeError -- #
        expected_str = "input dictionary should contain only (str: xarray Dataset) entries."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            processing._check_grid_datasets(d=d_example)

    def test_fill_missing_grids(self, mocker):
        mocker.patch("nemo_cookbook.processing._check_grid_dims", MagicMock())
        # -- Create example datatree dict -- #
        d_example = {
            "domain": xr.Dataset(),
            "gridT": xr.Dataset(),
        }
        # -- Verify missing grid types -- #
        result = processing._check_grid_datasets(d_example)
        grid_keys = ['domain', 'gridT', 'gridU', 'gridV', 'gridW', 'icemod']
        assert set(result.keys()) == set(grid_keys)
        for key in grid_keys:
            assert isinstance(result[key], xr.Dataset)
    

class TestOpenGridDatasets():
    def test_missing_domain_key(self):
        # -- Create example paths dict -- #
        d_in = {'gridT': 'gridT.nc'}
        # -- Verify KeyError -- #
        expected_str = "missing 'domain' key in paths dictionary"
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            processing._open_grid_datasets(d_in=d_in)
 
    def test_invalid_domain_path(self, mocker):
        mocker.patch("nemo_cookbook.processing.xr.open_dataset", side_effect=FileNotFoundError("/invalid/path/to/domain.nc"))
        mocker.patch("nemo_cookbook.processing._check_grid_dims", MagicMock())
        # -- Create example paths dict -- #
        d_in = {"domain": "/invalid/path/to/domain.nc"}
        # -- Verify FileNotFoundError -- #
        with pytest.raises(FileNotFoundError, match="could not open domain configuration file"):
            processing._open_grid_datasets(d_in=d_in)

    def test_grid_file_not_found(self, mocker):
        mocker.patch("nemo_cookbook.processing.xr.open_dataset", side_effect=[xr.Dataset(), FileNotFoundError("gridT.nc")])
        mocker.patch("nemo_cookbook.processing.glob.glob", return_value=['gridT.nc'])
        mocker.patch("nemo_cookbook.processing._check_grid_dims", MagicMock())
        # -- Create example paths dict -- #
        d_in = {'domain': '/path/to/domain.nc', 'gridT': '/path/to/gridT.nc'}
        # -- Verify FileNotFoundError -- #
        with pytest.raises(FileNotFoundError, match="could not open gridT file"):
            processing._open_grid_datasets(d_in=d_in)

    def test_grid_datasets_type(self, mocker):
        mocker.patch("nemo_cookbook.processing.xr.open_dataset", return_value=xr.Dataset())
        mocker.patch("nemo_cookbook.processing.glob.glob", return_value=['/path/to/gridT.nc'])
        mocker.patch("nemo_cookbook.processing._check_grid_dims", MagicMock())
        # -- Create example paths dict -- #
        d_in = {'domain': '/path/to/domain.nc', 'gridT': '/path/to/gridT.nc'}
        # -- Verify types -- #
        result = processing._open_grid_datasets(d_in=d_in)
        assert isinstance(result, dict)
        assert all(isinstance(ds, xr.Dataset) for ds in result.values())
        assert result.keys() == {'domain', 'gridT', 'gridU', 'gridV', 'gridW'}

    def test_grid_mfdatasets_type(self, mocker):
        mocker.patch("nemo_cookbook.processing.xr.open_dataset", return_value=xr.Dataset())
        mocker.patch("nemo_cookbook.processing.glob.glob", return_value=['/path/to/1_gridT.nc', '/path/to/2_gridT.nc'])
        mocker.patch("nemo_cookbook.processing.xr.open_mfdataset", return_value=xr.Dataset())
        mocker.patch("nemo_cookbook.processing._check_grid_dims", MagicMock())
        # -- Create example paths dict -- #
        d_in = {'domain': '/path/to/domain.nc', 'gridT': '/path/to/*_gridT.nc'}
        # -- Verify types -- #
        result = processing._open_grid_datasets(d_in=d_in)
        assert isinstance(result, dict)
        assert all(isinstance(ds, xr.Dataset) for ds in result.values())
        assert result.keys() == {'domain', 'gridT', 'gridU', 'gridV', 'gridW'}

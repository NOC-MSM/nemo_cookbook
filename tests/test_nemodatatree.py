"""
test_nemodatatree.py

Description:
This module includes unit tests for the NEMODataTree class, verifying input
validation and public properties using the idealised global and regional
NEMODataTree fixtures from conftest.py.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import re

import icechunk
import numpy as np
import pytest
import xarray as xr

from nemo_cookbook import NEMODataTree


class TestNEMODataTreePaths():
    @pytest.mark.parametrize("paths", [[], None, "invalid"])
    def test_paths_errors(self, paths):
        # -- Verify TypeError -- #
        expected_str = "`paths` must be a dictionary or nested dictionary."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths=paths)

    @pytest.mark.parametrize("nests", [[], "invalid"])
    def test_nests_errors(self, nests):
        # -- Verify TypeError -- #
        expected_str = "`nests` must be a dictionary or None."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, nests=nests)

    @pytest.mark.parametrize("name", [[], None, 123])
    def test_name_errors(self, name):
        # -- Verify TypeError -- #
        expected_str = "`name` must be a string."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, name=name)
    
    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, iperio):
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity (`iperio`) of parent domain must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, iperio=iperio)
    
    def test_nftype_errors(self):
        # -- Verify ValueError -- #
        expected_str = "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, nftype='invalid')

    @pytest.mark.parametrize("read_mask", ["False", 0])
    def test_read_mask_errors(self, read_mask):
        # -- Verify TypeError -- #
        with pytest.raises(TypeError, match=re.escape("`read_mask` must be a boolean.")):
            NEMODataTree.from_paths(paths={}, read_mask=read_mask)

    @pytest.mark.parametrize("key_linssh", ["False", 0])
    def test_key_linssh_errors(self, key_linssh):
        # -- Verify TypeError -- #
        expected_str = "linear free-surface approximation (`key_linssh`) must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, key_linssh=key_linssh)

    @pytest.mark.parametrize("nbghost_child", ["1", 1.5, None])
    def test_nbghost_child_errors(self, nbghost_child):
        # -- Verify TypeError -- #
        expected_str = "number of ghost cells along the western/southern boundaries (`nbghost_child`) must be an integer."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, nbghost_child=nbghost_child)

    @pytest.mark.parametrize("paths", [{'child': {}}, {'parent': []}])
    def test_paths_value_errors(self, paths):
        # -- Verify ValueError -- #
        expected_str = "Invalid `paths` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths=paths)

    def test_paths_missing_nests_errors(self):
        # -- Create example paths dict -- #
        paths = {'parent': {}, 'child': {}}
        # -- Verify ValueError -- #
        expected_str = "`nests` dictionary must be provided when defining NEMO child domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths=paths)

    def test_paths_key_errors(self):
        # -- Create example paths dict -- #
        key = 'unexpected_key'
        paths = {'parent': {}, key: {}}
        # -- Verify KeyError -- #
        expected_str = f"Unexpected key '{key}' in `paths` dictionary."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths=paths)

    def test_from_paths_type(self, mocker):
        mocker.patch("nemo_cookbook.nemodatatree.create_datatree_dict", return_value={'/gridT': xr.Dataset()})
        # -- Create example paths dict -- #
        paths = {'parent': {}}
        # -- Verify output type -- #
        result = NEMODataTree.from_paths(paths=paths)
        assert isinstance(result, NEMODataTree) & isinstance(result, xr.DataTree)


class TestNEMODataTreeDatasets():
    @pytest.mark.parametrize("datasets", [[], None, "invalid"])
    def test_datasets_errors(self, datasets):
        # -- Verify TypeError -- #
        expected_str = "`datasets` must be a dictionary or nested dictionary."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    @pytest.mark.parametrize("nests", [[], "invalid"])
    def test_nests_errors(self, nests):
        # -- Verify TypeError -- #
        expected_str = "`nests` must be a dictionary or None."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, nests=nests)
    
    @pytest.mark.parametrize("name", [[], None, 123])
    def test_name_errors(self, name):
        # -- Verify TypeError -- #
        expected_str = "`name` must be a string."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, name=name)
    
    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, iperio):
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity (`iperio`) of parent domain must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, iperio=iperio)
    
    def test_nftype_errors(self):
        # -- Verify ValueError -- #
        expected_str = "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, nftype='invalid')

    @pytest.mark.parametrize("read_mask", ["False", 0])
    def test_read_mask_errors(self, read_mask):
        # -- Verify TypeError -- #
        with pytest.raises(TypeError, match=re.escape("`read_mask` must be a boolean.")):
            NEMODataTree.from_datasets(datasets={}, read_mask=read_mask)

    @pytest.mark.parametrize("key_linssh", ["False", 0])
    def test_key_linssh_errors(self, key_linssh):
        # -- Verify TypeError -- #
        expected_str = "linear free-surface approximation (`key_linssh`) must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, key_linssh=key_linssh)

    @pytest.mark.parametrize("nbghost_child", ["1", 1.5, None])
    def test_nbghost_child_errors(self, nbghost_child):
        # -- Verify TypeError -- #
        expected_str = "number of ghost cells along the western/southern boundaries (`nbghost_child`) must be an integer."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, nbghost_child=nbghost_child)

    @pytest.mark.parametrize("datasets", [{'child': {}}, {'parent': []}])
    def test_datasets_value_errors(self, datasets):
        # -- Verify ValueError -- #
        expected_str = "Invalid `datasets` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    def test_datasets_missing_nests_errors(self):
        # -- Create example datasets dict -- #
        datasets = {'parent': {}, 'child': {}}
        # -- Verify ValueError -- #
        expected_str = "`nests` dictionary must be provided when defining NEMO child domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    def test_datasets_key_errors(self):
        # -- Create example datasets dict -- #
        key = 'unexpected_key'
        datasets = {'parent': {}, key: {}}
        # -- Verify KeyError -- #
        expected_str = f"Unexpected key '{key}' in `datasets` dictionary."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    def test_from_datasets_type(self, mocker):
        mocker.patch("nemo_cookbook.nemodatatree.create_datatree_dict", return_value={'gridT': xr.Dataset()})
        # -- Create example datasets dict -- #
        datasets = {'parent': {}}
        # -- Verify output type -- #
        result = NEMODataTree.from_datasets(datasets=datasets)
        assert isinstance(result, NEMODataTree) & isinstance(result, xr.DataTree)

class TestNEMODataTreeFromIcechunk():
    def test_repo_errors(self):
        # -- Verify TypeError -- #
        expected_str = "`repo` must implement readonly_session()."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_icechunk(repo="invalid_repo")

    @pytest.mark.parametrize("name", [["Model"], ("My", "Model"), 123, None])
    def test_name_errors(self, mocker, name):
        mock_repo = mocker.MagicMock(spec=icechunk.repository.Repository)
        # -- Verify TypeError -- #
        expected_str = "`name` must be a string."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_icechunk(repo=mock_repo, name=name)

    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, mocker, iperio):
        mock_repo = mocker.MagicMock(spec=icechunk.repository.Repository)
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity (`iperio`) of parent domain must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_icechunk(repo=mock_repo, iperio=iperio)
    
    @pytest.mark.parametrize("nftype", ["invalid", 123])
    def test_nftype_errors(self, mocker, nftype):
        mock_repo = mocker.MagicMock(spec=icechunk.repository.Repository)
        # -- Verify ValueError -- #
        expected_str = "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_icechunk(repo=mock_repo, nftype=nftype)

    def test_from_icechunk_properties(self, mocker, example_global_nemodatatree):
        # -- Create mock Icechunk repository and session -- #
        mock_repo = mocker.MagicMock(spec=icechunk.repository.Repository)
        mock_session = mocker.MagicMock()
        mock_session.store = "fake_store"
        mock_repo.readonly_session.return_value = mock_session

        # -- Mock xarray.open_datatree to return example NEMODataTree -- #
        mocker.patch("xarray.open_datatree", return_value=example_global_nemodatatree)

        # -- Verify NEMODataTree properties -- #
        result = NEMODataTree.from_icechunk(repo=mock_repo, name="MyModel", iperio=True, nftype="T")
        assert result.name == "MyModel"
        assert result.attrs["iperio"]
        assert result.attrs["nftype"] == "T"

class TestNEMODataTreeFromZarr():
    @pytest.mark.parametrize("store", [["store"], ("store",), 123, None])
    def test_store_errors(self, store):
        # -- Verify TypeError -- #
        expected_str = "`store` must be a string."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_zarr(store=store)

    @pytest.mark.parametrize("name", [['Model'], ("Model",), 123, None])
    def test_name_errors(self, name):
        # -- Verify TypeError -- #
        expected_str = "`name` must be a string."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_zarr(store="path/to/store", name=name)

    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, iperio):
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity (`iperio`) of parent domain must be a boolean."
        with pytest.raises(TypeError, match=re.escape(expected_str)):
            NEMODataTree.from_zarr(store="path/to/store", iperio=iperio)
    
    @pytest.mark.parametrize("nftype", ["invalid", 123])
    def test_nftype_errors(self, nftype):
        # -- Verify ValueError -- #
        expected_str = "north fold type (`nftype`) of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_zarr(store="path/to/store", nftype=nftype)

    def test_from_zarr_properties(self, mocker, example_global_nemodatatree):
        # -- Mock xarray.open_datatree to return example NEMODataTree -- #
        mocker.patch("xarray.open_datatree", return_value=example_global_nemodatatree)

        # -- Verify NEMODataTree properties -- #
        result = NEMODataTree.from_zarr(store="path/to/store", name="MyModel", iperio=True, nftype="T")
        assert result.name == "MyModel"
        assert result.attrs["iperio"]
        assert result.attrs["nftype"] == "T"

class TestNEMODataTreeUtils():
    @pytest.mark.parametrize("dom", [".", "1"])
    def test_get_dom_properties(self, dom):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # -- Verify domain properties -- #
        result = nemo._get_properties(dom=dom)
        assert isinstance(result, tuple)
        assert all(isinstance(item, str) for item in result)
        if dom == ".":
            assert result == ("", "")
        else:
            assert result == (f"{dom}_", f"{dom}")
    
    def test_get_dom_grid_error(self):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # -- Verify KeyError -- #
        grid = "gridT"
        expected_str = f"grid '{grid}' not found in available NEMODataTree grids"
        with pytest.raises(KeyError, match=expected_str):
            nemo._get_properties(grid=grid)

    @pytest.mark.parametrize("infer_dom", [True, False])
    def test_get_grid_properties(self, infer_dom: bool):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # Assign grid node without validation:
        nemo.__setitem__(key='gridT', value=xr.Dataset(), strict=False)
        # -- Verify grid properties -- #
        result = nemo._get_properties(grid="gridT", infer_dom=infer_dom)
        if infer_dom:
            assert isinstance(result, tuple)
            assert all(isinstance(item, str) for item in result)
            assert result == (".", "", "", "t")
        else:
            assert isinstance(result, str)
            assert result == "t"

    @pytest.mark.parametrize("dom", [".", "1"])
    def test_get_grid_node_paths(self, dom: str):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # Assign grid nodes without validation:
        nemo.__setitem__(key='gridT', value=xr.Dataset(), strict=False)
        nemo.__setitem__(key='gridT/1_gridT', value=xr.Dataset(), strict=False)
        # -- Verify grid node paths -- #
        result = nemo._get_grid_paths(dom=dom)
        assert isinstance(result, dict)
        if dom == ".":
            assert result == {"gridT": "gridT"}
        else:
            assert result == {"gridT": f"gridT/{dom}_gridT"}


    @pytest.mark.parametrize("dom", [".", "1"])
    def test_get_dom_ijk_names(self, dom: str):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # -- Verify ijk names -- #
        result = nemo._get_ijk_names(dom=dom)
        assert isinstance(result, dict)
        if dom == ".":
            expected_keys = {'i': 'i', 'j': 'j', 'k': 'k'}
        else:
            expected_keys = {'i': f'i{dom}', 'j': f'j{dom}', 'k': f'k{dom}'}
        assert result == expected_keys

    @pytest.mark.parametrize("grid", ["gridT", "gridT/1_gridT"])
    def test_get_grid_ijk_names(self, grid: str):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # Assign grid nodes without validation:
        nemo.__setitem__(key=grid, value=xr.Dataset(), strict=False)
        # -- Verify ijk names -- #
        result = nemo._get_ijk_names(grid=grid)
        assert isinstance(result, dict)
        if grid == "gridT":
            expected_keys = {'i': 'i', 'j': 'j', 'k': 'k'}
        elif grid == "gridT/1_gridT":
            expected_keys = {'i': 'i1', 'j': 'j1', 'k': 'k1'}
        assert result == expected_keys

    @pytest.mark.parametrize("dims", [['x'], ['x', 'y'], ['x', 'y', 'z']])
    def test_get_grid_weights_value_error(self, dims: list):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        # Assign grid nodes without validation:
        nemo.__setitem__(key="gridT", value=xr.Dataset(), strict=False)
        # -- Verify ValueError -- #
        expected_str = "dims must be a list containing one or more of the following dimensions: ['i', 'j', 'k']."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            nemo._get_weights(grid="gridT", dims=dims)

    def test_get_missing_grid_weights(self):
        # -- Create NEMODataTree instance -- #
        nemo = NEMODataTree()
        grid = "gridT"
        # Assign grid nodes without validation:
        nemo.__setitem__(key=grid, value=xr.Dataset(), strict=False)
        # -- Verify KeyError -- #
        dims = ["i"]
        expected_str = f"weights missing for dimensions {dims} of NEMO model grid {grid}"
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            nemo._get_weights(grid=grid, dims=dims)

    @pytest.mark.parametrize(
            "dom_type,dims",
            [["global", ["i"]], ["global", ["i", "j"]], ["global", ["i", "j", "k"]],
                ["regional", ["i"]], ["regional", ["i", "j"]], ["regional", ["i", "j", "k"]]
                ])
    def test_get_grid_weights_type_masked(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree, dims):
        # -- Create NEMODataTree instance based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")
            
        # Update masks to exclude all land (ocean-only):
        grid = "gridT"
        nemo[grid]["tmaskutil"][:, :] = True
        nemo[grid]["tmask"][:, :, : ] = True

        # -- Verify output type -- #
        result = nemo._get_weights(grid=grid, dims=dims, fillna=True)
        assert isinstance(result, xr.DataArray)
        # -- Verify land-sea mask conservation (no zero / land weights) -- #
        assert np.sum(result == 0) == 0

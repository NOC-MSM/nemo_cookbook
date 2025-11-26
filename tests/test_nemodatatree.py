"""
test_nemodatatree.py

Description:
This module includes unit tests for the NEMODataTree class.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import re
import pytest
import numpy as np
import xarray as xr
from nemo_cookbook import NEMODataTree


class TestNEMODataTreePaths():
    @pytest.mark.parametrize("paths", [[], None, "invalid"])
    def test_paths_errors(self, paths):
        # -- Verify TypeError -- #
        expected_str = "paths must be a dictionary or nested dictionary."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_paths(paths=paths)

    @pytest.mark.parametrize("nests", [[], "invalid"])
    def test_nests_errors(self, nests):
        # -- Verify TypeError -- #
        expected_str = "nests must be a dictionary or None."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_paths(paths={}, nests=nests)
    
    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, iperio):
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity of parent domain must be a boolean."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_paths(paths={}, iperio=iperio)
    
    def test_nftype_errors(self):
        # -- Verify ValueError -- #
        expected_str = "north fold type of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths={}, nftype='invalid')

    @pytest.mark.parametrize("read_mask", ["False", 0])
    def test_read_mask_errors(self, read_mask):
        # -- Verify TypeError -- #
        with pytest.raises(TypeError, match="read_mask must be a boolean."):
            NEMODataTree.from_paths(paths={}, read_mask=read_mask)

    @pytest.mark.parametrize("nbghost_child", ["1", 1.5, None])
    def test_nbghost_child_errors(self, nbghost_child):
        # -- Verify TypeError -- #
        expected_str = "number of ghost cells along the western/southern boundaries must be an integer."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_paths(paths={}, nbghost_child=nbghost_child)

    @pytest.mark.parametrize("paths", [{'child': {}}, {'parent': []}])
    def test_paths_value_errors(self, paths):
        # -- Verify ValueError -- #
        expected_str = "Invalid paths structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_paths(paths=paths)

    def test_paths_key_errors(self):
        # -- Create example paths dict -- #
        key = 'unexpected_key'
        paths = {'parent': {}, key: {}}
        # -- Verify KeyError -- #
        expected_str = f"Unexpected key '{key}' in paths dictionary."
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
        expected_str = "datasets must be a dictionary or nested dictionary."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_datasets(datasets=datasets)

    @pytest.mark.parametrize("nests", [[], "invalid"])
    def test_nests_errors(self, nests):
        # -- Verify TypeError -- #
        expected_str = "nests must be a dictionary or None."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_datasets(datasets={}, nests=nests)
    
    @pytest.mark.parametrize("iperio", ["False", 0])
    def test_iperio_errors(self, iperio):
        # -- Verify TypeError -- #
        expected_str = "zonal periodicity of parent domain must be a boolean."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_datasets(datasets={}, iperio=iperio)
    
    def test_nftype_errors(self):
        # -- Verify ValueError -- #
        expected_str = "north fold type of parent domain must be 'T' (T-pivot fold), 'F' (F-pivot fold), or None."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets={}, nftype='invalid')

    @pytest.mark.parametrize("read_mask", ["False", 0])
    def test_read_mask_errors(self, read_mask):
        # -- Verify TypeError -- #
        with pytest.raises(TypeError, match="read_mask must be a boolean."):
            NEMODataTree.from_datasets(datasets={}, read_mask=read_mask)

    @pytest.mark.parametrize("nbghost_child", ["1", 1.5, None])
    def test_nbghost_child_errors(self, nbghost_child):
        # -- Verify TypeError -- #
        expected_str = "number of ghost cells along the western/southern boundaries must be an integer."
        with pytest.raises(TypeError, match=expected_str):
            NEMODataTree.from_datasets(datasets={}, nbghost_child=nbghost_child)

    @pytest.mark.parametrize("datasets", [{'child': {}}, {'parent': []}])
    def test_datasets_value_errors(self, datasets):
        # -- Verify ValueError -- #
        expected_str = "Invalid datasets structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains."
        with pytest.raises(ValueError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    def test_datasets_key_errors(self):
        # -- Create example datasets dict -- #
        key = 'unexpected_key'
        datasets = {'parent': {}, key: {}}
        # -- Verify KeyError -- #
        expected_str = f"Unexpected key '{key}' in datasets dictionary."
        with pytest.raises(KeyError, match=re.escape(expected_str)):
            NEMODataTree.from_datasets(datasets=datasets)

    def test_from_datasets_type(self, mocker):
        mocker.patch("nemo_cookbook.nemodatatree.create_datatree_dict", return_value={'gridT': xr.Dataset()})
        # -- Create example datasets dict -- #
        datasets = {'parent': {}}
        # -- Verify output type -- #
        result = NEMODataTree.from_datasets(datasets=datasets)
        assert isinstance(result, NEMODataTree) & isinstance(result, xr.DataTree)


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
            expected_str = f"grid '{grid}' not found in the NEMODataTree."
            with pytest.raises(KeyError, match=expected_str):
                nemo._get_properties(grid=grid)

        @pytest.mark.parametrize("infer_dom", [True, False])
        def test_get_grid_properties(self, infer_dom: bool):
            # -- Create NEMODataTree instance -- #
            nemo = NEMODataTree()
            nemo["gridT"] = xr.Dataset()
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
            nemo["gridT"] = xr.Dataset()
            nemo["gridT/1_gridT"] = xr.Dataset()
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
            nemo[grid] = xr.Dataset()
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
            nemo["gridT"] = xr.Dataset()
            # -- Verify ValueError -- #
            expected_str = "dims must be a list containing one or more of the following dimensions: ['i', 'j', 'k']."
            with pytest.raises(ValueError, match=re.escape(expected_str)):
                nemo._get_weights(grid="gridT", dims=dims)

        def test_get_missing_grid_weights(self):
            # -- Create NEMODataTree instance -- #
            nemo = NEMODataTree()
            grid = "gridT"
            nemo[grid] = xr.Dataset()
            # -- Verify KeyError -- #
            dims = ["i"]
            expected_str = f"weights missing for dimensions {dims} of NEMO model grid {grid}:"
            with pytest.raises(KeyError, match=re.escape(expected_str)):
                nemo._get_weights(grid=grid, dims=dims)

        @pytest.mark.parametrize("dims", [["i"], ["i", "j"], ["i", "j", "k"]])
        def test_get_grid_weights_type(self, dims: list):
            # -- Create NEMODataTree instance -- #
            nemo = NEMODataTree()
            grid = "gridT"
            nemo[grid] = xr.Dataset(data_vars={
                'e1t': (("j", "i"), np.ones((10, 10))),
                'e2t': (("j", "i"), np.ones((10, 10))),
                'e3t': (("k", "j", "i"), np.ones((5, 10, 10))),
                'tmask': (("k", "j", "i"), np.ones((5, 10, 10)).astype(bool)),
                'tmaskutil': (("j", "i"), np.ones((10, 10)).astype(bool))
            })

            # -- Verify output type -- #
            result = nemo._get_weights(grid=grid, dims=dims)
            assert isinstance(result, xr.DataArray)
            # -- Verify no zero weights -- #
            assert np.sum(result == 0) == 0


# def test_cell_area():

# def_test_cell_volume():

# def test_gradient():

# def test_divergence():

# def test_curl():

# def test_integral():

# def test_clip_grid():

# def test_clip_domain():

# def test_mask_with_polygon():

# def test_masked_statistic():

# def test_extract_mask_boundary():

# def test_extract_section():

# def test_binned_statistic():

# def test_transform_vertical_grid():

# def test_transform_to():
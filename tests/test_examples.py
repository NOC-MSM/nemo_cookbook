"""
test_examples.py

Description:
This module includes unit tests for examples functions.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
from pathlib import Path

import pytest
import xarray as xr

from nemo_cookbook.examples import get_filepaths


class TestGetFilepaths():
    @pytest.mark.parametrize("invalid_example", [123, None, [], {}])
    def test_get_filepaths_TypeError(self, invalid_example: str):
        with pytest.raises(TypeError, match="`example` must be a string."):
            get_filepaths(example=invalid_example)

    @pytest.mark.parametrize("invalid_example", ["invalid_AMM", "invalid_AGRIF"])
    def test_get_filepaths_ValueError(self, invalid_example: str):
        with pytest.raises(ValueError, match="`example` must be one of"):
            get_filepaths(example=invalid_example)

    @pytest.mark.parametrize("example", ["AMM12", "AGRIF_DEMO", "IHO"])
    def test_get_valid_filepaths(self, example: str):
        # Define dictionary of example filepaths:
        d_example = get_filepaths(example=example)

        assert isinstance(d_example, dict)
        for filepath in d_example.values():
            # Verify that each file exists locally:
            fpath = Path(filepath)
            assert fpath.is_file()


class TestNEMODataTreeExamples():
    def test_orca2_nemodatatree(self, example_ORCA2_nemodatatree):
        # -- Create example NEMODataTree for AGRIF_DEMO configuration -- #
        nemo = example_ORCA2_nemodatatree(linssh=False, vco_ref=False, vco="1d")

        # -- Verify grid nodes, scale factors & coordinates -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridW']:
            assert node in nodes
            grid_suffix = node[-1].lower()
            for factor in [f"e1{grid_suffix}", f"e2{grid_suffix}", f"e3{grid_suffix}"]:
                assert factor in nemo[node].data_vars
            for coord in [f"glam{grid_suffix}", f"gphi{grid_suffix}", f"depth{grid_suffix}"]:
                assert coord in nemo[node].coords

        # -- Tear down -- #
        # Close files associated with NEMODataTree:
        nemo.close()

    def test_orca2_linssh_nemodatatree(self, example_ORCA2_nemodatatree):
        # -- Create example linear free-surface NEMODataTree from AGRIF_DEMO configuration -- #
        nemo = example_ORCA2_nemodatatree(linssh=True, vco_ref=False, vco="1d")

        # -- Verify grid nodes, scale factors & coordinates -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridW']:
            assert node in nodes
            grid_suffix = node[-1].lower()
            for factor in [f"e1{grid_suffix}", f"e2{grid_suffix}", f"e3{grid_suffix}"]:
                assert factor in nemo[node].data_vars
            for coord in [f"glam{grid_suffix}", f"gphi{grid_suffix}", f"depth{grid_suffix}"]:
                assert coord in nemo[node].coords

        # -- Tear down -- #
        # Close files associated with NEMODataTree:
        nemo.close()

    def test_orca2_vco_ref_nemodatatree(self, example_ORCA2_nemodatatree):
        # -- Create example vco_ref NEMODataTree from AGRIF_DEMO configuration -- #
        nemo = example_ORCA2_nemodatatree(linssh=False, vco_ref=True, vco="1d")

        # -- Verify grid nodes, scale factors & coordinates -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridF']:
            assert node in nodes
            grid_suffix = node[-1].lower()
            assert f"e3{grid_suffix}_0" in nemo[node].data_vars
            assert f"h{grid_suffix}_0" in nemo[node].data_vars

        # -- Tear down -- #
        # Close files associated with NEMODataTree:
        nemo.close()

    def test_orca2_agrif_nemodatatree(self, example_ORCA2_AGRIF_nemodatatree):
        # -- Create example nested ORCA2 global NEMODataTree from AGRIF_DEMO configuration -- #
        nemo = example_ORCA2_AGRIF_nemodatatree(nbghost_child=4)

        # -- Verify grid nodes & scale factors -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridW',
                     'gridT/1_gridT', 'gridU/1_gridU', 'gridV/1_gridV', 'gridW/1_gridW',
                     'gridT/1_gridT/2_gridT', 'gridU/1_gridU/2_gridU', 'gridV/1_gridV/2_gridV', 'gridW/1_gridW/2_gridW'
                     ]:
            assert node in nodes
            grid_suffix = node[-1].lower()
            for factor in [f"e1{grid_suffix}", f"e2{grid_suffix}", f"e3{grid_suffix}"]:
                assert factor in nemo[node].data_vars

            # -- Verify child domain sizes & coordinates -- #
            if "2_grid" in node:
                # Verify coords:
                for coord in [f"2_glam{grid_suffix}", f"2_gphi{grid_suffix}", f"2_depth{grid_suffix}"]:
                    assert coord in nemo[node].coords
                # Verify size(i2) = (imax - imin) * rx => (60 - 20) * 3 => 120
                assert nemo[node]["i2"].size == 120
                # Verify size(j2) = (jmax - jmin) * ry => (60 - 27) * 3 => 99
                assert nemo[node]["j2"].size == 99
            elif "1_grid" in node:
                # Verify coords:
                for coord in [f"1_glam{grid_suffix}", f"1_gphi{grid_suffix}", f"1_depth{grid_suffix}"]:
                    assert coord in nemo[node].coords
                # Verify size(i1) = (imax - imin) * rx => (146 - 121) * 4 => 100
                assert nemo[node]["i1"].size == 100
                # Verify size(j1) = (jmax - jmin) * ry => (133 - 113) * 4 => 80
                assert nemo[node]["j1"].size == 80

        # Close files associated with NEMODataTree:
        nemo.close()


    def test_orca2_agrif_nbghost_child_nemodatatree(self, example_ORCA2_AGRIF_nemodatatree):
        # -- Create example nested ORCA2 global NEMODataTree from AGRIF_DEMO configuration -- #
        nemo = example_ORCA2_AGRIF_nemodatatree(nbghost_child=None)

        # -- Verify grid nodes & scale factors -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridW',
                     'gridT/1_gridT', 'gridU/1_gridU', 'gridV/1_gridV', 'gridW/1_gridW',
                     'gridT/1_gridT/2_gridT', 'gridU/1_gridU/2_gridU', 'gridV/1_gridV/2_gridV', 'gridW/1_gridW/2_gridW'
                     ]:
            assert node in nodes
            grid_suffix = node[-1].lower()
            for factor in [f"e1{grid_suffix}", f"e2{grid_suffix}", f"e3{grid_suffix}"]:
                assert factor in nemo[node].data_vars

            # -- Verify child domain sizes & coordinates -- #
            if "2_grid" in node:
                # Verify coords:
                for coord in [f"2_glam{grid_suffix}", f"2_gphi{grid_suffix}", f"2_depth{grid_suffix}"]:
                    assert coord in nemo[node].coords
                # Verify size(i2) = (imax - imin) * rx + 2 * nbghost_child => (60 - 20) * 3 + 8 => 128
                assert nemo[node]["i2"].size == 128
                # Verify size(j2) = (jmax - jmin) * ry + 2 * nbghost_child => (60 - 27) * 3 + 8 => 107
                assert nemo[node]["j2"].size == 107
            elif "1_grid" in node:
                # Verify scale factors:
                for coord in [f"1_glam{grid_suffix}", f"1_gphi{grid_suffix}", f"1_depth{grid_suffix}"]:
                    assert coord in nemo[node].coords
                # Verify size(i1) = (imax - imin) * rx + 2 * nbghost_child => (146 - 121) * 4 + 8 => 108
                assert nemo[node]["i1"].size == 108
                # Verify size(j1) = (jmax - jmin) * ry + 2 * nbghost_child => (133 - 113) * 4 + 8 => 88
                assert nemo[node]["j1"].size == 88

        # Close files associated with NEMODataTree:
        nemo.close()


    def test_amm12_nemodatatree(self, example_AMM12_nemodatatree):
        # -- Create example NEMODataTree for AMM12 configuration -- #
        nemo = example_AMM12_nemodatatree

        # -- Verify grid nodes, scale factors & coordinates -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV']:
            assert node in nodes
            grid_suffix = node[-1].lower()
            for factor in [f"e1{grid_suffix}", f"e2{grid_suffix}"]:
                assert factor in nemo[node].data_vars
            for coord in [f"glam{grid_suffix}", f"gphi{grid_suffix}"]:
                assert coord in nemo[node].coords

        # Close files associated with NEMODataTree:
        nemo.close()

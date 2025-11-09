"""
test_examples.py

Description:
This module includes unit tests for examples functions.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import pytest
import xarray as xr
from pathlib import Path
from nemo_cookbook import NEMODataTree
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
    def test_orca2_nemodatatree(self):
        # -- Create example NEMODataTree for AGRIF_DEMO configuration -- #
        # Get dict of example filepaths:
        filepaths = get_filepaths("AGRIF_DEMO")
        # Define paths dict for NEMODataTree:
        paths = {"parent": {
                 "domain": filepaths["domain_cfg.nc"],
                 "gridT": filepaths["ORCA2_5d_00010101_00010110_grid_T.nc"],
                 "gridU": filepaths["ORCA2_5d_00010101_00010110_grid_U.nc"],
                 "gridV": filepaths["ORCA2_5d_00010101_00010110_grid_V.nc"],
                 "gridW": filepaths["ORCA2_5d_00010101_00010110_grid_W.nc"],
                 "icemod": filepaths["ORCA2_5d_00010101_00010110_icemod.nc"]
                }}
        # Create NEMODataTree from paths dict:
        nemo = NEMODataTree.from_paths(paths, iperio=True, nftype="T")

        # -- Verify output -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV', 'gridW']:
            assert node in nodes

        # -- Tear down -- #
        # Close files associated with NEMODataTree:
        nemo.close()


    def test_amm12_nemodatatree(self):
        # -- Create example NEMODataTree for AMM12 configuration -- #
        # Get dict of example filepaths:
        filepaths = get_filepaths("AMM12")
        # Define paths dict for NEMODataTree:
        paths = {"parent": {
                 "domain": filepaths["domain_cfg.nc"],
                 "gridT": filepaths["AMM12_1d_20120102_20120110_grid_T.nc"],
                 "gridU": filepaths["AMM12_1d_20120102_20120110_grid_U.nc"],
                 "gridV": filepaths["AMM12_1d_20120102_20120110_grid_V.nc"],
                }}
        # Create NEMODataTree from paths dict:
        nemo = NEMODataTree.from_paths(paths, iperio=False)

        # -- Verify output -- #
        assert isinstance(nemo, xr.DataTree)
        nodes = [entry[0] for entry in list(nemo.subtree_with_keys)]
        for node in ['gridT', 'gridU', 'gridV']:
            assert node in nodes

        # Close files associated with NEMODataTree:
        nemo.close()

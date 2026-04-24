"""
test_nemodatatree_core.py

Description:
This module includes unit tests for the NEMODataTree class core
operation methods using the idealised global and regional NEMODataTree
fixtures from conftest.py.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import re

import numpy as np
import pytest
import xarray as xr


class TestCellArea():
    @pytest.mark.parametrize(
            "dom_type, grid",
            [("global", "gridT"), ("regional", "gridT"), ("global", "gridU"), ("regional", "gridU"),
             ("global", "gridV"), ("regional", "gridV"), ("global", "gridW"), ("regional", "gridW"),
             ("global", "gridF"), ("regional", "gridF")
             ])
    def test_cell_area(self, dom_type, grid, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Define grid suffix -- #
        grid_suffix = grid[-1].lower()

        # -- Zonal grid cell face area -- #
        data = (
            nemo[grid][f"e3{grid_suffix}"].where(nemo[grid][f"{grid_suffix}mask"])
            * nemo[grid][f"e2{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        areacello = nemo.cell_area(grid=grid, dim='i')

        assert areacello.name == "areacello"
        assert areacello.equals(data)

        # -- Meridional grid cell face area -- #
        data = (
            nemo[grid][f"e3{grid_suffix}"].where(nemo[grid][f"{grid_suffix}mask"])
            * nemo[grid][f"e1{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        areacello = nemo.cell_area(grid=grid, dim='j')

        assert areacello.name == "areacello"
        assert areacello.equals(data)

        # -- Horizontal grid cell area -- #
        data = (
            nemo[grid][f"e1{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            * nemo[grid][f"e2{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        areacello = nemo.cell_area(grid=grid, dim='k')

        assert areacello.name == "areacello"
        assert areacello.equals(data)

class TestCellVolume():
    @pytest.mark.parametrize(
            "dom_type, grid",
            [("global", "gridT"), ("regional", "gridT"), ("global", "gridU"), ("regional", "gridU"),
             ("global", "gridV"), ("regional", "gridV"), ("global", "gridW"), ("regional", "gridW"),
             ("global", "gridF"), ("regional", "gridF")
             ])
    def test_cell_volume(self, dom_type, grid, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Define grid suffix -- #
        grid_suffix = grid[-1].lower()

        # -- Grid cell volume -- #
        data = (
            nemo[grid][f"e3{grid_suffix}"].where(nemo[grid][f"{grid_suffix}mask"])
            * nemo[grid][f"e2{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            * nemo[grid][f"e1{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        volcello = nemo.cell_volume(grid=grid)

        # -- Verify equal dims, coords and data values -- #
        assert volcello.name == "volcello"
        assert volcello.equals(data)

class TestClipGrid():
    @pytest.mark.parametrize("bbox", [[0, 1, 0, 2], (0, 1), "0 1 2 3"])
    def test_bbox_type(self, bbox, example_global_nemodatatree):
        # -- Verify ValueError is raised for invalid bbox -- #
        with pytest.raises(ValueError, match=re.escape("bounding box must be a tuple (lon_min, lon_max, lat_min, lat_max).")):
            example_global_nemodatatree.clip_grid(grid="gridT", bbox=bbox)

    @pytest.mark.parametrize(
            "dom_type, grid",
            [("global", "gridT"), ("regional", "gridT"), ("global", "gridU"), ("regional", "gridU"),
             ("global", "gridV"), ("regional", "gridV"), ("global", "gridW"), ("regional", "gridW"),
             ("global", "gridF"), ("regional", "gridF")
             ])
    def test_clip_grid(self, dom_type, grid, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
                bbox = (40, 62, -50, -32)
            case "global":
                nemo = example_global_nemodatatree
                bbox = (-45, 60, -25, 30)
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        grid_suffix = grid[-1].lower()

        # -- Clip NEMO model grid -- #
        nemo_clipped = nemo.clip_grid(grid=grid, bbox=bbox)

        # -- Validate Clipped NEMODataTree -- #
        # Expect NEMODataTree is returned:
        assert isinstance(nemo_clipped, type(nemo))
        # Expect all NEMO model grid nodes are retained:
        assert nemo_clipped.groups == nemo.groups

        # Expect clipped grid dims sizes to be <= original NEMO model grid:
        assert nemo_clipped[grid].sizes["i"] <= nemo[grid].sizes["i"]
        assert nemo_clipped[grid].sizes["j"] <= nemo[grid].sizes["j"]

        # Expect all grid coordinates to be within bounding box:
        assert nemo_clipped[grid][f"glam{grid_suffix}"].min() >= bbox[0]
        assert nemo_clipped[grid][f"glam{grid_suffix}"].max() <= bbox[1]
        assert nemo_clipped[grid][f"gphi{grid_suffix}"].min() >= bbox[2]
        assert nemo_clipped[grid][f"gphi{grid_suffix}"].max() <= bbox[3]

class TestClipDomain():
    @pytest.mark.parametrize("bbox", [[0, 1, 0, 2], (0, 1), "0 1 2 3"])
    def test_bbox_type(self, bbox, example_global_nemodatatree):
        # -- Verify ValueError is raised for invalid bbox -- #
        with pytest.raises(ValueError, match=re.escape("bounding box must be a tuple (lon_min, lon_max, lat_min, lat_max).")):
            example_global_nemodatatree.clip_grid(grid="gridT", bbox=bbox)

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_clip_domain(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
                bbox = (40, 62, -50, -32)
            case "global":
                nemo = example_global_nemodatatree
                bbox = (-45, 60, -25, 30)
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Clip NEMO model domain -- #
        nemo_clipped = nemo.clip_domain(dom='.', bbox=bbox)

        # -- Validate Clipped NEMODataTree -- #
        # Expect NEMODataTree is returned:
        assert isinstance(nemo_clipped, type(nemo))
        # Expect all NEMO model grid nodes are retained:
        assert nemo_clipped.groups == nemo.groups

        # Expect all T-grid coordinates to be within bounding box:
        assert nemo_clipped["gridT"]["glamt"].min() >= bbox[0]
        assert nemo_clipped["gridT"]["glamt"].max() <= bbox[1]
        assert nemo_clipped["gridT"]["gphit"].min() >= bbox[2]
        assert nemo_clipped["gridT"]["gphit"].max() <= bbox[3]

        grids = ["gridT", "gridU", "gridV", "gridW", "gridF"]
        # Expect clipped grid dims sizes to be <= original NEMO model grid:
        assert all(nemo_clipped[grid].sizes["i"] <= nemo[grid].sizes["i"] for grid in grids)
        assert all(nemo_clipped[grid].sizes["j"] <= nemo[grid].sizes["j"] for grid in grids)
    
        # Expect clipped grid dims sizes to be consistent across all grids:
        assert len(set([nemo_clipped[grid].sizes["i"] for grid in grids])) == 1
        assert len(set([nemo_clipped[grid].sizes["j"] for grid in grids])) == 1

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_clip_domain_sizes(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
                bbox = (40, 62, -50, -32)
            case "global":
                nemo = example_global_nemodatatree
                bbox = (-45, 60, -25, 30)
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Clip NEMO model domain -- #
        nemo_clipped = nemo.clip_domain(dom='.', bbox=bbox)

        # -- Validate Clipped NEMODataTree -- #
        grids = ["gridT", "gridU", "gridV", "gridW", "gridF"]
        # Expect clipped grid dims sizes to be consistent across all grids:
        assert len(set([nemo_clipped[grid].sizes["i"] for grid in grids])) == 1
        assert len(set([nemo_clipped[grid].sizes["j"] for grid in grids])) == 1

class TestExtractSection():
    @pytest.mark.parametrize("lon_section", [[0, 1, 0], "glamt"])
    def test_lon_type(self, lon_section, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid lon_section type -- #
        with pytest.raises(TypeError, match="lon_section must be a numpy array."):
            example_global_nemodatatree.extract_section(lon_section=lon_section, lat_section=np.array([0, 1]), uv_vars=["uo", "vo"], vars=None, dom=".")

    @pytest.mark.parametrize("lat_section", [[0, 1, 0], "gphit"])
    def test_lat_type(self, lat_section, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid lat_section type -- #
        with pytest.raises(TypeError, match="lat_section must be a numpy array."):
            example_global_nemodatatree.extract_section(lon_section=np.array([0, 1]), lat_section=lat_section, uv_vars=["uo", "vo"], vars=None, dom=".")

    @pytest.mark.parametrize("uv_vars", [{"u": "uo", "v": "vo"}, "uo, vo", ["uo", "vo", "wo"]])
    def test_uv_vars_values(self, uv_vars, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid uv_vars type -- #
        with pytest.raises(TypeError, match=re.escape("uv_vars must be a list of velocity variables to extract (e.g., ['uo', 'vo']).")):
            example_global_nemodatatree.extract_section(lon_section=np.array([0, 1]), lat_section=np.array([0, 1]), uv_vars=uv_vars, vars=None, dom=".")

    def test_extract_section(self, example_global_nemodatatree):
        # -- Define NEMODataTree based on domain type -- #
        nemo = example_global_nemodatatree

        # -- Defining idealised section endpoints -- #
        lon_section = np.array([-50, 50])
        lat_section = np.array([-20, 40])
        
        # -- Extract mask boundary -- #
        ds_bdy = nemo.extract_section(lon_section=lon_section,
                                      lat_section=lat_section,
                                      uv_vars=["uo", "vo"],
                                      vars=["thetao_con"],
                                      dom="."
                                      )

        # -- Verify section properties -- #
        # Expect section to have 6 grid cell faces:
        assert ds_bdy['bdy'].size == 6
        # Expected section coordinates:
        for coord in ["gphib", "glamb", "depthb"]:
            assert coord in ds_bdy.coords

        # -- Verify section grid cell faces -- #
        expected_flux_types = ["U", "V", "U", "V", "U", "V"]
        assert ds_bdy['flux_type'].values.tolist() == expected_flux_types
        # Flux direction is defined as positive for northward/eastward fluxes:
        expected_flux_dir = [-1, 1, -1, 1, -1, 1]
        assert ds_bdy['flux_dir'].values.tolist() == expected_flux_dir

        # -- Verify section variables -- #
        # Expected section variables:
        for var in ["i_bdy", "j_bdy", "e1b", "e3b", "bmask"]:
             assert var in ds_bdy.data_vars

        # Expected velocity variable:
        assert "velocity" in ds_bdy.data_vars
        assert ds_bdy['velocity'].dims == ("time_counter", "k", "bdy")
        # Expected tracer variable:
        assert "thetao_con" in ds_bdy.data_vars
        assert ds_bdy['thetao_con'].dims == ("time_counter", "k", "bdy")

class TestExtractMaskBoundary():
    @pytest.mark.parametrize("mask", [[0, 1, 0], "mask_name"])
    def test_mask_type(self, mask, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid mask type -- #
        with pytest.raises(TypeError, match="mask must be an xarray DataArray"):
            example_global_nemodatatree.extract_mask_boundary(mask=mask, uv_vars=["uo", "vo"], vars=None, dom=".")

    def test_mask_dims(self, example_global_nemodatatree):
        # -- Verify ValueError is raised for invalid mask dimensions -- #
        nemo = example_global_nemodatatree
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True
        da_mask = xr.DataArray(mask, dims=["y", "x"])
        with pytest.raises(ValueError, match=re.escape("mask must have dimensions 'i' and 'j'")):
            nemo.extract_mask_boundary(mask=da_mask, uv_vars=["uo", "vo"], vars=None, dom=".")

    @pytest.mark.parametrize("uv_vars", [{"u": "uo", "v": "vo"}, "uo, vo", ["uo", "vo", "wo"]])
    def test_uv_vars_values(self, uv_vars, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid uv_vars type -- #
        with pytest.raises(TypeError, match=re.escape("uv_vars must be a list of velocity variables to extract (e.g., ['uo', 'vo']).")):
            example_global_nemodatatree.extract_mask_boundary(mask=xr.DataArray(), uv_vars=uv_vars, vars=None, dom=".")

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_extract_mask_boundary(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Defining idealised mask -- #
        # Define 2x2 square mask at the domain centre:
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True
        da_mask = xr.DataArray(mask,
                               dims=["j", "i"],
                               coords={"j": nemo["gridT"]["j"], "i": nemo["gridT"]["i"]}
                               )
        
        # -- Extract mask boundary -- #
        ds_bdy = nemo.extract_mask_boundary(mask=da_mask,
                                            uv_vars=["uo", "vo"],
                                            vars=["thetao_con"],
                                            dom="."
                                            )

        # -- Verify boundary properties -- #
        # Expect boundary to have 8 grid cell faces:
        assert ds_bdy['bdy'].size == 8
        # Expected boundary coordinates:
        for coord in ["gphib", "glamb", "depthb"]:
            assert coord in ds_bdy.coords
        # Expected boundary variables:
        for var in ["i_bdy", "j_bdy", "e1b", "e3b", "bmask"]:
             assert var in ds_bdy.data_vars

        # -- Verify boundary grid cell faces -- #
        expected_flux_types = ["U", "U", "V", "V", "U", "U", "V", "V"]
        assert ds_bdy['flux_type'].values.tolist() == expected_flux_types
        # Flux direction is defined as the outward normal:
        expected_flux_dir = [1, 1, -1, -1, -1, -1, 1, 1]
        assert ds_bdy['flux_dir'].values.tolist() == expected_flux_dir

        # Expected velocity variable:
        assert "velocity" in ds_bdy.data_vars
        assert ds_bdy['velocity'].dims == ("time_counter", "k", "bdy")
        # Expected tracer variable:
        assert "thetao_con" in ds_bdy.data_vars
        assert ds_bdy['thetao_con'].dims == ("time_counter", "k", "bdy")

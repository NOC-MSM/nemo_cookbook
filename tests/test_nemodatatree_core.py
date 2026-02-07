"""
test_nemodatatree_core.py

Description:
This module includes unit tests for the NEMODataTree core
operation class methods.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import pytest


class TestVariableMasking():
    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_grid_error(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Verify KeyError is raised for non-existent grid -- #
        with pytest.raises(KeyError, match="grid '/gridT' not found in available NEMODataTree grids"):
            nemo['/gridT/tos_con']

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_global_mask_2D_var(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Verify equal dims, coords and data values -- #
        assert (nemo['gridT/tos_con']
                .equals(
                    nemo['gridT']['tos_con'].where(nemo['gridT']['tmaskutil'])
                    )
                )

    @pytest.mark.parametrize("dom_type", ["global", "regional"])
    def test_global_mask_3D_var(self, dom_type, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Verify equal dims, coords and data values -- #
        assert (nemo['gridT/thetao_con']
                .equals(
                    nemo['gridT']['thetao_con'].where(nemo['gridT']['tmask'])
                    )
                )

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

        # -- Meridional grid cell face area -- #
        data = (
            nemo[grid][f"e3{grid_suffix}"].where(nemo[grid][f"{grid_suffix}mask"])
            * nemo[grid][f"e1{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        areacello = nemo.cell_area(grid=grid, dim='j')

        # -- Horizontal grid cell area -- #
        data = (
            nemo[grid][f"e1{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            * nemo[grid][f"e2{grid_suffix}"].where(nemo[grid][f"{grid_suffix}maskutil"])
            )
        areacello = nemo.cell_area(grid=grid, dim='k')

        # -- Verify equal dims, coords and data values -- #
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

class TestDepthIntegral():
    @pytest.mark.parametrize("limits", [[0, 100], "0, 100", {"lower": 0, "upper": 100}])
    def test_limits_type(self, limits, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid limit type -- #
        with pytest.raises(TypeError, match="depth limits of integration should be given by a tuple"):
            example_global_nemodatatree.depth_integral(grid="gridT", var="thetao_con", limits=limits)

    def test_limits_size(self, example_global_nemodatatree):
        # -- Verify TypeError is raised for invalid limit size -- #
        with pytest.raises(TypeError, match="depth limits of integration should be given by a tuple"):
            example_global_nemodatatree.depth_integral(grid="gridT", var="thetao_con", limits=(0, 100, 100))

    @pytest.mark.parametrize("limits", [(-1, 5), (0, -1), (-5, -1)])
    def test_limits_value(self, limits, example_global_nemodatatree):
        # -- Verify ValueError is raised for negative limits -- #
        with pytest.raises(ValueError, match="depth limits of integration must be non-negative"):
            example_global_nemodatatree.depth_integral(grid="gridT", var='thetao_con', limits=limits)

    def test_limits_order(self, example_global_nemodatatree):
        # -- Verify ValueError is raised for lower limit >= upper limit -- #
        with pytest.raises(ValueError, match="lower depth limit must be less than upper depth limit"):
            example_global_nemodatatree.depth_integral(grid="gridT", var='thetao_con', limits=(5, 2))

    @pytest.mark.parametrize(
            "dom_type, limits",
            [("global", (0, 100)), ("regional", (0, 100)),
             ("global", (25, 125)), ("regional", (25, 125))
             ])
    def test_depth_integral(self, dom_type, limits, example_global_nemodatatree, example_regional_nemodatatree):
        # -- Select NEMODataTree based on domain type -- #
        match dom_type:
            case "regional":
                nemo = example_regional_nemodatatree
            case "global":
                nemo = example_global_nemodatatree
            case _:
                raise ValueError("dom_type must be 'global' or 'regional'")

        # -- Depth integration -- #
        data = (nemo['gridT/thetao_con']
                .isel(k=0)
                .drop_vars(['k', 'deptht'])
                )
        data = (limits[1] - limits[0]) * data

        integral = nemo.depth_integral(grid='gridT', var='thetao_con', limits=limits)

        # -- Verify equal dims, coords and data values -- #
        assert integral.name == "thetao_con_integral"
        assert integral.equals(data)

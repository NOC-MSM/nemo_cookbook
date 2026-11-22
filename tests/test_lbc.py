"""
test_lbc.py

Description:
This module includes unit tests for the lateral boundary condition functions
in lbc.py, verifying _apply_lbc_nfd() against a direct (vectorised) transcription
of the NEMO subroutine lbc_nfd (src/OCE/LBC/lbc_nfd_generic.h90, NEMO 5.0).

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import dask
import numpy as np
import pytest
import xarray as xr

from nemo_cookbook.lbc import _apply_lbc_nfd
from nemo_cookbook.masks import create_dom_mask


# ------------------------------
# NEMO reference implementation
# ------------------------------
# Vectorised transcription of NEMO lbc_nfd (lbc_nfd_generic.h90, NEMO 5.0),
# formerly nemo_cookbook.lbc._lbc_nfd. Operates in-place on a numpy array of
# shape (n, ipj, ipi) including halo points, using Fortran (1-based) indexes.
def _nemo_lbc_nfd(
    ptab: np.ndarray,
    c_NFtype: str,
    cd_nat: str,
    ihls: int,
    psgn: int,
    ipi: int,
    ipj: int,
    Ni0glo: int,
) -> np.ndarray:
    """
    Apply NEMO North Fold (NFD) Lateral Boundary Condition to a DataArray.

    This is a Python vectorised implementation of the original NEMO subroutine:
    /nemo/src/OCE/LBC/lbc_nfd_generic.h90

    Parameters
    ----------
    ptab : np.ndarray
        Array on which apply the boundary condition.
    c_NFtype : str
        Type of North Fold condition to apply.
        Options are "T" for T-point pivot or "F" for F-point pivot.
    cd_nat : str
        Nature of array grid-points.
        Options are "T", "W", "U", "V" or "F".
    ihls : int
        Number of halo points.
    psgn : int
        Sign used across the north fold boundary.
        Options are 1 for positive and -1 for negative.
    ipi : int
        Size of the i-dimension of the array.
    ipj : int
        Size of the j-dimension of the array.
    Ni0glo : int
        Number of interior points in the i-dimension (excluding halos).

    Returns
    -------
    np.ndarray
        Array with north fold boundary condition applied.
    """
    # === Apply north fold boundary condition === #
    # North fold  T-point pivot:
    if c_NFtype == "T":
        match cd_nat:
            case "T" | "W":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 2 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ihls+1:
                    ii1 = ihls + 1
                    ii2 = ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls+2 to ipi - ihls:
                    ii1 = 2 + ihls + np.arange(1, Ni0glo) - 1
                    ii2 = ipi - ihls - np.arange(1, Ni0glo) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ipi - ihls + 1:
                    ii1 = ipi - ihls + 1
                    ii2 = ihls + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls-1 points:
                    ii1 = ipi - ihls + 1 + np.arange(1, ihls)
                    ii2 = ipi - ihls + 1 - np.arange(1, ihls)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # -- Line number ipj - ihls : right half --- #
                # Equivalent to [DO jj = 1, 1]:
                ij1 = ipj - ihls
                ij2 = ij1  # same line

                # Points from ipi/2+2 to ipi - ihls:
                ii1 = ipi // 2 + np.arange(1, Ni0glo // 2) + 1
                ii2 = ipi // 2 - np.arange(1, Ni0glo // 2) + 1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # First ihls points: redo just in case:
                ii1 = np.arange(1, ihls + 1)
                ii2 = 2 * ihls + 2 - ii1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points (DO ji = 1, ihls):
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # --- Line number ipj - ihls : right half --- #
                # Equivalent to [DO jj = 1, 1]:
                ij1 = ipj - ihls
                ij2 = ij1  # same line

                # Points from ipi/2+1 to ipi - ihls  (DO ji = 1, Ni0glo/2):
                ii1 = ipi // 2 + np.arange(1, Ni0glo // 2 + 1)
                ii2 = ipi // 2 - np.arange(1, Ni0glo // 2 + 1) + 1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # First ihls points: redo them just in case:
                ii1 = np.arange(1, ihls + 1)
                ii2 = 2 * ihls + 1 - ii1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "V":
                # -- Last ihls+1 lines (from ipj to ipj-ihls) : full -- #
                for jj in range(1, ihls + 2):  # Fortran: DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 2 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ihls+1:
                    ii1 = ihls + 1
                    ii2 = ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls+2 to ipi - ihls  (Fortran: DO ji = 1, Ni0glo - 1):
                    ii1 = 2 + ihls + np.arange(1, Ni0glo) - 1
                    ii2 = ipi - ihls - np.arange(1, Ni0glo) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # IF( ihls > 0 ) THEN ; DO ji = 1, COUNT( (/ihls > 0/) ) -> one iteration if ihls>0:
                    if ihls > 0:
                        ii1 = ipi - ihls + 1
                        ii2 = ihls + 1
                        ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls-1 points:
                    ii1 = ipi - ihls + 1 + np.arange(1, ihls)
                    ii2 = ipi - ihls + 1 - np.arange(1, ihls)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "F":
                # -- Last ihls+1 lines (from ipj to ipj - ihls) : full -- #
                for jj in range(1, ihls + 2):  # DO jj = 1, ihls+1
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 2

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls to ipi - ihls  (DO ji = 1, Ni0glo):
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points (DO ji = 1, ihls):
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

    # North fold  F-point pivot:
    elif c_NFtype == "F":
        match cd_nat:
            case "T" | "W":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj + 1 - jj
                    ij2 = ipj - 2 * ihls + jj

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls to ipi - ihls:
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "U":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj + 1 - jj
                    ij2 = ipj - 2 * ihls + jj

                    # First ihls-1 points:
                    ii1 = np.arange(1, ihls)
                    ii2 = 2 * ihls - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ihls (ihls > 0):
                    ii1 = ihls
                    ii2 = ipi - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls+1 to ipi - ihls - 1:
                    ii1 = ihls + np.arange(1, Ni0glo)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ipi - ihls:
                    ii1 = ipi - ihls
                    ii2 = ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "V":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls points:
                    ii1 = np.arange(1, ihls + 1)
                    ii2 = 2 * ihls + 1 - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls to ipi - ihls:
                    ii1 = ihls + np.arange(1, Ni0glo + 1)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points (NEMO 5.0: ii2 = ipi - ihls - ji + 1):
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1) + 1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # -- Line number ipj - ihls : right half -- #
                # Equivalent to [DO jj = 1, 1]:
                ij1 = ipj - ihls
                ij2 = ij1  # same line

                # Points from ipi/2+1 to ipi - ihls:
                ii1 = ipi // 2 + np.arange(1, Ni0glo // 2 + 1)
                ii2 = ipi // 2 - np.arange(1, Ni0glo // 2 + 1) + 1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # First ihls points: redo them just in case:
                ii1 = np.arange(1, ihls + 1)
                ii2 = 2 * ihls + 1 - ii1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

            case "F":
                # -- Last ihls lines (from ipj to ipj - ihls + 1) : full -- #
                for jj in range(1, ihls + 1):  # DO jj = 1, ihls
                    ij1 = ipj - jj + 1
                    ij2 = ipj - 2 * ihls + jj - 1

                    # First ihls - 1 points:
                    ii1 = np.arange(1, ihls)
                    ii2 = 2 * ihls - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ihls:
                    ii1 = ihls
                    ii2 = ipi - ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Points from ihls+1 to ipi - ihls - 1:
                    ii1 = ihls + np.arange(1, Ni0glo)
                    ii2 = ipi - ihls - np.arange(1, Ni0glo)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Point ipi - ihls:
                    ii1 = ipi - ihls
                    ii2 = ii1
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                    # Last ihls points:
                    ii1 = ipi - ihls + np.arange(1, ihls + 1)
                    ii2 = ipi - ihls - np.arange(1, ihls + 1)
                    ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # -- Line number ipj - ihls : right half -- #
                # Equivalent to [DO jj = 1, 1]:
                ij1 = ipj - ihls
                ij2 = ij1  # same line

                # Points from ipi/2+1 to ipi - ihls - 1:
                ii1 = ipi // 2 + np.arange(1, Ni0glo // 2)
                ii2 = ipi // 2 - np.arange(1, Ni0glo // 2)
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

                # First ihls - 1 points: redo them just in case:
                ii1 = np.arange(1, ihls)
                ii2 = 2 * ihls - ii1
                ptab[:, ij1 - 1, ii1 - 1] = psgn * ptab[:, ij2 - 1, ii2 - 1]

    return ptab


# ----------------------------
# Utility functions
# ----------------------------
def _make_dataarray(cd_nat: str, nj: int = 10, ni: int = 12, seed: int = 0) -> xr.DataArray:
    """
    Create an integer-valued DataArray on an idealised (j, i) NEMO model grid.

    The 2-D coordinate `pos` encodes the (j, i) position of each grid point.
    U/F-points are located at i + 1/2 and V/F-points at j + 1/2.
    """
    rng = np.random.default_rng(seed)
    j_offset = 0.5 if cd_nat in ("V", "F") else 0
    i_offset = 0.5 if cd_nat in ("U", "F") else 0
    pos = 1000 * np.arange(1, nj + 1)[:, None] + np.arange(1, ni + 1)[None, :]

    return xr.DataArray(
        data=rng.integers(1, 100, (nj, ni)).astype(float),
        dims=("j", "i"),
        coords={
            "j": np.arange(1, nj + 1) + j_offset,
            "i": np.arange(1, ni + 1) + i_offset,
            "pos": (("j", "i"), pos),
        },
    )


def _nemo_reference(values: np.ndarray, c_NFtype: str, cd_nat: str, psgn: int, ihls: int) -> np.ndarray:
    """
    Expected output of _apply_lbc_nfd() using the NEMO lbc_nfd transcription.

    Interior values (..., Nj, Ni) are padded with periodic halo columns and
    `ihls` northern halo rows before applying _nemo_lbc_nfd(). The last interior
    row (Nj), which NEMO may overwrite, is restored since _apply_lbc_nfd() does
    not modify interior values.
    """
    nj, ni = values.shape[-2:]
    pad_width = [(0, 0)] * (values.ndim - 2) + [(0, ihls), (ihls, ihls)]
    full = np.pad(values, pad_width, mode="wrap")
    full[..., -ihls:, :] = 0
    last_interior_row = full[..., nj - 1, :].copy()

    ptab = full.reshape(-1, nj + ihls, ni + 2 * ihls).copy()
    ptab = _nemo_lbc_nfd(ptab, c_NFtype, cd_nat, ihls, psgn, ni + 2 * ihls, nj + ihls, ni)
    expected = ptab.reshape(full.shape)
    expected[..., nj - 1, :] = last_interior_row

    return expected


# ----------------------------
# _apply_lbc_nfd()
# ----------------------------
@pytest.mark.parametrize("c_NFtype", ["T", "F"])
@pytest.mark.parametrize("cd_nat", ["T", "W", "U", "V", "F"])
@pytest.mark.parametrize("ihls", [1, 2])
@pytest.mark.parametrize("psgn", [1, -1])
def test_apply_lbc_nfd_matches_nemo(c_NFtype, cd_nat, ihls, psgn):
    """Halo points match NEMO lbc_nfd for all north fold & grid point types."""
    da = _make_dataarray(cd_nat)
    result = _apply_lbc_nfd(da, c_NFtype=c_NFtype, cd_nat=cd_nat, psgn=psgn, ihls=ihls)
    expected = _nemo_reference(da.values, c_NFtype, cd_nat, psgn, ihls)

    assert result.shape == (da.sizes["j"] + ihls, da.sizes["i"] + 2 * ihls)
    # Exact comparison: north fold LBC only copies (& changes sign of) values.
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("c_NFtype", ["T", "F"])
@pytest.mark.parametrize("cd_nat", ["T", "W", "U", "V", "F"])
def test_apply_lbc_nfd_interior_unchanged(c_NFtype, cd_nat):
    """Interior values & coordinates are not modified."""
    ihls = 2
    da = _make_dataarray(cd_nat)
    result = _apply_lbc_nfd(da, c_NFtype=c_NFtype, cd_nat=cd_nat, psgn=-1, ihls=ihls)

    interior = result.isel(j=slice(None, da.sizes["j"]), i=slice(ihls, -ihls))
    xr.testing.assert_identical(interior, da)


@pytest.mark.parametrize("c_NFtype", ["T", "F"])
@pytest.mark.parametrize("cd_nat", ["T", "W", "U", "V", "F"])
def test_apply_lbc_nfd_coords(c_NFtype, cd_nat):
    """Halo coordinates extend the grid & 2-D coordinates are folded with the data."""
    ihls = 2
    da = _make_dataarray(cd_nat)
    # Set data equal to 2-D position coordinate:
    da = da.copy(data=da["pos"].values.astype(float))
    result = _apply_lbc_nfd(da, c_NFtype=c_NFtype, cd_nat=cd_nat, psgn=1, ihls=ihls)

    j, i = da["j"].values, da["i"].values
    np.testing.assert_array_equal(
        result["j"].values, np.concatenate([j, j[-1] + np.arange(1, ihls + 1)])
    )
    np.testing.assert_array_equal(
        result["i"].values, np.concatenate([i[:ihls] - ihls, i, i[-ihls:] + ihls])
    )
    np.testing.assert_array_equal(result["pos"].values, result.values)


@pytest.mark.parametrize("c_NFtype", ["T", "F"])
@pytest.mark.parametrize("cd_nat", ["T", "W", "U", "V", "F"])
def test_apply_lbc_nfd_bool_mask(c_NFtype, cd_nat):
    """Boolean masks retain their dtype with psgn = 1."""
    da = _make_dataarray(cd_nat)
    result = _apply_lbc_nfd(da > 50, c_NFtype=c_NFtype, cd_nat=cd_nat, psgn=1)

    assert result.dtype == bool
    np.testing.assert_array_equal(
        result.values, _apply_lbc_nfd(da, c_NFtype=c_NFtype, cd_nat=cd_nat, psgn=1).values > 50
    )


def test_apply_lbc_nfd_lazy_4d():
    """Dask-backed 4-D DataArrays remain lazy with dimension order preserved."""
    da = _make_dataarray("V")
    values = np.random.default_rng(1).integers(1, 100, (3, 4, *da.shape)).astype(float)
    da_4d = xr.DataArray(
        values, dims=("time_counter", "k", "j", "i"), coords={"j": da["j"], "i": da["i"]}
    ).chunk({"time_counter": 1, "k": 2})

    result = _apply_lbc_nfd(da_4d, c_NFtype="T", cd_nat="V", psgn=-1, ihls=1)

    assert dask.is_dask_collection(result)
    assert result.dims == da_4d.dims
    np.testing.assert_array_equal(result.values, _nemo_reference(values, "T", "V", -1, 1))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"c_NFtype": "X"}, ValueError),
        ({"cd_nat": "Z"}, ValueError),
        ({"psgn": 0}, ValueError),
        ({"ihls": 0}, ValueError),
        ({"ihls": 1.5}, TypeError),
        ({"ihls": 9}, ValueError),
        ({"psgn": -1, "da": _make_dataarray("U") > 50}, ValueError),
        ({"da": _make_dataarray("T").rename({"i": "x"})}, ValueError),
        ({"da": _make_dataarray("T").isel(i=slice(None, -1))}, ValueError),
        ({"da": _make_dataarray("T").values}, TypeError),
    ],
)
def test_apply_lbc_nfd_invalid_inputs(kwargs, error):
    """Invalid inputs raise errors."""
    args = {"da": _make_dataarray("T"), "c_NFtype": "T", "cd_nat": "T", "psgn": 1, "ihls": 1}
    with pytest.raises(error):
        _apply_lbc_nfd(**(args | kwargs))


# ----------------------------
# create_dom_mask()
# ----------------------------
@pytest.mark.parametrize("c_NFtype", ["T", "F"])
@pytest.mark.parametrize("cd_nat", ["T", "U", "V", "W", "F"])
def test_create_dom_mask_north_fold(c_NFtype, cd_nat):
    """
    Land-sea masks of a north folding domain match the NEMO dommsk.F90 formulae
    applied to tmask with halos defined by NEMO lbc_nfd.
    """
    nk, nj, ni = 6, 12, 16
    ka = np.arange(nk)

    # Random bathymetry satisfying the north fold condition along row Nj:
    bottom = np.random.default_rng(42).integers(0, nk + 1, (nj, ni)).astype(float)
    bottom = np.pad(bottom, ((0, 1), (1, 1)), mode="wrap")[None]
    bottom = _nemo_lbc_nfd(bottom, c_NFtype, "T", 1, 1, ni + 2, nj + 1, ni)[0, :nj, 1:-1]
    top = (bottom > 0).astype(float)

    # Expected masks (NEMO dommsk.F90) using tmask with 1 halo (NEMO lbc_nfd):
    tmask = ((ka[:, None, None] + 1) >= top) & ((ka[:, None, None] + 1) <= bottom)
    t = np.pad(tmask.astype(float), ((0, 0), (0, 1), (1, 1)), mode="wrap")
    t[:, -1, :] = 0
    t = _nemo_lbc_nfd(t, c_NFtype, "T", 1, 1, ni + 2, nj + 1, ni)
    expected = {
        "T": tmask,
        "U": t[:, :nj, 1:-1] * t[:, :nj, 2:],
        "V": t[:, :nj, 1:-1] * t[:, 1:, 1:-1],
        "W": np.concatenate([tmask[:1], tmask[:-1] & tmask[1:]]),
        "F": t[:, :nj, 1:-1] * t[:, :nj, 2:] * t[:, 1:, 1:-1] * t[:, 1:, 2:],
    }

    coords = {"y": np.arange(nj), "x": np.arange(ni)}
    mask = create_dom_mask(
        ka=xr.DataArray(ka, dims="nav_lev"),
        top_level=xr.DataArray(top, dims=("y", "x"), coords=coords).chunk(),
        bottom_level=xr.DataArray(bottom, dims=("y", "x"), coords=coords).chunk(),
        cd_nat=cd_nat,
        c_NFtype=c_NFtype,
        iperio=True,
    )

    assert dask.is_dask_collection(mask)
    assert mask.dims == ("nav_lev", "y", "x")
    np.testing.assert_array_equal(mask.values, expected[cd_nat].astype(bool))

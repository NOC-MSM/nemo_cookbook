"""
test_virtualize.py

Description:
This module includes unit tests for virtualize utility functions.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import re
from types import SimpleNamespace

import pytest
import xarray as xr

from nemo_cookbook.utils import PathsDict
from nemo_cookbook.virtualize import (
    _create_virtual_dataset,
    _get_invalid_vars,
    _resolve_filepaths,
    create_virtual_dataset_dict,
)


class FakeH5Dataset:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeH5File:
    def __init__(self, mapping):
        self.mapping = mapping

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return self.mapping.keys()

    def __getitem__(self, key):
        return self.mapping[key]

class TestResolveFilepaths:
    def test_prefix_mismatch_raises_value_error(self):
        # -- Verify ValueError when prefix not found in the pattern -- #
        with pytest.raises(ValueError, match=re.escape("prefix '/prefix' must be found")):
            _resolve_filepaths(pattern="/other/domain.nc",
                               prefix="/prefix",
                               label="parent/domain",
                               )

    def test_empty_glob_raises_value_error(self, mocker):
        # -- Verify ValueError when no filepaths match pattern -- #
        mocker.patch("nemo_cookbook.virtualize.glob.glob", return_value=[])

        with pytest.raises(ValueError, match="No files matched pattern"):
            _resolve_filepaths(pattern="/prefix/domain*.nc",
                               prefix="/prefix",
                               label="parent/domain",
                               )

class TestGetInvalidVars:
    def test_identifies_invalid_vars(self, mocker):
        # -- Verify that invalid variables are correctly identified with warning -- #
        fake_h5py = SimpleNamespace(
            File=lambda **_kwargs: FakeH5File(
                mapping={
                    "good": FakeH5Dataset(name="/good", shape=(3, 2)),
                    "bad": FakeH5Dataset(name="/bad", shape=(0, 2)),
                }
            )
        )
        mocker.patch("nemo_cookbook.virtualize.importlib.import_module",
                     return_value=fake_h5py,
                     )
        with pytest.warns(UserWarning, match="Dropping invalid variables"):
            result = _get_invalid_vars(filepath="/prefix/file.nc")
        assert result == ["bad"]

    def test_missing_optional_dependencies(self, mocker):
        # -- Verify ImportError when optional dependencies are missing -- #
        mocker.patch(
            "importlib.import_module",
            side_effect=ImportError("missing dependency"),
        )
        with pytest.raises(ImportError, match=re.escape("virtualize_from_paths() requires optional virtualization dependencies")):
            _get_invalid_vars(filepath="/prefix/file.nc")

class TestCreateVirtualDataset:
    def test_missing_optional_dependencies(self, mocker):
        # -- Verify ImportError when optional dependencies are missing -- #
        mocker.patch(
            "importlib.import_module",
            side_effect=ImportError("missing dependency"),
        )
        with pytest.raises(ImportError, match=re.escape("virtualize_from_paths() requires optional virtualization dependencies")):
            _create_virtual_dataset(
                prefix="/prefix",
                filepaths=["/prefix/file.nc"],
            )

class TestCreateVirtualDatasetDict:
    @pytest.mark.parametrize("paths", [
        {"bad_domain": {"domain": "/prefix/bad_domain.nc"}},
        {"bad_domain": "/prefix/bad_domain.nc"}
    ])
    def test_paths_structure_value_error(self, paths):
        # -- Verify ValueError is raised when child exists without nest dict -- #
        with pytest.raises(ValueError, match=re.escape("Invalid `paths` structure. Expected a nested dictionary defining NEMO 'parent', 'child' and 'grandchild' domains.")):
            create_virtual_dataset_dict(prefix="/prefix", paths=paths)

    def test_child_without_nests_value_error(self, mocker):
        # -- Verify ValueError is raised when child exists without nest dict -- #
        paths: PathsDict = {
            "parent": {"domain": "/prefix/parent_domain.nc"},
            "child": {"1": {"domain": "/prefix/child_domain.nc"}},
        }

        mocker.patch("nemo_cookbook.virtualize._resolve_filepaths",
                     return_value=["/prefix/mock.nc"],
                     )
        mocker.patch("nemo_cookbook.virtualize._get_invalid_vars",
                     return_value=[],
                     )
        mocker.patch("nemo_cookbook.virtualize._create_virtual_dataset",
                     return_value=xr.Dataset(),
                     )

        with pytest.raises(ValueError, match=re.escape("`nests` dictionary must be provided when defining NEMO child domains.")):
            create_virtual_dataset_dict(prefix="/prefix", paths=paths)

    def test_grandchild_without_child_value_error(self, mocker):
        # -- Verify ValueError is raised when grandchild exists without child -- #
        paths: PathsDict = {
            "parent": {"domain": "/prefix/parent_domain.nc"},
            "grandchild": {"2": {"domain": "/prefix/grandchild_domain.nc"}},
        }

        mocker.patch("nemo_cookbook.virtualize._resolve_filepaths",
                     return_value=["/prefix/mock.nc"],
                     )
        mocker.patch("nemo_cookbook.virtualize._get_invalid_vars",
                     return_value=[],
                     )
        mocker.patch("nemo_cookbook.virtualize._create_virtual_dataset",
                     return_value=xr.Dataset(),
                     )

        with pytest.raises(ValueError, match=re.escape("`child_paths` must be defined when defining NEMO grandchild domains.")):
            create_virtual_dataset_dict(prefix="/prefix", paths=paths, nests={})

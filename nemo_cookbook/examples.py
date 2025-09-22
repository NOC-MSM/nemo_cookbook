"""
examples.py

Description:
This module includes functions to retrieve and cache
example NEMO model output data for nemo_cookbook
demonstrators.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import os
import pooch
import importlib.resources as ir

def _create_pooch_manager() -> pooch.Pooch:
    """
    Create and return a pooch data manager for handling example
    NEMO model output files.

    Returns:
    --------
    pooch.Pooch
        pooch data manager configured for example NEMO model output files.
    """
    pooch_manager = pooch.create(
        # Use the default cache folder for the operating system:
        path=pooch.os_cache("nemo_cookbook"),
        base_url="https://noc-msm-o.s3-ext.jc.rl.ac.uk/nemo-cookbook/example_data/",
        registry=None
    )
    registry_path = str(ir.files("nemo_cookbook") / "registry.txt")
    pooch_manager.load_registry(registry_path)

    return pooch_manager


def get_filepaths(
    example: str
    ) -> str:
    """
    Retrieve filepaths for example outputs for a given
    NEMO model reference configuration.

    Parameters:
    -----------
    example: str
        Name of NEMO model reference configuration to retrieve
        example output filepaths.

    Returns:
    --------
    dict[str, str]
        Dictionary with keys as filenames and values as output filepaths.
    """
    # -- Validate input -- #
    if not isinstance(example, str):
        raise TypeError("`example` must be a string.")
    valid_examples = ["AGRIF_DEMO", "AMM12", "IHO"]
    if example not in valid_examples:
        raise ValueError(
            f"`example` must be one of {valid_examples}, got {example}."
        )
    
    # -- Collect filepath as dictionary -- #
    pooch_data = _create_pooch_manager()
    registry = pooch_data.registry
    filenames = [key for key in registry.keys() if example in key]

    return {fname.split("/")[-1]: pooch_data.fetch(fname) for fname in filenames}


def _create_pooch_registry():
    """
    Create a pooch registry file for the example NEMO model output data.

    This function should be used to create or update the `registry.txt`
    file used by the pooch data manager. It downloads all example data
    files and computes their hashes for integrity checks.

    Returns:
    --------
    None
    """
    # Define NEMO reference configuration output filenames and URLs:
    base_url = "https://noc-msm-o.s3-ext.jc.rl.ac.uk/nemo-cookbook/example_data/"
    fnames_and_urls = {
        "AGRIF_DEMO/domain_cfg.nc": f"{base_url}AGRIF_DEMO/domain_cfg.nc",
        "AGRIF_DEMO/2_domain_cfg.nc": f"{base_url}AGRIF_DEMO/2_domain_cfg.nc",
        "AGRIF_DEMO/3_domain_cfg.nc": f"{base_url}AGRIF_DEMO/3_domain_cfg.nc",
        "AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_T.nc": f"{base_url}AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_T.nc",
        "AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_U.nc": f"{base_url}AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_U.nc",
        "AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_V.nc": f"{base_url}AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_V.nc",
        "AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_W.nc": f"{base_url}AGRIF_DEMO/ORCA2_5d_00010101_00010110_grid_W.nc",
        "AGRIF_DEMO/ORCA2_5d_00010101_00010110_icemod.nc": f"{base_url}AGRIF_DEMO/ORCA2_5d_00010101_00010110_icemod.nc",
        "AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_T.nc": f"{base_url}AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_T.nc",
        "AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_U.nc": f"{base_url}AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_U.nc",
        "AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_V.nc": f"{base_url}AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_V.nc",
        "AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_W.nc": f"{base_url}AGRIF_DEMO/2_Nordic_5d_00010101_00010110_grid_W.nc",
        "AGRIF_DEMO/2_Nordic_5d_00010101_00010110_icemod.nc": f"{base_url}AGRIF_DEMO/2_Nordic_5d_00010101_00010110_icemod.nc",
        "AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_T.nc": f"{base_url}AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_T.nc",
        "AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_U.nc": f"{base_url}AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_U.nc",
        "AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_V.nc": f"{base_url}AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_V.nc",
        "AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_W.nc": f"{base_url}AGRIF_DEMO/3_Nordic_5d_00010101_00010110_grid_W.nc",
        "AGRIF_DEMO/3_Nordic_5d_00010101_00010110_icemod.nc": f"{base_url}AGRIF_DEMO/3_Nordic_5d_00010101_00010110_icemod.nc",
        "AMM12/domain_cfg.nc": f"{base_url}AMM12/domain_cfg.nc",
        "AMM12/AMM12_1d_20120102_20120110_grid_T.nc": f"{base_url}AMM12/AMM12_1d_20120102_20120110_grid_T.nc",
        "AMM12/AMM12_1d_20120102_20120110_grid_U.nc": f"{base_url}AMM12/AMM12_1d_20120102_20120110_grid_U.nc",
        "AMM12/AMM12_1d_20120102_20120110_grid_V.nc": f"{base_url}AMM12/AMM12_1d_20120102_20120110_grid_V.nc",
        "IHO/IHO_World_Seas_v3_polygons.parquet": f"{base_url}IHO/IHO_World_Seas_v3_polygons.parquet"
    }

    # Create a new directory to download output files:
    directory = "example_data"
    os.makedirs(directory, exist_ok=True)

    # Create a new registry file:
    with open("registry.txt", "w") as registry:
        for fname, url in fnames_and_urls.items():
            # Download each data file to example_data/:
            path = pooch.retrieve(
                url=url, known_hash=None, fname=fname, path=directory
            )
            # Append the filename, hash, and URL to a new registry file:
            registry.write(
                f"{fname} {pooch.file_hash(path)} {url}\n"
            )

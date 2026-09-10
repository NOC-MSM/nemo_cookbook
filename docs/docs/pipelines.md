# NEMO Pipeline

[**NEMO Pipeline**](https://github.com/NOC-MSM/nemo_pipeline) is a sibling library to **NEMO Cookbook** dedicated to building & executing data pipelines to calculate diagnostics using NEMO ocean model outputs at scale.

## What is a Data Pipeline?

Data pipelines are automated systems which open raw input data, perform one or more calculations / transformations and write the results to an output file.

**NEMO Pipeline** enables users to create scalable diagnostic pipelines for ocean analysis or model validation using these three steps.

* **Input** --> Opening a collection of NEMO NetCDF output files as one or more `xarray.Datasets` to create a NEMODataTree object introduced in [**NEMO Cookbook**](https://github.com/NOC-MSM/nemo_cookbook/).

* **Diagnostics** --> Use the NEMODataTree and NEMODataArray objects to calculate a diagnostic, such as masked statistics or extracting hydrographic sections.

* **Output** --> Write the diagnostics stored in a CF-compliant `xarray.Dataset` to a local file (NetCDF or Zarr).

## Getting Started

### Installation

To get started, clone the **NEMO Pipeline** repository to your local machine:

```bash
git clone git@github.com:NOC-MSM/nemo_pipeline.git
```

Next, install **NEMO Pipeline** into a new Python virtual environment in editable mode by using `pip` in your local copy of the repository:

```bash
cd nemo_pipeline

pip install -e .
```

## Usage

### NEMO Pipeline CLI

The **NEMO Pipeline** library allows users to create scalable pipelines to produce diagnostics from NEMO ocean model outputs using a **command line interface (CLI)**.

All diagnostic pipelines are defined using a config `.toml` file.

There are three commands available to users:

* `describe` --> Summarise & validate the steps of a pipeline (analogous to a dry-run).

* `run` --> Run pipeline in the current Python process.

* `submit` --> Submit pipeline to be executed via a single SLURM job or SLURM job array.

Each command can be used with the following syntax:

```bash
nemo_pipeline COMMAND [ARGS]
```

where `COMMAND` corresponds to the chosen command and `[ARGS]` represents the required and optional arguments used to define a pipeline. See below for details.

### Config `.toml` Files

To define a new **NEMO Pipeline**, we must populate a config `.toml` file which is structured into four tables as follows:

* `[slurm]` --> Define parameters used to create single SLURM job or SLURM job array script for submission.

* `[inputs]` --> Define filepaths to NEMO ocean model domain and grid variables to construct NEMODataTree.

* `[diagnostics]` --> Define diagnostic function to compute diagnostic from NEMODataTree and return a CF-compliant `xarray.Dataset`.

* `[outputs]` --> Define output filepaths, naming conventions and chunk sizes.

#### Example

Below we present an example config `.toml` file to extract the Overturning in the Subpolar North Atlantic Program (OSNAP) array from a typical NEMO model output dataset...

* The `[slurm]` table is populated to divide the pipeline into 10 tasks (1 per year, 1990-1999 using `ip_start/end/step`), which will be submitted as a job array with a maximum of 2 tasks being executed concurrently (`max_concurrent_jobs`).

* Each task in the SLURM job array will be executed on a single CPU (`ntasks`), which is assigned 6GB of memory (`mem`) and a 10 minute run time limit (`time`). If any SLURM job fails, the job array will be cancelled (`kill_on_fail`).

* We use `venv_cmd` to define a bash command to activate the Python virtual environment used to execute NEMO Pipeline.

```toml
[slurm]
# Define directories for SLURM job scripts and logs:
job_dir = "/my/hpc/nemo_pipeline/examples/outputs/jobs"
log_dir = "/my/hpc/nemo_pipeline/examples/outputs/logs"
# Define log file prefix for SLURM job outputs:
log_prefix = "eORCA1_ERA5_OSNAP"

    [slurm.sbatch]
    # SLURM SBATCH submission directives:
    job_name = "nemo_pipeline_osnap"
    time = "00:10:00"
    partition = "standard"
    ntasks = 1
    mem = "6G"
    # Optional SLURM SBATCH submission directives:
    kwargs = { qos = "standard", account = "my_account" }

    [slurm.jobs]
    # Define Python virtual environment activation command:
    venv_cmd = "source ~/miniforge3/bin/activate; conda activate env_nemo"
    # Define SLURM job type ("array" or "single"):
    job_type = "array"

    # Define the initial, final and step input patterns {ip} for SLURM job arrays:
    ip_start = 1990
    ip_end = 1999
    ip_step = 1
    # Define maximum number of concurrent jobs for SLURM job arrays:
    max_concurrent_jobs = 2
    # Kill SLURM job array in the event of any job failure:
    kill_on_fail = true
```

* In the `[inputs]` table, we define the path to the directory containing our NEMO ocean model outputs and domain_cfg NetCDF files.

* To construct a NEMODataTree from CMORISED NEMO model outputs (i.e., variables must be merged into grid datasets from separate NetCDF files), we can use `cmorised = true`. We also specify the `iperio`, `nftype` and `read_mask` arguments that will be passed to the `NEMODataTree.from_datasets()` constructor.

* In each of the `[inputs]` sub-tables (i.e., `[inputs.gridT]`), we specify the filepath or list of filepaths to our NEMO model output files, alongside a list of variables and chunk-sizes to be used.

* Note that both `${nemo_dir}` and `{ip}` will be substituted for the NEMO output directory path (`nemo_dir`) and the input pattern (--input-pattern) during execution. In the case of a SLURM job array, the input pattern will be determined automatically from the `ip_start/end/step` arguments included in the `[slurm.jobs]` sub-table.  

```toml
[inputs]
# Define NEMO output directory and domain_cfg filepath to construct NEMODataTree:
nemo_dir = "/dssgfs01/scratch/npd/simulations/eORCA1_ERA5_v1"
domain_filepath = "/dssgfs01/scratch/npd/simulations/Domains/eORCA1/domain_cfg.nc"
# Create NEMO model grid datasets from CMORISED variables:
cmorised = false
# Input arguments to construct NEMODataTree:
iperio = true
nftype = "T"
read_mask = false

    [inputs.gridT]
    # -- NEMO gridT (scalar) variables -- #
    # Define filepath or list of filepaths with optional {ip} pattern to be substituted using --input-pattern argument:
    filepath = "${nemo_dir}/eORCA1_ERA5_1m_grid_T_{ip}*.nc"
    # Define list of variable names or list of list of variable names to be read from each file:
    vars = [ "thetao_con", "so_abs" ]
    # Define dictionary of chunks for all gridT variables:
    chunks = { k = 75 }

    [inputs.gridU]
    # -- NEMO gridU (zonal vector) variables -- #
    filepath = "${nemo_dir}/eORCA1_ERA5_1m_grid_U_{ip}*.nc"
    vars = [ "uo", "uo_eiv", "e3u" ]
    chunks = { k = 75 }

    [inputs.gridV]
    # -- NEMO gridV (meridional vector) variables -- #
    filepath = "${nemo_dir}/eORCA1_ERA5_1m_grid_V_{ip}*.nc"
    vars = [ "vo", "vo_eiv", "e3v" ]
    chunks = { k = 75 }

    [inputs.gridW]
    # -- NEMO gridW (vertical vector) variables -- #

    [inputs.gridF]
    # -- NEMO gridF (vorticity vector) variables -- #

    [inputs.icemod]
    # -- NEMO icemod (sea-ice) variables -- #
```

* In the `[diagnotics]` table, we specify the names of the Python module and diagnostic function as a dictionary, alongside any keyword arguments `kwargs` to be passed to the diagnostic.

```toml
[diagnostics]
# Define diagnostics to be computed using NEMODataTree:
diagnostic = { module = "nemo_pipeline.diagnostics.core", function = "extract_osnap_section" }
# Define keyword arguments to be passed to diagnostic function:
kwargs = { include_eiv = true }
```

* Finally, in the `[outputs]` table, we provide the directory path, date format and file name used to write our diagnostic dataset to a local NetCDF file or Zarr store.

```toml
[outputs]
# Define NEMO ocean model pipeline output file:
output_dir = "/my/hpc/nemo_pipeline/examples/outputs"
output_name = "eORCA1_ERA5_v1_OSNAP"
# Define date format for output files. Options are "Y" (YYYY), "M" (YYYY-MM) or "D" (YYYY-MM-DD):
date_format = "M"
# Define output file format ("NetCDF" or "zarr"):
format = "NetCDF"
# Define output dataset chunking:
chunks = { time_counter = 1, k = 75 }
```

### Running NEMO Pipeline

When executing of a NEMO Pipeline defined in a `.toml` configuration file using the `run` command only a single `.log` file is created.

During the execution of a NEMO Pipeline using SLURM via the `submit` command, the following files and directories are created by default:

```
nemo_pipeline.log
./jobs/
./logs/
./logs/slurm/
```

We can modify each of the paths above by modifying the `job_dir`, `log_dir`, `log_prefix` variables in the `[slurm]` table of our `.toml` configuration file.

* The `jobs/` directory is populated with the SLURM job script submitted to the scheduler, named: `{log_prefix}_nemo_pipeline.slurm`

* The `logs/` directory will be populated with separate `.log` files for each job submitted to the scheduler, meaning one `.log` file per array job or a single `.log` file is produced.

* The `logs/slurm/` directory will be populated with the SLURM `.out` files output for each job submitted to the scheduler.
    * For a single SLURM job, this yields a single output file named `{log_prefix}-%j.out`, where `%j` is replaced with the job ID.
    * For a SLURM array job, this yields one output file per job in the array named `{log_prefix}-%A_%a.out`, where `%A` is replaced by the job ID and `%a` with the array index.

## How To...

**Describe**

* Describe & validate the NEMO Pipeline defined in the `/path/to/config.toml` without performing any calculations and write the summary to `/path/to/pipeline.log`

```bash
nemo_pipeline describe /path/to/config.toml --log /path/to/pipeline.log
```

**Run**

* Run the NEMO Pipeline defined in the `/path/to/config.toml` in the current process. Substitute for the input pattern `{ip}` -> `2010` and write the summary to `/path/to/pipeline.log`. 

```bash
nemo_pipeline describe /path/to/config.toml --log /path/to/pipeline.log -input-pattern "2010"
```

* Run the NEMO Pipeline defined in the `/path/to/config.toml` in the current process. Substitute for the input pattern `{ip}` -> `2010`, `2011`, `2012` and write the summary to `/path/to/pipeline.log`. Note, that a NEMODataTree will be constructed from a list of filepaths produced by separately substituting each of the input patterns included in the list.

```bash
nemo_pipeline describe /path/to/config.toml --log /path/to/pipeline.log -input-pattern "[2010, 2011, 2012]"
```

**Submit**

* Submit the NEMO Pipeline defined in the `/path/to/config.toml` as a single job to the SLURM scheduler. Substitute for the input pattern `{ip}` -> `2010` and write the summary to `/path/to/pipeline.log`. 

```bash
nemo_pipeline submit /path/to/config.toml --log /path/to/pipeline.log -input-pattern 2010
```

* Submit the NEMO Pipeline defined in the `/path/to/config.toml` as an array job to the SLURM scheduler and write the summary to `/path/to/pipeline.log`. 

```bash
nemo_pipeline submit /path/to/config.toml --log /path/to/pipeline.log
```

* Submit the NEMO Pipeline defined in the `/path/to/config.toml` as a single job to the SLURM scheduler dependent upon the successful completion of the existing SLURM job with job ID 12345. Substitute for the input pattern `{ip}` -> `2010` and write the summary to `/path/to/pipeline.log`. 

```bash
nemo_pipeline submit /path/to/config.toml --log /path/to/pipeline.log -input-pattern 2010 --depends-on 12345
```

* Create a SLURM array job script for the NEMO Pipeline defined in the `/path/to/config.toml` without submitting to the SLURM scheduler and write the summary to `/path/to/pipeline.log`. This allows users to review the script before submitting the job using `sbatch` later. 

```bash
nemo_pipeline submit /path/to/config.toml --log /path/to/pipeline.log --no-submit
```

## Reference

### CLI Arguments

| Long version | Optional | Description |
|---|---|---|
| `COMMAND` | **No** | Specify the action: `describe` / `run` / `submit`. |
| `config` | **No** | Path to NEMO pipeline config .toml file |
| `--log` | **Yes** | Path to write NEMO pipeline log file. |
| `--input-pattern` | **Yes** | Pattern used to subsititute `{ip}` in NEMO model input file paths in config file. |
| `--depends-on` | **Yes** | Defer the start of this NEMO Pipeline job until the specified SLURM job ID has completed successfully. |
| `--submit` / `--no-submit` | **Yes** | Submit the job to the SLURM scheduler. |

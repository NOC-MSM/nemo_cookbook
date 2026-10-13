# NEMO Cookbook — Agent Guidelines

This file provides guidance for AI agents (Claude Code, GitHub Copilot, OpenAI Codex, and similar tools) contributing to NEMO Cookbook. Read it before making any changes.

---

## Project Overview

**NEMO Cookbook** is a Python library for reproducible analysis of
[NEMO](https://www.nemo-ocean.eu/) ocean general circulation model outputs using xarray.

The library extends [xarray](https://docs.xarray.dev/) with NEMO-aware data structures and discrete numerical operators (differences, derivatives, integrals, interpolation, vertical transforms) that aim to be consistent with the discretisation used in the NEMO ocean model itself.

Key dependencies: `xarray`, `dask`, `numpy`, `numba`, `cartopy`, `zarr`, `icechunk`.

---

## Repository Layout

```
nemo_cookbook/        # Source package
  nemodatatree.py     # NEMODataTree — hierarchical data structure (one or more model domains)
  nemodataarray.py    # NEMODataArray — grid-aware DataArray with discrete operators
  extract.py          # Hydrographic section / boundary extraction
  integrate.py        # Integration operations (depth, volume, area)
  interpolate.py      # Horizontal grid interpolation
  masks.py            # Land/sea masks, polygon masks, masked statistics
  stats.py            # Grid-aware statistical functions
  transform.py        # Vertical grid transformations
  virtualize.py       # Virtual NEMODataTree construction
  lbc.py              # Lateral boundary conditions
  processing.py       # Internal DataTree construction helpers
  accessors.py        # xarray accessor registrations
  examples.py         # Example dataset file paths (for tests and docs)
  utils.py            # Internal utilities
tests/
  conftest.py         # Shared pytest fixtures (idealised & ORCA2/AMM12 NEMODataTrees)
  test_*.py           # Unit tests (mirror source module names)
recipes/              # Jupyter Notebook recipes (contributed analysis examples)
docs/                 # MkDocs documentation source
  docs/               # Markdown pages and recipe notebooks rendered in the docs
pyproject.toml        # Build metadata, optional extras, ruff config
pixi.toml             # Pixi workspace: environments, tasks
```

---

## Development Environment

NEMO Cookbook uses [Pixi](https://pixi.sh/) for environment and task management.

```bash
# Enter the development environment
pixi shell -e dev

# Or run a task directly without entering the shell
pixi run test
pixi run docs
```

The `dev` environment includes the package in editable mode plus all optional extras.

---

## Common Commands

| Task | Command |
|---|---|
| Run test suite | `pixi run test` |
| Lint (check) | `ruff check nemo_cookbook/ tests/` |
| Format (fix) | `ruff format nemo_cookbook/ tests/` |
| Lint + format (fix) | `ruff check --fix nemo_cookbook/ tests/ && ruff format nemo_cookbook/ tests/` |
| Serve docs locally | `pixi run docs` |

Always run `pixi run test` before marking a task complete. Fix any ruff violations before committing.

---

## Code Conventions

- **Docstrings**: [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
  for all public functions, methods, and classes.
- **Style**: PEP 8; enforced by `ruff` (line length 88, `E`, `F`, `I`, `B` rules).
- **Dask compatibility**: all numerical operations must work lazily. Never call `.compute()`
  inside library code; let the user control when computation occurs.
- **Type hints**: use them on public APIs; Python ≥ 3.11 syntax is fine.
- **Simplicity over cleverness**: prefer clear, readable implementations over compact or highly abstracted ones. A future reader — human or model — should be able to trace the logic without needing to unpack layers of indirection or abstraction.

---

## Physical Correctness

**This is the most important correctness criterion for numerical code.**

All discrete operators, integrals, interpolations, and diagnostics must be consistent with their equivalent implementation in the NEMO ocean model source code ([NEMO GitLab](https://forge.nemo-ocean.eu/nemo/nemo)). When implementing or modifying a diagnostic:

1. Identify the corresponding NEMO Fortran routine (e.g. `diawri.F90`) where applicable.
2. Verify that grid staggering, scale factors (`e1`, `e2`, `e3`), and mask conventions match the NEMO C-grid (Arakawa) discretisation.
3. Document deviations — if a simplification is intentional (e.g. linear free-surface approximation), note it in the docstring with a reference to the NEMO equivalent.

If you are unsure whether a formulation is NEMO-consistent, flag it with a `# TODO: verify against NEMO <routine>` comment rather than silently introducing an inconsistency.

---

## Core Abstractions

### `NEMODataTree`
A subclass of `xarray.DataTree` representing one or more NEMO model domains (parent + optional AGRIF child/grandchild nests). Each child node stores a `gridT/U/V/W/F` dataset containing grid-scale factors, masks, and model variables. Constructed via `NEMODataTree.from_paths(paths, ...)`, `NEMODataTree.from_datasets(datasets, ...)`, `NEMODataTree.from_zarr(...)`, `NEMODataTree.from_icechunk(...)`.

### `NEMODataArray`
A subclass of `xarray.DataArray` that carries NEMO grid metadata (grid point type, domain attributes) and exposes discrete operators (differences, derivatives, integrals, grid interpolation). Created automatically when variables are accessed via a `NEMODataTree`.

---

## Testing Guidelines

- Tests live in `tests/`; filenames mirror the source module they test
  (e.g. `test_nemodatatree.py` ↔ `nemodatatree.py`).
- Shared fixtures for idealised and realistic (ORCA2, AMM12, AGRIF) `NEMODataTree` instances are in `tests/conftest.py` — use these rather than constructing test data inline.
- Each new public function should have at least one unit test covering the expected output and one covering edge cases (e.g. masked arrays, regional vs. global domains).
- Run with `pixi run test`; the suite uses `--maxfail=1` so fix failures before continuing.

---

## GitHub Interaction Guidelines

- **NEVER impersonate the user on GitHub**, always sign off with something like
  "[This is Claude Code on behalf of Jane Doe]".
- Never create issues nor pull requests on the GitHub repository unless
  explicitly instructed.
- Never post "update" messages, progress reports, or explanatory comments on
  GitHub issues/PRs unless specifically instructed.
- When creating commits, always include a co-authorship trailer:
  `Co-authored-by: Claude <noreply@anthropic.com>`.

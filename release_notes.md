# Building & Distributing NEMO Cookbook to TestPyPI + PyPI

### Building the NEMO Cookbook package:

* Install `build` and `twine` packages from PyPI.

* Checkout the latest release (tag):

```bash
# Checkout v2026.06.01 tag:
git checkout v2026.06.01 
```

* **After activating the `release` Python virtual environment, create a local build of the release:**

```bash
python3 -m build
```

* **Verify build integrity:**

```bash
python -m twine check --strict dist/*
```

### Upload Package to TestPyPI:

```bash
python3 -m twine upload --repository testpypi dist/*
```

* When prompted pass your API token, including the `pypi-` prefix.

### Upload Package to PyPI:

```bash
python3 -m twine upload dist/*
```

* When prompted pass your API token, including the `pypi-` prefix.

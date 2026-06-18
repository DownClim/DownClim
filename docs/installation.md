# Installation

`DownClim` is a pure Python package. Its pure Python dependencies are handled
automatically by pip. A few dependencies require compiled native libraries and
are easiest to install via conda.

## From conda (recommended)

Conda handles all dependencies, including the compiled ones:

```bash
conda install -c conda-forge downclim
```

## From pip

```bash
python -m pip install downclim
```

Pip will automatically install all pure Python dependencies (aenum,
esgf-pyclient, geopandas, numpy, pandas, pydantic, rioxarray, shapely, xarray,
xee, zarr, etc.).

A few dependencies have compiled extensions and may require system libraries:
**earthengine-api**, **numba**, **netCDF4**, **xesmf** (via llvmlite). If you
encounter build errors with any of these, install them first with conda:

```bash
conda install -c conda-forge earthengine-api numba netcdf4 xesmf
pip install downclim
```

## Optional dependencies

```bash
# For running tests
python -m pip install "downclim[test]"

# For development
python -m pip install "downclim[dev]"

# For building documentation
python -m pip install "downclim[docs]"
```

## Development version

```bash
git clone https://github.com/DownClim/DownClim.git
cd DownClim
pip install -e ".[dev]"
```

## Testing

```bash
pip install "downclim[test]"
pytest -v
```

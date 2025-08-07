# DownClim

`DownClim` is an open source project that will make your climate downscaling
easily! It provides a bunch of tools to download and process climate data, and
to downscale it.

## Description

The purpose of `DownClim` is to offer a tool for regional climate projections
including the mechanistic 'dynamic' downscaling of the CORDEX initiative.
`DownClim` is opposed to the direct statistical downscaling of global climate
projections found in [WorldClim](https://www.worldclim.org/) and
[CHELSA](https://chelsa-climate.org/). The approach is justified by an
improvement in regional projections of [CORDEX](https://cordex.org/) compared to
[CMIP](https://www.wcrp-cmip.org/), although it can increase uncertainty and
sometimes be less reliable. The tool is an automated snakemake workflow easily
reproducible and scalable associated to conda environments for enhance
reproducibility and portability.

```{toctree}
:hidden:

Home <self>
Installation <installation>
Getting Started <getting_started>
Examples <examples>
API Reference <reference>
```

## Quick Navigation

- [Installation](installation.md)
- [Getting Started](getting_started.md)
- [Examples](examples.md)
- [API Reference](reference.md)

## Badges

| Package                                                                                                                                                   | Metadata                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [![PyPI version](https://img.shields.io/pypi/v/downclim)](https://pypi.org/project/downclim)                                                              | [![Python versions](https://img.shields.io/pypi/pyversions/downclim)](https://pypi.org/project/downclim)                                                     |
| [![Conda-forge version](https://img.shields.io/conda/vn/conda-forge/downclim)](https://anaconda.org/conda-forge/downclim)                                 | [![License](https://img.shields.io/github/license/DownClim/DownClim)](https://github.com/DownClim/DownClim/blob/main/LICENSE)                                |
| [![Tests status](https://github.com/DownClim/DownClim/actions/workflows/ci.yml/badge.svg)](https://github.com/DownClim/DownClim/actions/workflows/ci.yml) | [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://downclim.github.io/DownClim)                                               |
| [![Code coverage](https://codecov.io/gh/DownClim/DownClim/branch/main/graph/badge.svg)](https://codecov.io/gh/DownClim/DownClim)                          | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)                                             |
| [![Types: mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://github.com/python/mypy)                                                        | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

## Useful Links

- [Home](https://downclim.github.io/DownClim)
- [Code Repository](https://github.com/DownClim/DownClim)
- [Issues](https://github.com/DownClim/DownClim/issues)
- [Discussions](https://github.com/DownClim/DownClim/discussions)
- [Releases](https://github.com/DownClim/DownClim/releases)

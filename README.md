# DownClim - Downscale Climate Projections

Sylvain Schmitt - Thomas Arsouze - Ghislain Vieilledent

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/thomasarsouze/DownClim/workflows/CI/badge.svg
[actions-link]:             https://github.com/thomasarsouze/DownClim/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/DownClim
[conda-link]:               https://github.com/conda-forge/DownClim-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/thomasarsouze/DownClim/discussions
[pypi-link]:                https://pypi.org/project/DownClim/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/DownClim
[pypi-version]:             https://img.shields.io/pypi/v/DownClim
[rtd-badge]:                https://readthedocs.org/projects/DownClim/badge/?version=latest
[rtd-link]:                 https://DownClim.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

- [DownClim - Downscale Climate Projections](#downclim---downscale-climate-projections)
  - [Installation](#installation)
  - [Credentials](#credentials)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Data](#data)
    - [Baselines](#baselines)
    - [Projections](#projections)

The purpose of `DownClim` is to offer a tool for regional and national climate
projections including the mechanistic ‘dynamic’ downscaling of the CORDEX
initiative. `DownClim` is opposed to the direct statistical downscaling of
global climate projections found in WorldClim and CHELSA. The approach is
justified by an improvement in regional projections of CORDEX compared to CMIP,
although it can increase uncertainty and sometimes be less reliable. The tool is
an automated `snakemake` workflow easily reproducible and scalable associated to
`conda` environments for enhance reproducibility and portability.

![Workflow.](dag/dag.svg)

## Installation

As `DownClim` is under active development, no official release is available so
far. You can install from the source code:

```bash
git clone git@github.com:DownClim/DownClim.git
cd DownClim
# To install in "editable" / development mode
pip install -e .
```

## Credentials

Data are retrieve from the
[Institut Pierre-Simon Laplace node](https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/).
You need first to
[create an account](https://esgf.github.io/esgf-user-support/user_guide.html#create-an-account)
on this page
([create account](https://esgf-node.ipsl.upmc.fr/user/add/?next=http://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/)
link at corner right).

Then you’ll need to register credentials locally to use the workflow. For that
use a credentials_esgf yaml file reported in config.yml with keys openid and
pwd. For example using bash in linux:

```bash
openid=https://esgf-node.ipsl.upmc.fr/esgf-idp/openid/{user}
pwd={pwd}
echo -e "openid: $openid\npwd: $pwd" > config/credentials_esgf.yml
```

## Usage

Please follow the [documentation](https://DownClim.readthedocs.io/en/latest/)
for usage instructions. Start with the
[getting started guide](https://DownClim.readthedocs.io/en/latest/getting_started.html),
and then dive deeper in the
[advanced usage section](https://DownClim.readthedocs.io/en/latest/advanced_usage.html).

## Configuration

Very minimal set of information is needed by `DownClim` to run.

- Area of interest
  - **_aoi_**: names of the area to work with, _e.g._ New-Caledonia.
- Time
  - **_time_frequency_**: time frequency of data (month “mon”, day “day” or
    x-hourly “3hr”), currently only “mon” is available.
  - **_historical_period_**: period on which to adjust projections, \_e.g.\*
    1980-2005.
  - **_evaluation_period_**: period on which to evaluate projections, \_e.g.\*
    2006-2019.
  - **_projection_period_**: period on which to downscale the projections,
    \_e.g.\* 2071-2100.
- Variables
  - **_variables_**: variables to downscale, \_e.g. temperature at surface
    ’tas”, minimum temperature “tasmin”, maximum temperature “tasmax”,
    precipitations “pr”.
- Baseline
  - **_baseline_product_**: climate product for the baseline (CHELSA V2, GSHTD,
    CHIRPS).
- Projection
  - **_CMIP6Context_**: information about the CMIP6 projections to use.
  - **_CORDEXContext_**: information about the CORDEX projections to use.
  - **_esgf_credentials_**: path to the file defining the user credentials on
    esgf.
- Downscaling
  - **_aggregation_**: time aggregation before downscaling, currently only
    “monthly-means” are available.
  - **_downscaling_method_**: downscaling method to be used (bias correction
    “bc”, quantile-based “qt”, currently only bc is available).
- Evaluation
  - **_evaluation_product_**: climate product for the evaluation (CHELSA V2,
    CHRIPS, GSHTD).

## Data

### Baselines

[**CHELSA V2.1.1**](https://chelsa-climate.org/)**: Climatologies at high
resolution for the earth’s land surface areas**

_CHELSA (Climatologies at high resolution for the earth’s land surface areas) is
a very high resolution (30 arc sec, ~1km) global downscaled climate data set
currently hosted by the Swiss Federal Institute for Forest, Snow and Landscape
Research WSL. It is built to provide free access to high resolution climate data
for research and application, and is constantly updated and refined._

[**CHIRPS**](https://www.chc.ucsb.edu/data/chirps)**: Rainfall Estimates from
Rain Gauge and Satellite Observations**

_Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) is a
35+ year quasi-global rainfall data set. Spanning 50°S-50°N (and all longitudes)
and ranging from 1981 to near-present, CHIRPS incorporates our in-house
climatology, CHPclim, 0.05° resolution satellite imagery, and in-situ station
data to create gridded rainfall time series for trend analysis and seasonal
drought monitoring._

[**GSHTD**](https://gee-community-catalog.org/projects/gshtd/)**: Global
Seamless High-resolution Temperature Dataset**

_The Global Seamless High-resolution Temperature Dataset (GSHTD) presented in
this study offers a comprehensive and valuable resource for researchers across
various fields. Covering the period from 2001 to 2020, this dataset focuses on
land surface temperature (Ts) and near-surface air temperature (Ta). A unique
feature of GSHTD is its incorporation of seven types of temperature data,
including clear-sky daytime and nighttime Ts, all-sky daytime and nighttime Ts,
and mean, maximum, and minimum Ta. Notably, the dataset achieves global coverage
with an impressive 30 arcsecond or 1km spatial resolution._

### Projections

[**CMIP**](https://wcrp-cmip.org/)**: Coupled Model Intercomparison Project**

_CMIP is a project of the World Climate Research Programme (WCRP) providing
climate projections to understand past, present and future climate changes. CMIP
and its associated data infrastructure have become essential to the
Intergovernmental Panel on Climate Change (IPCC) and other international and
national climate assessments._

[**CORDEX**](https://cordex.org/)**: Coordinated Regional Climate Downscaling
Experiment**

_The CORDEX vision is to advance and coordinate the science and application of
regional climate downscaling through global partnerships._

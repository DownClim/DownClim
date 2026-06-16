"""Data product connectors and processors for DownClim."""

from __future__ import annotations

from .chelsa2 import get_chelsa2
from .chirps import get_chirps
from .cmip6 import CMIP6Context, get_cmip6, get_cmip6_context_from_filename
from .connectors import connect_to_ee, connect_to_esgf, connect_to_gcfs
from .cordex import (
    CORDEXContext,
    get_cordex,
    get_cordex_context_from_filename,
    get_download_scripts,
    inspect_cordex,
)
from .era5 import get_era5
from .gshtd import get_gshtd
from .tmf import compute_tmf, get_tmf
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    VariableAttributes,
    check_input_dir,
    climatology_filename,
    get_lon_lat_names,
    get_monthly_climatology,
    get_monthly_mean,
    prep_dataset,
    save_grid_file,
    sel_period,
    split_period,
)

__all__ = [
    "Aggregation",
    "CMIP6Context",
    "CORDEXContext",
    "DataProduct",
    "Frequency",
    "VariableAttributes",
    "check_input_dir",
    "climatology_filename",
    "compute_tmf",
    "connect_to_ee",
    "connect_to_esgf",
    "connect_to_gcfs",
    "get_chelsa2",
    "get_chirps",
    "get_cmip6",
    "get_cmip6_context_from_filename",
    "get_cordex",
    "get_cordex_context_from_filename",
    "get_download_scripts",
    "get_era5",
    "get_gshtd",
    "get_lon_lat_names",
    "get_monthly_climatology",
    "get_monthly_mean",
    "get_tmf",
    "inspect_cordex",
    "prep_dataset",
    "save_grid_file",
    "sel_period",
    "split_period",
]

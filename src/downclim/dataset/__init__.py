"""Data product connectors and processors for DownClim."""

from __future__ import annotations

from .chelsa2 import get_chelsa2
from .chirps import get_chirps
from .cmip6 import CMIP6Context, get_cmip6
from .connectors import connect_to_ee, connect_to_esgf, connect_to_gcfs
from .cordex import CORDEXContext, get_cordex, inspect_cordex
from .era5 import get_era5
from .gshtd import get_gshtd
from .tmf import compute_tmf
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    VariableAttributes,
)

__all__ = [
    "Aggregation",
    "CMIP6Context",
    "CORDEXContext",
    "DataProduct",
    "Frequency",
    "VariableAttributes",
    "compute_tmf",
    "connect_to_ee",
    "connect_to_esgf",
    "connect_to_gcfs",
    "get_chelsa2",
    "get_chirps",
    "get_cmip6",
    "get_cordex",
    "get_era5",
    "get_gshtd",
    "inspect_cordex",
]

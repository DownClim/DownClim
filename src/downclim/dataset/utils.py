from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as dt
from enum import Enum
from typing import Any

import numpy as np
import xarray as xr
from aenum import MultiValueEnum


class Frequency(MultiValueEnum):
    """Class to define the frequency of the data handled.
    Usually 'daily | monthly | annualy'.
        So far only 'monthly' is implemented.

    Args:
        MultiValueEnum ('str'): string representation of the time frequency.

    Raises:
        ValueError: if the frequency is not implemented.
    """

    MONTHLY = "monthly", "m", "mon", "month", "Monthly", "Month", "MONTHLY", "MONTH"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented frequency '{value}'. Right now only 'monthly' is implemented."
        raise ValueError(msg)


class Aggregation(MultiValueEnum):
    """Class to define the aggregation method of the data handled.
    Usually 'mean | sum | max | min'.
    So far only 'monthly-mean' is implemented.

    Args:
        MultiValueEnum ('str'): string representation of the aggregation method.

    Raises:
        ValueError: if the aggregation method is not implemented.
    """

    MONTHLY_MEAN = "monthly_mean", "monthly-mean", "monthly_means", "monthly-means"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented aggregation method '{value}'. Right now only 'monthly-mean' is implemented."
        raise ValueError(msg)


@dataclass
class DataProductProperties:
    """Data class to define the properties of the data products handled.

    Args:
        product_name (str): name of the data product.
        period (tuple[int, int]): period of the data product in the format (begin_year, end_year).
        scale_factor (dict[str, float]): scale factor for each variable in the data product.
        add_offset (dict[str, float]): add offset for each variable in the data product.
        url (str): url to download the data product.
            If the data is stored on earthengine, provides the image collection path instead.

    """

    product_name: str
    period: tuple[int, int]
    scale_factor: dict[str, float]
    add_offset: dict[str, float]
    url: str


class DataProduct(Enum):
    """Enum Class to define the data products handled.

    Raises:
        ValueError: if no method exists for downloading the data product.

    Returns:
        DataProduct: the data product to retrieve.
    """

    CHELSA = (
        "chelsa",
        (1980, 2019),
        {"pr": 0.1, "tas": 0.1, "tasmin": 0.1, "tasmax": 0.1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        "https://os.zhdk.cloud.switch.ch/envicloud/chelsav2/GLOBAL",
    )
    CMIP6 = (
        "cmip6",
        (1850, 2100),
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    )
    CORDEX = (
        "cordex",
        (1850, 2100),
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        "https://esgf-node.ipsl.upmc.fr/esg-search",
    )
    GSHTD = (
        "gshtd",
        (2001, 2020),
        {"tas": 0.02, "tasmin": 0.02, "tasmax": 0.02},
        {"tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        "projects/sat-io/open-datasets/GSHTD/",
    )
    CHIRPS = "chirps", (1980, 2024), {"pr": 1}, {"pr": 0}, "UCSB-CHG/CHIRPS/DAILY"

    @classmethod
    def _missing_(cls, value: Any) -> DataProduct:
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if value in member.to_numpy():
                    return member
        msg = f"Unknown or not implemented retrieval of data product '{value}'."
        raise ValueError(msg)


variables_attributes = {
    "pr": {
        "standard_name": "precipitation",
        "long_name": "Monthly precipitation",
        "units": "mm month-1",
        "explanation": "Precipitation in the earth's atmosphere, monthly means precipitation of water in all phases.",
    },
    "tas": {
        "standard_name": "temperature at surface",
        "long_name": "Monthly mean daily air temperature",
        "units": "°C",
        "explanation": "Monthly mean air temperatures at 2 meters.",
    },
    "tasmin": {
        "standard_name": "minimum temperature at surface",
        "long_name": "Monthly minimum daily air temperature",
        "units": "°C",
        "explanation": "Monthly minimum air temperatures at 2 meters.",
    },
    "tasmax": {
        "standard_name": "maximum temperature at surface",
        "long_name": "Monthly maximum daily air temperature",
        "units": "°C",
        "explanation": "Monthly maximum air temperatures at 2 meters.",
    },
    "pet": {
        "standard_name": "potential evapotranspiration",
        "long_name": "Monthly potential evapotranspiration",
        "units": "mm month-1",
        "explanation": "Potential evapotranspiration for each month; calculated with the Penman-Monteith equation.",
    },
}

scale_factors = {
    "chelsa2": {"pr": 0.1, "tas": 0.1, "tasmin": 0.1, "tasmax": 0.1},
    "cmip6": {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
    "cordex": {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
    "gshtd": {"tas": 0.02, "tasmin": 0.02, "tasmax": 0.02},
}

add_offsets = {
    "chelsa2": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "cmip6": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "cordex": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "gshtd": {"tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
}


def split_period(period: tuple[int, int]) -> tuple[str, str]:
    """
    Split a period string into two dates to define dmin and dmax.

    Parameters
    ----------
    period: tuple[int, int]
        Period in the format (begin_year, end_year).

    Returns
    -------
    tuple
        Tuple with two strings representing the start and end of the period.
    """
    return (f"{period[0]}-01-01", f"{period[1]}-12-31")


def convert_cf_to_dt(t: np.datetime64) -> dt:
    return dt.strptime(str(t), "%Y-%m-%d %H:%M:%S")


def prep_dataset(ds: xr.Dataset, dataset_type: str = "cordex") -> xr.Dataset:
    """Prepare a dataset for downscaling.

    Some operations (scaling and offsets) are applied to the dataset depending on
    the variables in the dataset.

    Parameters
    ----------
    ds: xr.Dataset original dataset to prepare.
    dataset_type: str, optional
        Type of dataset to prepare. Default is "cordex" but can be "cmip6", "chelsa2" or "gshtd".

    Returns
    -------
    xr.Dataset
        Prepared dataset with scaling and offsets applied to the variables.
    """
    cf = not isinstance(ds["time"].to_numpy()[0], np.datetime64)
    if cf:  # only cftime if not dt but should include more cases
        ds["time"] = [*map(convert_cf_to_dt, ds.time.values)]
    for key in ds:
        ds[key] = (
            ds[key] * scale_factors[dataset_type][key] + add_offsets[dataset_type][key]
        )
        ds[key].attrs = variables_attributes[key]
    return ds


def get_monthly_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the monthly mean of a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with daily values.

    Returns
    -------
    xr.Dataset
        Dataset with average monthly means values.
    """
    return ds.resample(time="1M").mean()


def get_monthly_climatology(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the monthly climatology of a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with monthly values over multiple years.

    Returns
    -------
    xr.Dataset
        Dataset with average monthly means values.
    """

    return ds.groupby("time.month").mean("time")

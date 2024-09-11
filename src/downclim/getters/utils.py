from __future__ import annotations

from enum import Enum, auto

import numpy as np
import xarray as xr


class Frequency(Enum):
    MONTHLY = auto()
    DAILY = auto()


class Aggregation(Enum):
    MONTHLY_MEAN = auto()


def get_frequency(frequency: str) -> Frequency:
    """
    Get the frequency type from a string.

    Parameters
    ----------
    frequency: str
        frequency string. Usually 'daily | monthly | annualy'.
        So far only 'monthly' is implemented.

    Returns
    -------
    Frequency
    """
    if frequency in [
        "m",
        "mon",
        "monthly",
        "month",
        "Monthly",
        "Month",
        "MONTHLY",
        "MONTH",
    ]:
        return Frequency.MONTHLY
    if frequency in ["d", "day", "daily", "Daily", "Day", "D", "DAILY", "DAY"]:
        return Frequency.DAILY
    msg = (
        "Unknown or not implemented frequency. Right now only 'monthly' is implemented."
    )
    raise ValueError(msg)


def get_aggregation(aggregation: str) -> Aggregation:
    """
    Get the aggregation type from a string.

    Parameters
    ----------
    aggregation: str
        Aggregation string.

    Returns
    -------
    Aggregation
        Aggregation type.
    """
    if aggregation in ["mean", "average", "avg", "Mean", "Average", "Avg"]:
        return Aggregation.MEAN
    if aggregation in ["sum", "Sum", "SUM"]:
        return Aggregation.SUM
    if aggregation in ["max", "Max", "MAX"]:
        return Aggregation.MAX
    if aggregation in ["min", "Min", "MIN"]:
        return Aggregation.MIN
    msg = (
        "Unknown or not implemented aggregation. Must be 'mean', 'sum', 'max' or 'min'."
    )
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
    "gshtd": 0.02,
}

add_offsets = {
    "chelsa2": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "cmip6": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "cordex": {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
    "gshtd": -273.15,
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


def convert_cf_to_dt(x):
    from datetime import datetime as dt

    return dt.strptime(str(x), "%Y-%m-%d %H:%M:%S")


def prep_dataset(ds, dataset_type="cordex"):
    """
    Prepare a dataset for downscaling.
    Some operations (scaling and offsets) are applied to the dataset depending on
    the variables in the dataset.
    """
    cf = type(ds["time"].to_numpy()[0]) is not np.datetime64
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

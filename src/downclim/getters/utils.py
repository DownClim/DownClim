from __future__ import annotations

import xarray as xr

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


def split_period(period: str) -> tuple[str, str]:
    """
    Split a period string into two dates to define dmin and dmax.

    Parameters
    ----------
    period: str
        Period string in the format "YYYY-YYYY".

    Returns
    -------
    tuple
        Tuple with two strings representing the start and end of the period.
    """
    return (period.split("-")[0] + "-01-01", period.split("-")[1] + "-12-31")


def convert_cf_to_dt(x):
    from datetime import datetime as dt

    return dt.strptime(str(x), "%Y-%m-%d %H:%M:%S")


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

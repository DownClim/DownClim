from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from .connectors import connect_to_ee, ee_image_collection
from .get_aoi import get_aois_informations
from .utils import (
    Aggregation,
    Frequency,
    get_frequency,
    get_monthly_climatology,
    get_monthly_mean,
    split_period,
    variables_attributes,
)


def get_chirps_single_climatology(
    aoi: pd.DataFrame,
    period: tuple[(int, int)] = ((1980, 2005), (2006, 2019)),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
) -> xr.Dataset:
    """
    Retrieve CHIRPS climatology for a given area of interest and time period.

    Parameters
    ----------
    aoi: pd.DataFrame
        Area of interest. It is a pandas.DataFrame (boundaries) on the format:
            minx     miny     maxx     maxy
        0   0        0        10       10
        These are the boundaries of a geopandas.GeoDataFrame that can be generated from the `get_aois` function.
    period: tuple
        Time period to retrieve. e.g. ((1980-2005)).
    time_frequency: str
        Time frequency of CHRIPS data. Default is "mon", i.e. original CHIRPS data is averaged monthly.
    aggregation: str
        Aggregation method to build the climatology. Default is "monthly-means".

    Returns
    -------
    xr.Dataset
        CHIRPS data for the given area of interest and time period.
    """
    bounds = aoi.bounds
    dmin, dmax = split_period(period)

    ic = ee.ImageCollection(ee_image_collection["CHIRPS"]).filterDate(dmin, dmax)
    leg1 = ee.Geometry.Rectangle(
        bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]
    )
    ds = xr.open_mfdataset(
        [ic], engine="ee", projection=ic.first().select(0).projection(), geometry=leg1
    )
    ds = ds.transpose("time", "lat", "lon")
    if time_frequency == Frequency.MONTHLY:
        ds = get_monthly_mean(ds)
    else:
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)

    if aggregation == Aggregation.MONTHLY_MEAN:
        ds = get_monthly_climatology(ds)
    else:
        msg = "Currently only monthly-means aggregation available!"
        raise Exception(msg)

    ds = ds.rename({"precipitation": "pr"})
    ds.pr.attrs = variables_attributes["pr"]
    return ds


def get_chirps(
    aois: Iterable[gpd.GeoDataFrame],
    periods: tuple[(int, int)] = ((1980, 2005), (2006, 2019)),
    time_frequency: str = "mon",
    aggregation: str = "monthly-means",
) -> None:
    """Retrieve CHIRPS precipitation data for a list of areas of interest and periods. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Data is available at: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY

    Parameters
    ----------
    aois: Iterable[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    periods: tuple[(int, int)]
        Tuple of time frames to retrieve, and build the climatologies on.
        Should correspond to the historical period and the evaluation period.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: ((1980, 2005), (2006,2019)).
    time_frequency: str, optional
        Time frequency of CHIRPS data (currently only "mon" available). Defaults to "mon".
    aggregation: str, optional
        Defaults to "monthly-means".

    Returns
    -------
    No output from the function. Dataset is stored in the "./results/chirps/" directory.
    """
    output_directory = "./results/chirps"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    aois_names, aois_bounds = get_aois_informations(aois)

    time_frequency = get_frequency(time_frequency)

    connect_to_ee()

    for aoi_name, aoi_bounds in zip(aois_names, aois_bounds, strict=False):
        for period in periods:
            ds = get_chirps_single_climatology(
                aoi_bounds, period, time_frequency, aggregation
            )
            output_file = (
                f"{output_directory}/{aoi_name}_chirps_{aggregation}_{period}.nc"
            )
            ds.to_netcdf(output_file)

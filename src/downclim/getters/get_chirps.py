from __future__ import annotations

from pathlib import Path

import ee
import geopandas
import pandas as pd
import xarray as xr

from .connectors import connect_to_ee
from .get_aoi import get_aois_informations
from .utils import (
    ee_image_collection,
    get_monthly_climatology,
    get_monthly_mean,
    split_period,
    variables_attributes,
)


def get_chirps(
    aoi: geopandas.GeoDataFrame | pd.DataFrame,
    period: str,
    time_frequency: str = "mon",
    aggregation: str = "monthly-means",
) -> xr.Dataset:
    """
    Retrieve CHIRPS climatology for a given area of interest and time period.

    Parameters
    ----------
    aoi: geopandas.GeoDataFrame | pandas.DataFrame
        Area of interest. It can be a geopandas.GeoDataFrame (from a shapefile)
            or a pandas.DataFrame (boundaries) on the format:
            minx     miny     maxx     maxy
        0   0        0        10       10
    period: str
        Time period to retrieve. e.g. "1980-2005".
    time_frequency: str
        Time frequency of CHRIPS data. Default is "mon", i.e. original CHIRPS data is averaged monthly.
    aggregation: str
        Aggregation method to build the climatology. Default is "monthly-means".

    Returns
    -------
    xr.Dataset
        CHIRPS data for the given area of interest and time period.
    """
    bounds = aoi.bounds if isinstance(aoi, geopandas.GeoDataFrame) else aoi
    dmin, dmax = split_period(period)
    connect_to_ee()
    ic = ee.ImageCollection(ee_image_collection["CHIRPS"]).filterDate(dmin, dmax)
    leg1 = ee.Geometry.Rectangle(
        bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]
    )
    ds = xr.open_mfdataset(
        [ic], engine="ee", projection=ic.first().select(0).projection(), geometry=leg1
    )
    ds = ds.transpose("time", "lat", "lon")
    if time_frequency == "mon":
        ds = get_monthly_mean(ds)
    else:
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)
    if aggregation == "monthly-means":
        ds = get_monthly_climatology(ds)
    else:
        msg = "Currently only monthly-means aggregation available!"
        raise Exception(msg)
    ds = ds.rename({"precipitation": "pr"})
    ds.pr.attrs = variables_attributes["pr"]
    return ds


def get_chirps_multiple(
    aois: list[geopandas.GeoDataFrame],
    periods: list[str],
    time_frequency: str = "mon",
    aggregation: str = "monthly-means",
    output_directory: str = "./results/chirps/",
) -> None:
    aois_names, aois_bounds = get_aois_informations(aois)
    if time_frequency != "mon":
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)
    for aoi_name, aoi_bounds in zip(aois_names, aois_bounds, strict=False):
        for period in periods:
            ds = get_chirps(aoi_bounds, period, time_frequency, aggregation)
            output_file = f"{Path(output_directory).parent}/{aoi_name}_chirps_{aggregation}_{period}.nc"
            ds.to_netcdf(output_file)

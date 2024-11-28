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
    add_offsets,
    get_monthly_climatology,
    get_monthly_mean,
    scale_factors,
    split_period,
    variables_attributes,
)


# funs
def get_gshtd_single(
    aoi: pd.DataFrame,
    var: list[str],
    period: tuple[(int, int)],
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
) -> xr.Dataset:
    bounds = aoi.bounds
    dmin, dmax = split_period(period)

    collection = {"tas": "TMEAN", "tasmin": "TMIN", "tasmax": "TMAX"}
    ic = ee.ImageCollection(ee_image_collection["GSHTD"] + collection[var]).filterDate(
        dmin, dmax
    )

    leg = ee.Geometry.Rectangle(
        bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]
    )

    ds = xr.open_dataset(
        ic, engine="ee", projection=ic.first().select(0).projection(), geometry=leg
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
    ds = ds.where(ds.b1 > 0)
    ds["b1"] = ds.b1 * scale_factors["GSHTD"] + add_offsets["GSHTD"]  # K to °C
    ds.b1.attrs = variables_attributes[var]
    return ds.rename({"b1": var})


# code
def get_gshtd(
    aois: Iterable[gpd.GeoDataFrame],
    vars: Iterable[str] = ["tas", "tasmin", "tasmax"],
    periods: tuple[(int, int)] = ((1980, 2005), (2006, 2019)),
    time_frequency: str = "mon",
    aggregation: str = "monthly-mean",
) -> None:
    """
    Get GSHTD data (https://gee-community-catalog.org/projects/gshtd/)
    for a given set of areas of interest and time periods, for given variables.
    Sequential calls get_gshtd.

    Parameters
    ----------
    aois: Iterable[geopandas.GeoDataFrame | pandas.DataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    vars: List[str]
        List of variables to retrieve. Default is ["tas", "tasmin", "tasmax"].
    periods: List[str]
        List of time periods to retrieve. e.g. ["1980-2005", "2006-2019", "2071-2100"].
    time_frequency: str
        Time frequency of the data. Default is "mon" (monthly)
    aggregation: str
        Aggregation method to build the climatology. Default is "monthly-means".

    """

    output_directory = "./results/gshtd"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    aois_names, aois_bounds = get_aois_informations(aois)

    time_frequency = Frequency(time_frequency)

    connect_to_ee()

    for aoi_name, aoi_bounds in zip(aois_names, aois_bounds, strict=False):
        for period in periods:
            ds = xr.merge([get_gshtd_single(aoi_bounds, var, period) for var in vars])
            output_file = (
                f"{output_directory}/{aoi_name}_gshtd_{aggregation}_{period}.nc"
            )
            ds.to_netcdf(output_file)

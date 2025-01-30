from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from .aoi import get_aoi_informations
from .connectors import connect_to_ee, ee_image_collection
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
    aoi_bounds: pd.DataFrame,
    aoi_name: str,
    variable: str,
    period: tuple[int, int],
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
) -> xr.Dataset:
    print(
        f'Getting GSHTD data for period : "{period}" and variable : "{variable}" on area of interest : "{aoi_name}"'
    )
    dmin, dmax = split_period(period)

    collection = {"tas": "TMEAN", "tasmin": "TMIN", "tasmax": "TMAX"}
    ic = ee.ImageCollection(
        ee_image_collection["GSHTD"] + collection[variable]
    ).filterDate(dmin, dmax)
    geom = ee.Geometry.Rectangle(*aoi_bounds.to_numpy()[0])
    ds = xr.open_dataset(
        ic, engine="ee", projection=ic.first().select(0).projection(), geometry=geom
    )
    ds = ds.transpose("time", "lat", "lon")
    if time_frequency == Frequency.MONTHLY:
        ds = get_monthly_mean(ds)
    else:
        msg = "Currently only monthly time frequency available!"
        raise ValueError(msg)

    if aggregation == Aggregation.MONTHLY_MEAN:
        ds = get_monthly_climatology(ds)
    else:
        msg = "Currently only monthly-means aggregation available!"
        raise ValueError(msg)
    ds = ds.where(ds.b1 > 0)
    ds["b1"] = (
        ds.b1 * scale_factors["gshtd"][variable] + add_offsets["gshtd"][variable]
    )  # K to °C
    ds.b1.attrs = variables_attributes[variable]
    return ds.rename({"b1": variable})


# code
def get_gshtd(
    aoi: Iterable[gpd.GeoDataFrame | pd.DataFrame],
    variable: Iterable[str] = ("tas", "tasmin", "tasmax"),
    period: tuple[int, int] = (1980, 2005),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    output_dir: str = "./results/gshtd",
) -> None:
    """
    Get GSHTD data (https://gee-community-catalog.org/projects/gshtd/)
    for a given set of areas of interest and time periods, for given variables.
    Sequential calls get_gshtd.

    Parameters
    ----------
    aoi: Iterable[geopandas.GeoDataFrame | pandas.DataFrame]
        Iterable of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    variable: Iterable[str]
        Iterable of variables to retrieve. Default is ["tas", "tasmin", "tasmax"].
    period: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (1980, 2005).
    time_frequency: Frequency, optional
        Time frequency of CHIRPS data (currently only Frequency.MONTHLY available).
        Defaults to Frequency.MONTHLY.
    aggregation: Aggregation, optional
        Method used to aggregate the data and build the climatology
        (currently only Aggregation.MONTHLY_MEAN available).
        Defaults to Aggregation.MONTHLY_MEAN.
    output_dir: str, optional
        Output directory where the GSHTD climatology will be stored.
        Defaults to "./results/gshtd".

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    aoi_name, aoi_bound = get_aoi_informations(aoi)

    connect_to_ee()

    print("Downloading GSHTD data...")
    for aoi_n, aoi_b in zip(aoi_name, aoi_bound, strict=False):
        # First check if the data is already downloaded
        output_file = (
            f"{output_dir}/{aoi_n}_gshtd_{aggregation.value}_{period[0]}-{period[1]}.nc"
        )
        if Path(output_file).is_file():
            print(
                f"""File {output_file} already exists, skipping...
                If this is not the expected behaviour, please remove the file and run the function again."""
            )
            continue
        ds = xr.merge(
            [
                get_gshtd_single(aoi_b, aoi_n, var, period, time_frequency, aggregation)
                for var in variable
            ]
        )
        ds.to_netcdf(output_file)

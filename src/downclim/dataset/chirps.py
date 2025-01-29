from __future__ import annotations

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
    get_monthly_climatology,
    get_monthly_mean,
    split_period,
    variables_attributes,
)


def get_chirps_single_climatology(
    aoi_bounds: pd.DataFrame,
    aoi_name: str,
    period: tuple[int, int],
    time_frequency: Frequency,
    aggregation: Aggregation,
) -> xr.Dataset:
    """
    Retrieve CHIRPS climatology for a given area of interest and time period.

    Parameters
    ----------
    aoi_bounds: pd.DataFrame
        Area of interest. It is a pandas.DataFrame (boundaries) on the format:
            minx     miny     maxx     maxy
        0   0        0        10       10
        These are the boundaries of a geopandas.GeoDataFrame that can be generated from the `get_aoi` function.
    aoi_name: str
        Name of the area of interest.
    period: tuple
        Time period to retrieve. e.g. (1980-2005).
    time_frequency: Frequency
        Time frequency of CHRIPS data.
    aggregation: Aggregation
        Aggregation method to build the climatology.

    Returns
    -------
    xr.Dataset
        CHIRPS data for the given area of interest and time period.
    """
    print(
        f'Getting CHIRPS data for period : "{period}" and area of interest : "{aoi_name}"'
    )
    dmin, dmax = split_period(period)

    ic = ee.ImageCollection(ee_image_collection["CHIRPS"]).filterDate(dmin, dmax)
    geom = ee.Geometry.Rectangle(*aoi_bounds.to_numpy()[0])
    ds = xr.open_mfdataset(
        [ic], engine="ee", projection=ic.first().select(0).projection(), geometry=geom
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

    ds = ds.rename({"precipitation": "pr"})
    ds.pr.attrs = variables_attributes["pr"]
    return ds


def get_chirps(
    aoi: list[gpd.GeoDataFrame],
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
) -> None:
    """Retrieve CHIRPS precipitation data for a list of areas of interest and periods. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Data is available at: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY

    Parameters
    ----------
    aoi: Iterable[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    baseline_year: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Should correspond to the historical period.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (1980, 2005).
    evaluation_year: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Should correspond to the evaluation period.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (2006, 2019).
    time_frequency: Frequency, optional
        Time frequency of CHIRPS data (currently only Frequency.MONTHLY available).
        Defaults to Frequency.MONTHLY, i.e. original CHIRPS data is averaged monthly
    aggregation: Aggregation, optional
        Method used to aggregate the data and build the climatology
        (currently only Aggregation.MONTHLY_MEAN available).
        Defaults to Aggregation.MONTHLY_MEAN.

    Returns
    -------
    No output from the function. Dataset is stored in the "./results/chirps/" directory.
    """
    output_directory = "./results/chirps"
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print("Downloading CHIRPS data...")

    aois_names, aois_bounds = get_aoi_informations(aoi)

    connect_to_ee()

    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        for period in [baseline_year, evaluation_year]:
            # First check if the data is already downloaded
            output_file = f"{output_directory}/{aoi_n}_chirps_{aggregation.value}_{period[0]}-{period[1]}.nc"
            if Path(output_file).is_file():
                print(
                    f"""File {output_file} already exists, skipping...
                    If this is not the expected behaviour, please remove the file and run the function again."""
                )
                continue
            ds = get_chirps_single_climatology(
                aoi_b, aoi_n, period, time_frequency, aggregation
            )
            ds.to_netcdf(output_file)

from __future__ import annotations

from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from .aoi import get_aoi_informations
from .connectors import connect_to_ee
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    VariableAttributes,
    get_monthly_climatology,
    get_monthly_mean,
    split_period,
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

    ic = ee.ImageCollection(DataProduct.CHIRPS.url).filterDate(dmin, dmax)
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
    ds.pr.attrs = VariableAttributes["pr"]
    return ds


def get_chirps(
    aoi: list[gpd.GeoDataFrame],
    period: tuple[int, int] = (1980, 2005),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    output_dir: str = "./results/chirps",
) -> None:
    """Retrieve CHIRPS precipitation data for a list of areas of interest and periods. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Data is available at: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY

    Parameters
    ----------
    aoi: Iterable[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    period: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (1980, 2005).
    time_frequency: Frequency, optional
        Time frequency of CHIRPS data (currently only Frequency.MONTHLY available).
        Defaults to Frequency.MONTHLY, i.e. original CHIRPS data is averaged monthly
    aggregation: Aggregation, optional
        Method used to aggregate the data and build the climatology
        (currently only Aggregation.MONTHLY_MEAN available).
        Defaults to Aggregation.MONTHLY_MEAN.
    output_dir: str, optional
        Output directory where the CHIRPS climatology will be stored.
        Defaults to "./results/chirps".

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)

    connect_to_ee()

    print("Downloading CHIRPS data...")
    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        # First check if the data is already downloaded
        output_file = f"{output_dir}/{aoi_n}_chirps_{aggregation.value}_{period[0]}-{period[1]}.nc"
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

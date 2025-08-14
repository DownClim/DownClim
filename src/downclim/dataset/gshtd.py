from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from ..aoi import get_aoi_informations
from .connectors import connect_to_ee
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    VariableAttributes,
    climatology_filename,
    get_grid,
    get_monthly_climatology,
    get_monthly_mean,
    prep_dataset,
    split_period,
)


# funs
def _get_gshtd_single(
    aoi_bounds: pd.DataFrame,
    aoi_name: str,
    variable: str,
    period: tuple[int, int],
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
) -> xr.Dataset:
    """
    Internal function. Get GSHTD data for one area of interest, one variable and one time period.
    """
    print(
        f'Getting GSHTD data for period : "{period}" and variable : "{variable}" on area of interest : "{aoi_name}"'
    )
    dmin, dmax = split_period(period)

    collection = {"tas": "TMEAN", "tasmin": "TMIN", "tasmax": "TMAX"}
    ic = ee.ImageCollection(DataProduct.GSHTD.url + collection[variable]).filterDate(dmin, dmax)
    geom = ee.Geometry.Rectangle(*aoi_bounds.to_numpy()[0])
    if ic.size().getInfo() == 0:
        msg = f"""
                No data found for the period {dmin} - {dmax} and variable {variable}.
                GSHTD dataset is available from {DataProduct.GSHTD.period[0]} to {DataProduct.GSHTD.period[1]}.
                """
        raise ValueError(msg)

    ds = xr.open_dataset(
        ic, engine="ee", projection=ic.first().select(0).projection(), geometry=geom
    )

    ds = ds.transpose("time", "lat", "lon")
    ds = ds.rename({"b1": variable})
    ds = prep_dataset(ds, DataProduct.GSHTD)
    ds[variable].attrs = asdict(VariableAttributes[variable])

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

    return ds.where(ds[variable] > 0)


# code
def get_gshtd(
    aoi: list[gpd.GeoDataFrame],
    variable: Iterable[str] = ("tas", "tasmin", "tasmax"),
    period: Iterable[int, int] = (2015, 2018),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    output_dir: str | None = None,
    ee_project: str | None = None,
) -> None:
    """
    Get GSHTD data (https://gee-community-catalog.org/projects/gshtd/)
    for a given set of areas of interest and time periods, for given variables.

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
    ee_project: str | None = None,
        Earth Engine project ID to use for the download.

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    data_product = DataProduct.GSHTD

    # Create output directory
    if output_dir is None:
        output_dir = f"./results/{data_product.product_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, aoi_bound = get_aoi_informations(aoi)

    connect_to_ee(ee_project=ee_project)

    print("Downloading GSHTD data...")
    for aoi_n, aoi_b in zip(aoi_name, aoi_bound, strict=False):
        # First check if the data is already downloaded
        output_file = (
            climatology_filename(output_dir, aoi_n, data_product, aggregation, tuple(period))
        )
        if Path(output_file).is_file():
            print(
                f"""File {output_file} already exists, skipping...
                If this is not the expected behaviour, please remove the file and run the function again."""
            )
            continue
        ds = xr.merge(
            [
                _get_gshtd_single(aoi_b, aoi_n, var, tuple(period), time_frequency, aggregation)
                for var in variable
            ]
        )
        ds.to_netcdf(output_file)

        if not Path(f"{output_dir}/{data_product.product_name}_{aoi_n}_grid.nc").is_file():
            # Save the grid for the dataset
            print(f"Saving {data_product.product_name} grid for {aoi_n}...")
            grid = get_grid(ds, data_product)
            grid.to_netcdf(f"{output_dir}/{data_product.product_name}_{aoi_n}_grid.nc")

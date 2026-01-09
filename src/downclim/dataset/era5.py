from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from ..aoi import get_aoi_informations
from ..logging_config import get_logger
from .connectors import connect_to_ee
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    _check_output_dir,
    climatology_filename,
    get_monthly_climatology,
    get_monthly_mean,
    prep_dataset,
    save_grid_file,
    split_period,
)

data_product = DataProduct.ERA5

logger = get_logger(__name__)


def _get_era5_area_period(
    aoi_bounds: pd.DataFrame,
    aoi_name: str,
    variable: Iterable[str],
    period: tuple[int, int],
    time_frequency: Frequency,
    aggregation: Aggregation,
) -> xr.Dataset:
    """
    Retrieve ERA5 climatology for a given area of interest and time period.

    Parameters
    ----------
    aoi_bounds: pd.DataFrame
        Area of interest. It is a pandas.DataFrame (boundaries) on the format:
            minx     miny     maxx     maxy
        0   0        0        10       10
        These are the boundaries of a geopandas.GeoDataFrame that can be generated from the `get_aoi` function.
    aoi_name: str
        Name of the area of interest.
    variable: Iterable[str]
        Iterable of variables to retrieve from the dataset.
    period: tuple
        Time period to retrieve. e.g. (1980-2005).
    time_frequency: Frequency
        Time frequency of CHRIPS data.
    aggregation: Aggregation
        Aggregation method to build the climatology.

    Returns
    -------
    xr.Dataset
        ERA5 data for the given area of interest and time period.
    """
    logger.info(
        'Getting ERA5 data for period : "%s" and area of interest : "%s"',
        period,
        aoi_name,
    )
    dmin, dmax = split_period(period)

    ic = ee.ImageCollection(data_product.url).filterDate(dmin, dmax)  # type: ignore
    geom = ee.Geometry.Rectangle(*aoi_bounds.to_numpy()[0])  # type: ignore
    if ic.size().getInfo() == 0:
        msg = f"""
                No data found for the period {dmin} - {dmax}.
                ERA5 dataset is available from {data_product.period[0]} to {data_product.period[1]}.
                """
        logger.error(msg)
        raise ValueError(msg)

    ds = xr.open_mfdataset(
        [ic], engine="ee", projection=ic.first().select(0).projection(), geometry=geom
    )

    execute_variables = []
    for var in variable:
        for k, v in data_product.variables_names.items():
            if var == v:
                execute_variables.append(k)
                break
        else:
            logger.warning(
                "Variable %s not available for %s. Skipping them...",
                var,
                data_product.product_name,
            )

    ds = (
        ds.transpose("time", "lat", "lon")[execute_variables]
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        .rio.write_crs("epsg:4362")
    )
    ds = prep_dataset(ds, data_product)

    if time_frequency == Frequency.MONTHLY:
        ds = get_monthly_mean(ds)
    else:
        msg = "Currently only monthly time frequency available!"
        logger.error(msg)
        raise ValueError(msg)

    if aggregation == Aggregation.MONTHLY_MEAN:
        ds = get_monthly_climatology(ds)
    else:
        msg = "Currently only monthly-means aggregation available!"
        logger.error(msg)
        raise ValueError(msg)

    return ds


def get_era5(
    aoi: list[gpd.GeoDataFrame],
    variable: Iterable[str] = ("pr", "tas", "tasmin", "tasmax"),
    period: tuple[int, int] = (1980, 2005),
    time_frequency: Frequency = Frequency.MONTHLY,  # type: ignore[assignment]
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,  # type: ignore[assignment]
    output_dir: str | None = None,
    ee_project: str | None = None,
) -> None:
    """Retrieve ERA5 precipitation data for a list of areas of interest and periods. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Data is available at: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_ERA5_DAILY

    Parameters
    ----------
    aoi: Iterable[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects (from shapefiles) or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    variable: Iterable[str]
        Iterable of variables to retrieve from the dataset. Default is ("pr", "tas", "tasmin", "tasmax").
    period: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (1980, 2005).
    time_frequency: Frequency, optional
        Time frequency of ERA5 data (currently only Frequency.MONTHLY available).
        Defaults to Frequency.MONTHLY, i.e. original ERA5 data is averaged monthly
    aggregation: Aggregation, optional
        Method used to aggregate the data and build the climatology
        (currently only Aggregation.MONTHLY_MEAN available).
        Defaults to Aggregation.MONTHLY_MEAN.
    output_dir: str, optional
        Output directory where the ERA5 climatology will be stored.
        Defaults to "./results/era5".
    ee_project: str | None = None,
        Earth Engine project ID to use for the download.

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    logger.info("Downloading ERA5 data...")

    # Create output directory
    output_dir = _check_output_dir(output_dir, f"./results/{data_product.product_name}")

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)

    # Connect to Earth Engine
    connect_to_ee(ee_project=ee_project)

    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        # First check if the data is already downloaded
        output_file = climatology_filename(
            output_dir, aoi_n, data_product, aggregation, period
        )
        if Path(output_file).is_file():
            logger.warning(
                """File %s already exists, skipping...
                If this is not the expected behaviour, please remove the file and run the function again.""",
                output_file,
            )
            continue
        ds = _get_era5_area_period(
            aoi_b, aoi_n, variable, period, time_frequency, aggregation
        )
        ds.to_netcdf(output_file)

        save_grid_file(output_dir, data_product, aoi_n, ds)

from __future__ import annotations

from dataclasses import asdict
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
    VariableAttributes,
    check_output_dir,
    climatology_filename,
    get_monthly_climatology,
    get_monthly_mean,
    prep_dataset,
    save_grid_file,
    split_period,
)

logger = get_logger(__name__)

def _get_chirps_area_period(
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
    logger.info('Getting CHIRPS data for period : "%s" and area of interest : "%s"', period, aoi_name)
    dmin, dmax = split_period(period)

    ic = ee.ImageCollection(DataProduct.CHIRPS.url).filterDate(dmin, dmax) # type: ignore
    geom = ee.Geometry.Rectangle(*aoi_bounds.to_numpy()[0]) # type: ignore
    if ic.size().getInfo() == 0:
        msg = f"""
                No data found for the period {dmin} - {dmax}.
                CHIRPS dataset is available from {DataProduct.CHIRPS.period[0]} to {DataProduct.CHIRPS.period[1]}.
                """
        logger.error(msg)
        raise ValueError(msg)

    ds = xr.open_mfdataset(
        [ic], engine="ee", projection=ic.first().select(0).projection(), geometry=geom
    )

    ds = ds.transpose("time", "lat", "lon")
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds = ds.rio.write_crs("epsg:4362")
    ds = ds.rename({"precipitation": "pr"})
    ds = prep_dataset(ds, DataProduct.CHIRPS)
    ds.pr.attrs = asdict(VariableAttributes["pr"])

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


def get_chirps(
    aoi: list[gpd.GeoDataFrame],
    period: tuple[int, int] = (1980, 2005),
    time_frequency: Frequency = Frequency.MONTHLY, # type: ignore[assignment]
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN, # type: ignore[assignment]
    output_dir: str | None = None,
    ee_project: str | None = None,
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
    ee_project: str | None = None,
        Earth Engine project ID to use for the download.

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    logger.info("Downloading CHIRPS data...")

    data_product = DataProduct.CHIRPS

    # Create output directory
    output_dir = check_output_dir(output_dir, f"./results/{data_product.product_name}")

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)

    # Connect to Earth Engine
    connect_to_ee(ee_project=ee_project)

    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        # First check if the data is already downloaded
        output_file = climatology_filename(output_dir, aoi_n, data_product, aggregation, period)
        if Path(output_file).is_file():
            logger.warning(
                """File %s already exists, skipping...
                If this is not the expected behaviour, please remove the file and run the function again.""", output_file
            )
            continue
        ds = _get_chirps_area_period(
            aoi_b, aoi_n, period, time_frequency, aggregation
        )
        ds.to_netcdf(output_file)

        save_grid_file(output_dir, data_product, aoi_n, ds)

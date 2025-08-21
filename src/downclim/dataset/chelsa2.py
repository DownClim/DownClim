from __future__ import annotations

import datetime
import shutil
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import geopandas as gpd
import multiprocess as mp
import pandas as pd
import rioxarray as rio
import xarray as xr

from ..aoi import get_aoi_informations
from ..logging_config import get_logger
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    VariableAttributes,
    check_output_dir,
    climatology_filename,
    get_monthly_climatology,
    prep_dataset,
    save_grid_file,
    split_period,
)

logger = get_logger(__name__)


def _get_chelsa2_one_file(
    aoi_name: list[str],
    aoi_bound: list[pd.DataFrame],
    variable: str,
    month: int,
    year: int,
    time_freq: Frequency,
) -> dict[str, xr.Dataset]:
    """
    Internal function to get CHELSA data for a given year, a given month and given variables, over a list of areas of interest.

    Returns
    -------
    dict:
        Dictionary:
        - key: area of interest name
        - value: xr.Dataset of CHELSA data for the given year, month and variable.

    """

    chelsa_files = {}

    if time_freq == Frequency.MONTHLY:
        url = f"{DataProduct.CHELSA.url}/monthly/{variable}/CHELSA_{variable}_{month:02d}_{year}_V.2.1.tif"
    else:
        msg = "Only monthly time frequency is available for retrieving CHELSA data."
        raise ValueError(msg)

    try:
        urlopen(url)
    except URLError as e:
        msg = (
            f"Page {url} not found, please check the name of the variable or the year."
        )
        raise URLError(msg) from e

    ##ds = rio.open_rasterio(url).to_dataset("band").rename_vars({1: variable})
    ##for aoi_n, aoi_b in zip(aoi_name, aoi_bound, strict=False):
    ##    chelsa_files[aoi_n] = ds.rio.clip_box(*aoi_b.to_numpy()[0]).assign_coords(time=datetime.datetime(year,month,1))

    with rio.open_rasterio(url) as ds_rio:
        for aoi_n, aoi_b in zip(aoi_name, aoi_bound, strict=False):
            chelsa_files[aoi_n] = ds_rio.to_dataset("band").rename_vars({1: variable}).rio.clip_box(*aoi_b.to_numpy()[0]).assign_coords(time=datetime.datetime(year,month,1))

    return chelsa_files

def _get_chelsa2_year_var(
    aoi_name: list[str],
    aoi_bound: list[pd.DataFrame],
    year: int,
    variable: str,
    time_freq: Frequency,
    tmp_directory: str,
    chunks: dict[str, int] | None = None,
) -> dict[str, str]:
    """
    Get CHELSA data for one given year and one given variables.

    Returns
    -------
    dict[str, str]:
        Paths to the downloaded CHELSA files for the given year, variable and area of interest.

    """

    if all(
        Path(f"{tmp_directory}/{DataProduct.CHELSA.product_name}_{aoi_n}_{variable}_{year}.nc").is_file()
        for aoi_n in aoi_name
    ):
        logger.warning("""CHELSA data for year '%s' and variable '%s' already downloaded. Not downloading,
              but the behaviour of the function is not affected.
              If this is not the desired behavior, please remove the file(s) from the temporary folder
              %s and rerun the function.""", year, variable, tmp_directory)
        paths = {
            aoi_n: f"{tmp_directory}/{DataProduct.CHELSA.product_name}_{aoi_n}_{variable}_{year}.nc"
            for aoi_n in aoi_name
        }

    else:
        logger.info(
            'Getting year "%s" for variables "%s" and areas of interest : "%s"', year, variable, aoi_name)
        chelsa_datas = [
            _get_chelsa2_one_file(aoi_name, aoi_bound, variable, month, year, time_freq)
            for month in range(1, 13)
        ]

        paths = {}

        for aoi_n in aoi_name:
            logger.info("Concatenating data for area of interest : %s", aoi_n)
            ds_chelsa = xr.concat([chelsa_data[aoi_n] for chelsa_data in chelsa_datas], dim='time')
            for chelsa_data in chelsa_datas:
                del chelsa_data[aoi_n]
            ds_chelsa = ds_chelsa[["time", "x", "y", variable]]
            ds_chelsa = prep_dataset(ds_chelsa, DataProduct.CHELSA)
            ds_chelsa = ds_chelsa.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

            ds_chelsa[variable].attrs = asdict(VariableAttributes[variable])
            output_file = f"{tmp_directory}/{DataProduct.CHELSA.product_name}_{aoi_n}_{variable}_{year}.nc"
            paths[aoi_n] = output_file
            logger.info("Saving file %s", output_file)
            ds_chelsa.chunk(chunks).to_netcdf(output_file) if chunks else ds_chelsa.to_netcdf(output_file)
            del ds_chelsa
    return paths


def get_chelsa2(
    aoi: list[gpd.GeoDataFrame],
    variable: list[str],
    period: tuple[int, int] = (1980, 2005),
    frequency: Frequency = Frequency.MONTHLY, # type: ignore[assignment]
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN, # type: ignore[assignment]
    nb_threads: int = 4,
    output_dir: str | None = None,
    tmp_dir: str | None = None,
    keep_tmp_dir: bool = False,
) -> None:
    """
    Retrieve CHELSA data for a list of regions, and a list of variables and a given period. This returns one monthly climatological
    xarray.Dataset object / netcdf file with all variables included for each region.

    Note: CHELSA data is available from 1980 to 2019.

    Parameters
    ----------
    aoi: list[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects. You can use
        the `get_aois` function to retrieve them from various inputs types.
    variable: list[str]
        List of variables to collect. For CHELSAv2, choose in :
            "clt", "cmi", "hurs", "pet", "pr", "rsds", "sfcWind",
            "tas", "tasmin", "tasmax", "vpd"
    period: tuple[(int, int)]
        Tuple of time frame to retrieve, and build the climatologies on.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: (1980, 2005).
    time_frequency: Frequency
        Time frequency of Chelsa data (currently only monthly available).
    aggregation: Aggregation
        Aggregation method to build the climatology. Default is "monthly-means".
    nb_threads: int
        Number of threads to use for parallel downloading.
    output_dir: str
        Output directory where the CHELSA2 climatology will be stored.
        Default is "./results/chelsa2".
    tmp_dir: str
        Temporary directory where the intermediate CHELSA files will be stored.
        Default is "./results/tmp/chelsa2".
    keep_tmp_dir: bool
        Whether to keep the temporary directory or not. This includes the intermediate CHELSA files
        downloaded for each area of interest, each variable and each year.
        Warning: this can represent a large amount of data.

    Returns
    -------
    No output from the function. New file with dataset is stored in the output_dir.
    """

    logger.info("Downloading CHELSA data...")

    data_product = DataProduct.CHELSA

    # Create output and tmp directories
    output_dir = check_output_dir(output_dir, f"./results/{data_product.product_name}")
    tmp_dir = check_output_dir(tmp_dir, f"./results/tmp/{data_product.product_name}")

    # Get AOIs information
    aoi_name, aoi_bound = get_aoi_informations(aoi)

    # Get years to retrieve
    years = list(range(period[0], period[1] + 1))

    # Specific case for CHELSA "pr" data
    if "pr" in variable and any(year in [2013, 2016] for year in years):
        logger.warning(
            "CHELSA data for years 2013 and 2016 is not available for precipitation (file is corrupted). \
                      We will not use these years for computing climatology."
        )
        years.remove(2013)  # issue with the tif
        years.remove(2016)  # issue with the tif

    logger.info("Downloading CHELSA data...")

    # We first need to check if the files exist before downloading and processing data
    # We update the list of AOIs to only include those that need to be processed
    for aoi_n in aoi_name:
        output_filename = climatology_filename(output_dir, aoi_n, data_product, aggregation, period)
        if Path(output_filename).is_file():
            msg = f"""
            File for area {aoi_n} and period {period} already exists: {output_filename}.
            No action is done for {aoi_n}.
            Please make sure this is the expected behaviour.
            Continue..."""
            logger.warning(msg)
            aoi_name.remove(aoi_n)

    # Actual data retrieval with update aois if needed
    if aoi_name:
        paths = []
        with mp.Pool(nb_threads) as pool:
            for var in variable:
                paths.append(
                    pool.starmap_async(
                    #pool.starmap(
                        _get_chelsa2_year_var,
                        [
                            (aoi_name, aoi_bound, year, var, frequency, tmp_dir)
                            for year in years
                        ],
                    ).get()
                )
        paths2 = {
            aoi_n: [p[aoi_n] for path in paths for p in path] for aoi_n in aoi_name
        }
        del paths

        logger.info("Merging files by aoi...")
        # Merge files to get one file per aoi for the period
        for aoi_n in aoi_name:
            output_filename = climatology_filename(output_dir, aoi_n, data_product, aggregation, period)
            logger.info("Merging files for area %s...", aoi_n)
            ds_aoi_period: xr.Dataset = xr.open_mfdataset(
                paths2[aoi_n], decode_coords="all", parallel=True
            )
            dmin, dmax = split_period(period)
            if aggregation == Aggregation.MONTHLY_MEAN:
                ds_aoi_period_clim = get_monthly_climatology(
                    ds_aoi_period.sel(time=slice(dmin, dmax))
                )
            else:
                msg = "Currently only monthly-means aggregation available!"
                raise ValueError(msg)
            ds_aoi_period_clim = ds_aoi_period_clim.\
                rename({data_product.lon_lat_names['lon']:'lon', data_product.lon_lat_names['lat']:'lat'})
            ds_aoi_period_clim.to_netcdf(output_filename)

            save_grid_file(output_dir, data_product, aoi_n, ds_aoi_period_clim)

    if not keep_tmp_dir:
        shutil.rmtree(tmp_dir)

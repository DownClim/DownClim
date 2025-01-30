from __future__ import annotations

import datetime
import shutil
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
import rioxarray as rio
import xarray as xr
from multiprocess import Pool

from .aoi import get_aoi_informations
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    get_monthly_climatology,
    split_period,
    variables_attributes,
)


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
    base_url = DataProduct.CHELSA.url

    if time_freq == Frequency.MONTHLY:
        url = f"{base_url}/monthly/{variable}/CHELSA_{variable}_{month:02d}_{year}_V.2.1.tif"
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

    ds = rio.open_rasterio(url).to_dataset("band").rename_vars({1: variable})
    for aoi_n, aoi_b in zip(aoi_name, aoi_bound, strict=False):
        chelsa_files[aoi_n] = ds.rio.clip_box(*aoi_b.to_numpy()[0])
    return chelsa_files


def _get_chelsa2_year(
    aoi_name: list[str],
    aoi_bound: list[pd.DataFrame],
    year: int,
    variable: str,
    time_freq: Frequency,
    tmp_directory: str,
    chunks: dict[str, int] | None = None,
) -> dict[str, str]:
    """
    Get CHELSA data for a given year and given variables.

    Returns
    -------
    dict:

    """

    if all(
        Path(f"{tmp_directory}/CHELSA_{aoi_name}_{variable}_{year}.nc").is_file()
        for aoi_name in aoi_name
    ):
        print(f"""CHELSA data for year '{year}' and variable '{variable}' already downloaded. Not downloading,
              but the behaviour of the function is not affected.
              If this is not the desired behavior, please remove the file(s) from the temporary folder
              {tmp_directory} and rerun the function.""")
        paths = {
            aoi_name: f"{tmp_directory}/CHELSA_{aoi_name}_{variable}_{year}.nc"
            for aoi_name in aoi_name
        }

    else:
        print(
            f'Getting year "{year}" for variables "{variable}" and areas of interest : "{aoi_name}"'
        )
        chelsa_datas = [
            _get_chelsa2_one_file(aoi_name, aoi_bound, variable, month, year, time_freq)
            for month in range(1, 13)
        ]

        paths = {}

        time_index = pd.Index(
            pd.date_range(datetime.datetime(year, 1, 1), periods=12, freq="M"),
            name="time",
        )

        for aoi_n in aoi_name:
            ds_chelsa = xr.concat(
                [chelsa_data[aoi_n] for chelsa_data in chelsa_datas], time_index
            )
            for chelsa_data in chelsa_datas:
                del chelsa_data[aoi_n]
            ds_chelsa = ds_chelsa[["time", "x", "y", variable]]
            ds_chelsa[variable].values = (
                ds_chelsa[variable].to_numpy()
                * DataProduct.CHELSA.scale_factor[variable]
                + DataProduct.CHELSA.add_offset[variable]
            )
            ds_chelsa[variable].attrs = variables_attributes[variable]
            output_file = f"{tmp_directory}/CHELSA_{aoi_n}_{variable}_{year}.nc"
            paths[aoi_n] = output_file
            ds_chelsa.chunk(chunks).to_netcdf(output_file)
            del ds_chelsa
    return paths


def get_chelsa2(
    aoi: list[gpd.GeoDataFrame],
    variable: list[str],
    period: tuple[int, int] = (1980, 2005),
    frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    nb_threads: int = 4,
    output_dir: str = "./results/chelsa2",
    tmp_dir: str = "./results/tmp/chelsa2",
    keep_tmp_dir: bool = False,
) -> None:
    """
    Retrieve CHELSA data for a list of regions, variables and years. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

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

    # Create directories
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, aoi_bound = get_aoi_informations(aoi)

    # Get years to retrieve
    #    periods = [baseline_year, evaluation_year]
    #    years = set().union(*[list(range(period[0], period[1] + 1)) for period in periods])
    years = list(range(period[0], period[1] + 1))

    # Specific case for CHELSA "pr" data
    if "pr" in variable and any(year in [2013, 2016] for year in years):
        warnings.warn(
            "CHELSA data for years 2013 and 2016 is not available for precipitation (file is corrupted). \
                      We will not use these years for computing climatology.",
            stacklevel=1,
        )
        years.remove(2013)  # issue with the tif
        years.remove(2016)  # issue with the tif

    print("Downloading CHELSA data...")
    # Actual data retrieval
    pool = Pool(nb_threads)
    paths = []
    for var in variable:
        paths.append(
            #            pool.starmap_async(
            pool.starmap(
                _get_chelsa2_year,
                [
                    (aoi_name, aoi_bound, year, var, frequency, tmp_dir)
                    for year in years
                ],
            ).get()
        )
    pool.close()
    paths2 = {
        aoi_name: [p[aoi_name] for path in paths for p in path] for aoi_name in aoi_name
    }
    del paths

    print("Merging files...")
    # Merge files to get one file per aoi for all periods
    for aoi_n in aoi_name:
        # We first need to check if the files exist before processing data
        output_filename = f"{output_dir}/{aoi_n}_chelsa2_{aggregation.value}_{period[0]}-{period[1]}.nc"
        if not Path(output_filename).is_file():
            # if not all files for aoi_n exist, we need to process the data
            print("Merging files for area " + aoi_n + "...")
            ds_aoi_period = xr.open_mfdataset(
                paths2[aoi_n], decode_coords="all", parallel=True
            )
            dmin, dmax = split_period(period)
            ds_aoi_period_clim = get_monthly_climatology(
                ds_aoi_period.sel(time=slice(dmin, dmax))
            )
            ds_aoi_period_clim.to_netcdf(output_filename)
        else:
            print(
                f"""File for area {aoi_n} and period {period} already exists. Not downloading.
                Please make sure this is the expected behaviour"""
            )

    if not keep_tmp_dir:
        shutil.rmtree(tmp_dir)

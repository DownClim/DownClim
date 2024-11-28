from __future__ import annotations

import datetime
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import geopandas as gpd
import multiprocess as mp
import pandas as pd
import rioxarray as rio
import xarray as xr

from .connectors import data_urls
from .get_aoi import get_aois_informations
from .utils import (
    Frequency,
    add_offsets,
    get_monthly_climatology,
    scale_factors,
    split_period,
    variables_attributes,
)


def _get_chelsa_one_file(
    aois_names: list[str],
    aois_bounds: list[pd.DataFrame],
    var: str,
    month: int,
    year: int,
    time_freq: Frequency,
) -> dict:
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
    base_url = data_urls["chelsa"]

    if time_freq == Frequency.MONTHLY:
        url = f"{base_url}/monthly/{var}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
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

    ds = rio.open_rasterio(url).to_dataset("band").rename_vars({1: var})
    for aoi_name, aoi_bounds in zip(aois_names, aois_bounds, strict=False):
        chelsa_files[aoi_name] = ds.rio.clip_box(
            minx=aoi_bounds.minx[0],
            miny=aoi_bounds.miny[0],
            maxx=aoi_bounds.maxx[0],
            maxy=aoi_bounds.maxy[0],
        )
    return chelsa_files


def _get_chelsa_year(
    aois_names: list[str],
    aois_bounds: list[pd.DataFrame],
    year: int,
    var: str,
    time_freq: Frequency,
    tmp_directory: str,
    chunks: dict | None = None,
) -> dict:
    """
    Get CHELSA data for a given year and given variables.
    """

    if all(
        Path(f"{tmp_directory}/CHELSA_{aoi_name}_{var}_{year}.nc").is_file()
        for aoi_name in aois_names
    ):
        print(f"""CHELSA data for year '{year}' and variable '{var}' already downloaded. Not downloading,
              but the behaviour of the function is not affected.
              If this is not the desired behavior, please remove the file(s) from the temporary folder
              {tmp_directory} and rerun the function.""")
        paths = {
            aoi_name: f"{tmp_directory}/CHELSA_{aoi_name}_{var}_{year}.nc"
            for aoi_name in aois_names
        }

    else:
        print(
            f'Getting year "{year}" for variables "{var}" and areas of interest : "{aois_names}"'
        )
        chelsa_datas = [
            _get_chelsa_one_file(aois_names, aois_bounds, var, month, year, time_freq)
            for month in range(1, 13)
        ]

        paths = {}

        time_index = pd.Index(
            pd.date_range(datetime.datetime(year, 1, 1), periods=12, freq="M"),
            name="time",
        )

        for aoi_name in aois_names:
            ds_chelsa = xr.concat(
                [chelsa_data[aoi_name] for chelsa_data in chelsa_datas], time_index
            )
            for chelsa_data in chelsa_datas:
                del chelsa_data[aoi_name]
            ds_chelsa = ds_chelsa[["time", "x", "y", var]]
            ds_chelsa[var].values = (
                ds_chelsa[var].to_numpy() * scale_factors["chelsa2"][var]
                + add_offsets["chelsa2"][var]
            )
            ds_chelsa[var].attrs = variables_attributes[var]
            output_file = f"{tmp_directory}/CHELSA_{aoi_name}_{var}_{year}.nc"
            paths[aoi_name] = output_file
            ds_chelsa.chunk(chunks).to_netcdf(output_file)
            del ds_chelsa
    return paths


def get_chelsa(
    aois: Iterable[gpd.GeoDataFrame],
    variables: Iterable[str],
    periods: tuple[(int, int)] = ((1980, 2005), (2006, 2019)),
    time_frequency: str = "mon",
    downscaling_aggregation: str = "monthly-means",
    nb_threads: int = 4,
    keep_tmp_directory: bool = False,
) -> None:
    """
    Retrieve CHELSA data for a list of regions, variables and years. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Note: CHELSA data is available from 1980 to 2019.

    Parameters
    ----------
    aois: list[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects. You can use
        the `get_aois` function to retrieve them from various inputs types.
    variables: list[str]
        List of variables to collect. For CHELSAv2, choose in :
            "clt", "cmi", "hurs", "pet", "pr", "rsds", "sfcWind",
            "tas", "tasmin", "tasmax", "vpd"
    periods: tuple[(int, int)]
        Tuple of time frames to retrieve, and build the climatologies on.
        Should correspond to the historical period and the evaluation period.
        Must be provided as a list of pairs of integers defining the start and end years of the period.
        e.g.: [(1980, 2005), (2006,2019)].
    time_frequency: str
        Time frequency of Chelsa data (currently only "mon" available).
    nb_threads: int
        Number of threads to use for parallel downloading.
    keep_tmp_directory: bool
        Whether to keep the temporary directory or not. This includes the intermediate CHELSA files
        downloaded for each area of interest, each variable and each year.
        Tmp directory is located at "./results/tmp/chelsa".
        Warning: this can represent a large amount of data.
    """

    # Create directories
    tmp_directory = "./results/tmp/chelsa2"
    output_directory = "./results/chelsa2"
    Path(tmp_directory).mkdir(parents=True, exist_ok=True)
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, aois_bounds = get_aois_informations(aois)

    # Get years to retrieve
    years = set().union(*[list(range(period[0], period[1] + 1)) for period in periods])

    # Specific case for CHELSA "pr" data
    if "pr" in variables and any(year in [2013, 2016] for year in years):
        warnings.warn(
            "CHELSA data for years 2013 and 2016 is not available for precipitation (file is corrupted). \
                      We will not use these years for computing climatology.",
            stacklevel=1,
        )
        years.remove(2013)  # issue with the tif
        years.remove(2016)  # issue with the tif

    # Get time frequency
    time_frequency = Frequency(time_frequency)

    # Actual data retrieval
    pool = mp.Pool(nb_threads)
    paths = []
    for var in variables:
        paths.append(
            pool.starmap_async(
                _get_chelsa_year,
                [
                    (aois_names, aois_bounds, year, var, time_frequency, tmp_directory)
                    for year in years
                ],
            ).get()
        )
    pool.close()
    paths2 = {
        aoi_name: [p[aoi_name] for path in paths for p in path]
        for aoi_name in aois_names
    }
    del paths

    # Merge files to get one file per aoi for all periods
    for aoi_name in aois_names:
        print("Merging files for area " + aoi_name + "...")
        ds_aoi_period = xr.open_mfdataset(
            paths2[aoi_name], decode_coords="all", parallel=True
        )
        for period in periods:
            dmin, dmax = split_period(period)
            ds_aoi_period_clim = get_monthly_climatology(
                ds_aoi_period.sel(time=slice(dmin, dmax))
            )
            ds_aoi_period_clim.to_netcdf(
                f"{output_directory}/{aoi_name}_chelsa2_{downscaling_aggregation}_{period[0]}-{period[1]}.nc"
            )

    if not keep_tmp_directory:
        shutil.rmtree(tmp_directory)

from __future__ import annotations

import datetime
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import geopandas
import multiprocess as mp
import pandas as pd
import rioxarray as rio
import xarray as xr

from .connectors import chelsa_url
from .get_aoi import get_aois_informations
from .utils import (
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
    time_freq: str = "monthly",
    base_url: str = chelsa_url,
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
    url = f"{base_url}{time_freq}/{var}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    try:
        urlopen(url)
    except URLError as e:
        msg = (
            f"Page {url} not found, please check the name of the variable or the year."
        )
        raise Exception(msg) from e
    ds = (
        rio.open_rasterio(url, decode_coords="all")
        .to_dataset("band")
        .rename_vars({1: var})
    )
    for aoi_name, aoi_bounds in zip(aois_names, aois_bounds, strict=False):
        chelsa_files[aoi_name] = ds.rio.clip_box(
            minx=aoi_bounds.minx[0],
            miny=aoi_bounds.miny[0],
            maxx=aoi_bounds.maxx[0],
            maxy=aoi_bounds.maxy[0],
        )
    return chelsa_files


def get_chelsa_year(
    aois: list[geopandas.GeoDataFrame],
    year: int,
    var: str,
    time_freq: str = "monthly",
    temp_fold: str = "./results/chelsa/",
    chunks: dict | None = None,
) -> dict:
    """
    Get CHELSA data for a given year and given variables.
    """

    Path.mkdir(temp_fold)

    aois_names, aois_bounds = get_aois_informations(aois)
    print(f"Getting year {year} for variables {var} and areas of interest {aois_names}")
    chelsa_datas = [
        _get_chelsa_one_file(
            aois_names, aois_bounds, var, month, year, chelsa_url, time_freq
        )
        for month in range(1, 13)
    ]

    paths = {}
    time_index = pd.Index(
        pd.date_range(datetime.datetime(year, 1, 1), periods=12, freq="M"), name="time"
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
        paths[aoi_name] = f"{temp_fold}/CHELSA_{aoi_name}_{var}_{year}.nc"
        ds_chelsa.chunk(chunks).to_netcdf(paths[aoi_name])
        del ds_chelsa
    return paths


def get_chelsa(
    aois: Iterable[geopandas.GeoDataFrame | pd.DataFrame],
    variables: Iterable[str],
    periods: Iterable[str] = ["1980-2005", "2006-2019"],
    time_frequency: str = "mon",
    nb_threads: int = 4,
    aggregation: str = "monthly-means",
    temp_fold: Path = "./results/chelsa/",
    output_files: Path = "./results/chelsa/",
    aois_names: Iterable[str] | None = None,
) -> xr.Dataset:
    """
    Retrieve CHELSA data for a list of of regions, variables and years. This returns one monthly climatological
    xarray.Dataset object / netcdf file for each region and period.

    Note: CHELSA data is available from 1980 to 2019.

    Parameters
    ----------
    aois: list[geopandas.GeoDataFrame | pd.DataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects or
        as a pandas.DataFrame with bounds [minx, miny, maxx, maxy].
    variables: list[str]
        List of variables to collect.
    periods: list[str]
        List of time frames to retrieve, and build the climatologies on.
        Should correspond to the historical period and the evaluation period.
        Default is ["1980-2005", "2006-2019"].
    time_frequency: str
        Time frequency of Chelsa data (currently only "mon" available).
    nb_threads: int
        Number of threads to use for parallel downloading.
    aggregation: str
        Aggregation method to build the climatology. Default is "monthly-means".
    aois_names: list[str] | None
        List of areas of interest names. If not provided, the function will retrieve them from the geopandas.GeoDataFrame objects.
        Default is None.

    """

    Path.mkdir(temp_fold)

    if isinstance(aois[0], geopandas.GeoDataFrame):
        aois_names, _ = get_aois_informations(aois)
    elif isinstance(aois[0], pd.DataFrame) and not aois_names:
        msg = "Please provide the names of the areas of interest in `aois_names`."
        raise Exception(msg)

    years = set()
    for period in periods:
        ymin, ymax = period.split("-")
        years_period = set(range(int(ymin), int(ymax) + 1))
        years = years.union(years_period)
    if "pr" in variables and 2016 in years:
        warnings.warn(
            "CHELSA data for 2016 is not available for precipitation (file is corrupted). \
                      We will not use this year for computing climatology.",
            stacklevel=1,
        )
    if "pr" in variables and 2013 in years:
        warnings.warn(
            "CHELSA data for 2013 is not available for precipitation (file is corrupted). \
                      We will not use this year for computing climatology.",
            stacklevel=1,
        )
        years.remove(2013)  # issue with the tif
        years.remove(2016)  # issue with the tif

    if time_frequency == "mon":
        tf = "monthly"
    else:
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)

    pool = mp.Pool(nb_threads)
    paths = []
    for var in variables:
        paths.append(
            pool.starmap_async(
                get_chelsa_year, [(aois, year, var, tf, temp_fold) for year in years]
            ).get()
        )
    pool.close()
    paths2 = {key: [] for key in aois_names}
    for path in paths:
        for p in path:
            for aoi_name in aois_names:
                paths2[aoi_name].append(p[aoi_name])
    del paths

    for aoi_name in aois_names:
        print("Merging files for area " + aoi_name + "...")
        ds = xr.open_mfdataset(paths2[aoi_name], decode_coords="all", parallel=True)
        for period in periods:
            dmin, dmax = split_period(period)
            ds_a = get_monthly_climatology(ds.sel(time=slice(dmin, dmax)))
            path = (
                f"{output_files[0].parent}/{aoi_name}_chelsa2_{aggregation}_{period}.nc"
            )
            ds_a.to_netcdf(path)

    shutil.rmtree(temp_fold)

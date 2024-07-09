from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import ee
import geopandas
import pandas as pd
import xarray as xr

from .utils import (
    add_offsets,
    connect_to_ee,
    ee_image_collection,
    get_aois_informations,
    get_monthly_climatology,
    get_monthly_mean,
    scale_factors,
    split_period,
    variables_attributes,
)


# funs
def get_var_gshtd(var, leg, period):
    dmin, dmax = split_period(period)
    collection = {"tas": "TMEAN", "tasmin": "TMIN", "tasmax": "TMAX"}
    ic = ee.ImageCollection(ee_image_collection["GSHTD"] + collection[var]).filterDate(
        dmin, dmax
    )
    ds = xr.open_dataset(
        ic, engine="ee", projection=ic.first().select(0).projection(), geometry=leg
    )
    ds = ds.transpose("time", "lat", "lon")
    # ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    # ds = ds.rio.write_crs("epsg:4362")
    ds = get_monthly_mean(ds)
    ds = get_monthly_climatology(ds)
    ds = ds.where(ds.b1 > 0)
    ds["b1"] = ds.b1 * scale_factors["GSHTD"] + add_offsets["GSHTD"]  # K to °C
    ds.b1.attrs = variables_attributes[var]
    return ds.rename({"b1": var})


def get_gshtd(
    bounds: pd.DataFrame, period: str, vars: Iterable[str] = ["tas", "tasmin", "tasmax"]
) -> xr.Dataset:
    """
    Get GSHTD data (https://gee-community-catalog.org/projects/gshtd/) for a given
    area of interest and time period, for a set of variables.

    Parameters
    ----------
    bounds: pd.DataFrame
        Bounds of the area of interest.
    period: str
        Time period to retrieve. e.g. "1980-2005".
    vars: List[str]
        List of variables to retrieve. Default is ["tas", "tasmin", "tasmax"].
    """
    aoi_rectangle = ee.Geometry.Rectangle(
        bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]
    )
    return xr.merge([get_var_gshtd(var, aoi_rectangle, period) for var in vars])


# code
def get_multiple_gshtd(
    aois: Iterable[geopandas.GeoDataFrame],
    periods: Iterable[str],
    nc_files: Iterable[str],
    time_frequency: str = "mon",
    vars: Iterable[str] = ["tas", "tasmin", "tasmax"],
    aggregation: str = "monthly-mean",
) -> None:
    """
    Get GSHTD data (https://gee-community-catalog.org/projects/gshtd/)
    for a given set of areas of interest and time periods, for given variables.
    Sequential calls get_gshtd.

    Parameters
    ----------
    aois: List[geopandas.GeoDataFrame]
        List of areas of interest.
    periods: List[str]
        List of time periods to retrieve. e.g. ["1980-2005", "2006-2019", "2071-2100"].
    time_frequency: str
        Time frequency of the data. Default is "mon" (monthly)
    vars: List[str]
        List of variables to retrieve. Default is ["tas", "tasmin", "tasmax"].
    """
    aois_names, aois_bounds = get_aois_informations(aois)
    if time_frequency != "mon":
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)
    connect_to_ee()
    for aoi_name, aoi_bound in zip(aois_names, aois_bounds, strict=False):
        for _, period in enumerate(periods):
            ds = get_gshtd(aoi_bound, period, vars)
            path = (
                f"{Path(nc_files[0]).parent}/{aoi_name}_gshtd_{aggregation}_{period}.nc"
            )
            ds.to_netcdf(path)

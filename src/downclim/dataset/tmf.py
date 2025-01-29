from __future__ import annotations

import geopandas
import numpy as np
import xarray as xr

from .utils import variables_attributes


def compute_tmf(proj, rsds, pet_pr_max=1, tmin=16, prmin=1000):
    ds = xr.merge([proj, rsds])
    ds["pet"] = 0.0023 * ds.rsds * (ds.tas + 17.8) * pow((ds.tasmax - ds.tasmin), 0.5)
    ds.pet.attrs = variables_attributes["pet"]
    ds_year = ds.mean("month")
    ds_year["pr"] = ds[["pr"]].sum("month").pr
    ds_year["pet"] = ds[["pet"]].sum("month").pet
    ds_year["pet_pr"] = ds_year.pet / ds_year.pr
    ds_year["tmf"] = (
        (ds_year.tas >= tmin).astype(int)
        * (ds_year.pr >= prmin).astype(int)
        * (ds_year.pet_pr <= pet_pr_max).astype(int)
    )
    ds_year = ds_year.where(np.invert(np.isnan(ds_year.tas)), drop=True)
    return ds_year[["tas", "pr", "pet", "pet_pr", "tmf"]]


def get_tmf(
    aoi: geopandas.GeoDataFrame,
    future_file: str,
    present_file: str,
    rsds_file: str,
    out_file: str,
) -> None:
    geometry, crs = aoi.geometry.to_numpy(), aoi.crs
    if not crs:
        crs = "EPSG:4326"
    rsds = xr.open_dataset(rsds_file).rio.clip(geometry, crs)
    present = xr.open_dataset(present_file).rio.clip(geometry, crs)
    future = xr.open_dataset(future_file).rio.clip(geometry, crs)
    present = compute_tmf(present, rsds)
    future = compute_tmf(future, rsds)
    anomalies = future
    anomalies["tmf"] = (future - present).tmf
    anomalies.to_netcdf(out_file)

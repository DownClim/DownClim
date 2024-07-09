from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

from .utils import (
    connect_to_gcfs,
    convert_cf_to_dt,
    get_monthly_climatology,
    split_period,
    variables_attributes,
)


def get_cmip6(
    base_files: Iterable[str],
    areas: Iterable[str],
    institute: str,
    model: str,
    experiment: str = "ssp126",
    ensemble: str = "r1i1p1f1",
    baseline: str = "chelsa2",
    variables: Iterable[str] = ["pr", "tas", "tasmin", "tasmax"],
    time_frequency: str = "mon",
    periods: Iterable[str] = ["1980-2005", "2006-2019", "2071-2100"],
    aggregation: str = "monthly-means",
) -> None:
    """
    Get CMIP6 data for a given region and period.

    Parameters
    ----------
    model: str
        Model.
    experiment: str
        Experiment.
    ensemble: str
        Ensemble.
    baseline: str
        Baseline.
    variables: list
        List of variables.
    time_frequency: str
        Time frequency.
    check_file: str
        Check file.
    threads: int
        Number of threads.
    periods: list
        List of periods.
    aggregation: str
        Aggregation.
    """

    # conversions
    if time_frequency == "mon":
        table_id = "Amon"
    else:
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)

    gcs, df = connect_to_gcfs()
    a = []
    for exp in ["historical", experiment]:
        activity = "CMIP" if exp == "historical" else "ScenarioMIP"
        for var in variables:
            search_string = f"activity_id == {activity} & table_id == {table_id} & variable_id == {var} & experiment_id == {exp} & institution_id == {institute} & source_id == {model} & member_id == {ensemble}"
            df_ta = df.query(search_string)
            zstore = df_ta.zstore.to_numpy()[-1]
            mapper = gcs.get_mapper(zstore)
            a.append(xr.open_zarr(mapper, consolidated=True))
    ds = xr.merge(a)
    ds["time"] = np.sort(ds["time"].values)
    dmin = min(x.split("-")[0] for x in periods) + "-01-01"
    dmax = max(x.split("-")[1] for x in periods) + "-01-01"
    ds = ds.sel(time=slice(dmin, dmax))
    ds = ds.chunk(chunks={"time": 100, "lat": 400, "lon": 400})
    cf = type(ds["time"].to_numpy()[0]) is not np.datetime64
    if cf:
        ds["time"] = [*map(convert_cf_to_dt, ds.time.values)]
    if "pr" in list(ds.keys()):
        ds["pr"] = ds.pr * 60 * 60 * 24 * ds.time.dt.days_in_month  # s-1 to month-1
        ds.pr.attrs = variables_attributes["pr"]
    if "tas" in list(ds.keys()):
        ds["tas"] = ds.tas - 273.15  # K to °C
        ds.tas.attrs = variables_attributes["tas"]
    if "tasmin" in list(ds.keys()):
        ds["tasmin"] = ds.tasmin - 273.15  # K to °C
        ds.tasmin.attrs = variables_attributes["tasmin"]
    if "tasmax" in list(ds.keys()):
        ds["tasmax"] = ds.tasmax - 273.15  # K to °C
        ds.tasmax.attrs = variables_attributes["tasmax"]

    check_file = "toto.txt"
    for period in periods:
        dmin, dmax = split_period(period)
        ds_a = get_monthly_climatology(ds.sel(time=slice(dmin, dmax)))
        for i in list(range(len(areas))):
            base = xr.open_dataset(base_files[i])
            regridder = xe.Regridder(ds_a, base, "bilinear")
            ds_r = regridder(ds_a, keep_attrs=True)
            path = f"{Path(check_file).parent}/{areas[i]}_CMIP6_global_{institute}_{model}_{experiment}_{ensemble}_none_none_{baseline}_{aggregation}_{period}.nc"
            ds_r.to_netcdf(path)

    with Path.open(check_file, "a") as f:
        f.write("Done.")

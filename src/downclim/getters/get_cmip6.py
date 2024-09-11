from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import xesmf as xe

from .connectors import connect_to_gcfs
from .get_aoi import get_aois_informations
from .utils import (
    Frequency,
    get_frequency,
    get_monthly_climatology,
    prep_dataset,
    split_period,
)


def get_cmip6(
    aois: Iterable[gpd.GeoDataFrame],
    variables: Iterable[str] = ["pr", "tas", "tasmin", "tasmax"],
    periods: tuple[(int, int)] = ((1980, 2005), (2006, 2019), (2071, 2100)),
    time_frequency: str = "mon",
    aggregation: str = "monthly-means",
    institute: str = "IPSL",
    model: str = "IPSL-CM6A-LR",
    experiment: str = "ssp126",
    ensemble: str = "r1i1p1f1",
    baseline: str = "chelsa2",
) -> None:
    """
    Get CMIP6 data for given regions, variables and periods. Uses google cloud storage to retrieve data.
    It also regrids the data to the given baseline dataset.


    Parameters
    ----------
    aois: list[geopandas.GeoDataFrame]
        List of areas of interest, defined as geopandas.GeoDataFrame objects. You can use
        the `get_aois` function to retrieve them from various inputs types.
    variables: list
        List of variables to collect.
    periods: list
        List of periods. Each period is a tuple of two integers. It should correspond to the historical period,
        evaluation period and the projection period. e.g. ((1980, 2005), (2006, 2019), (2071, 2100)).
        These periods must match the periods of the baseline dataset.
    time_frequency: str
        Time frequency of Chelsa data (currently only "mon" available).
    aggregation: str
        Aggregation method to build the climatology. Default is "monthly-means".
    institute: str
        Name of the institute that produced the CMIP6 data. e.g. "IPSL".
    model: str
        Name of the model that has been run by the `institute` to produce the data. e.g. "IPSL-CM6A-LR".
    experiment: str
        Name of the experiment, which is typically the name of the scenario used for future projections, e.g. "ssp126", "ssp585"...
        "historical" is automatically added and used for the historical period.
    ensemble: str
        Name of the ensemble run of the data. e.g. "r1i1p1f1".
    baseline: str
        Baseline dataset used for regridding. e.g. "chelsa2".
    """

    output_directory = "./results/cmip6"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    aois_names, _ = get_aois_informations(aois)

    # conversions
    time_frequency = get_frequency(time_frequency)
    if time_frequency == Frequency.MONTHLY:
        table_id = "Amon"
    else:
        msg = "Currently only monthly time frequency available!"
        raise Exception(msg)

    gcs, df = connect_to_gcfs()
    a = []
    for exp in ["historical", experiment]:
        activity = "CMIP" if exp == "historical" else "ScenarioMIP"
        for var in variables:
            search_string = f"""activity_id == '{activity}' & table_id == '{table_id}' & variable_id == '{var}' & experiment_id == '{exp}' & institution_id == '{institute}' & source_id == '{model}' & member_id == '{ensemble}'"""
            df_ta = df.query(search_string)
            zstore = df_ta.zstore.to_numpy()[-1]
            mapper = gcs.get_mapper(zstore)
            a.append(xr.open_zarr(mapper, consolidated=True))
    ds = xr.merge(a)
    ds["time"] = np.sort(ds["time"].values)

    dminsmaxs = [split_period(period) for period in periods]
    dmin = min(dminsmaxs, key=lambda x: x[0])[0]
    dmax = max(dminsmaxs, key=lambda x: x[1])[1]
    ds = ds.sel(time=slice(dmin, dmax))
    ds = ds.chunk(chunks={"time": 100, "lat": 400, "lon": 400})
    ds = prep_dataset(ds, "cmip6")

    for i, period in enumerate(periods):
        dmin, dmax = dminsmaxs[i]
        ds_clim = get_monthly_climatology(ds.sel(time=slice(dmin, dmax)))
        for aoi_name in aois_names:
            baseline_file = (
                f"./results/{baseline}/{aoi_name}_chelsa2_{aggregation}_*.nc"
            )
            base = xr.open_dataset(baseline_file[0])
            regridder = xe.Regridder(ds_clim, base, "bilinear")
            ds_r = regridder(ds_clim, keep_attrs=True)
            output_file = f"{output_directory}/{aoi_name}_CMIP6_global_{institute}_{model}_{experiment}_{ensemble}_none_none_{baseline}_{aggregation}_{period[0]}-{period[1]}.nc"
            ds_r.to_netcdf(output_file)

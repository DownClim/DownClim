from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from pydantic import BaseModel, Field, field_validator

from .aoi import get_aoi_informations
from .connectors import connect_to_gcfs
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    get_monthly_climatology,
    prep_dataset,
    split_period,
)


class CMIP6Context(BaseModel):
    """Context about the query on the CMIP6 dataset.

    Entries of the dictionary can be either `str` or `list` of `str` if multiple values are provided. These following keys are available. None are mandatory:
    - activity_id: str, e.g "ScenarioMIP", "CMIP"
    - institution_id: str, e.g "IPSL", "NCAR"
    - source_id: str, e.g "IPSL-CM6A-LR", "CMCC-CM2-HR4"
    - experiment_id: str, e.g "ssp126", "historical"
    - member_id: str, e.g "r1i1p1f1"
    - table_id: str, e.g "Amon", "day"
    - variable_id: str, e.g "tas", "pr"
    - grid_label: str, e.g "gn", "gr"
    """

    activity_id: str | list[str] | None = Field(
        default=["ScenarioMIP", "CMIP"], description="Name of the CMIP6 activity"
    )
    institution_id: str | list[str] | None = Field(
        default=["IPSL", "NCAR"], description="Institute name that produced the data"
    )
    source_id: str | list[str] | None = Field(
        default=None, description="Global climate model name"
    )
    experiment_id: str | list[str] | None = Field(
        default=["ssp245", "historical"],
        description="Name of the experiment type of the simulation",
    )
    member_id: str | list[str] | None = Field(
        default="r1i1p1f1", description="Ensemble member"
    )
    table_id: str | None = Field(default="Amon", description="CMIP6 table name")
    variable_id: str | list[str] | None = Field(
        default=["tas", "tasmin", "tasmax", "pr"], description="Variables name"
    )
    grid_label: str | None = Field(default=None, description="Grid label")

    @field_validator("experiment_id", mode="before")
    @classmethod
    def validate_experiment_id(cls, v: str | list[str] | None) -> list[str]:
        if not any(exp == "historical" for exp in v):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
            if isinstance(v, str):
                return [v, "historical"]
            return [*v, "historical"]
        return v


def _get_cmip6_catalog(
    url: str,
) -> pd.DataFrame:
    """
    Get CMIP6 catalog from ESGF.

    Parameters
    ----------
    url: str
        URL of the CMIP6 catalog, on csv format.

    Returns
    -------
    pd.DataFrame
        CMIP6 catalog.
    """
    return pd.read_csv(url)


def get_cmip6(
    aoi: Iterable[gpd.GeoDataFrame],
    variable: Iterable[str] = ("pr", "tas", "tasmin", "tasmax"),
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    projection_year: tuple[int, int] = (2071, 2100),
    time_frequency: Frequency = Frequency.MONTHLY,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    activity: str | Iterable[str] = ("ScenarioMIP", "CMIP"),
    institution: str = "IPSL",
    source: str = "IPSL-CM6A-LR",
    experiment: str | Iterable[str] = "ssp126",
    member: str = "r1i1p1f1",
    grid_label: str = "gn",
    baseline: str = "chelsa2",
    chunks: dict[str, int] | None = None,
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

    if chunks is None:
        chunks = {"time": 100, "lat": 400, "lon": 400}

    output_directory = "./results/cmip6"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    aoi_name, _ = get_aoi_informations(aoi)

    # conversions
    time_frequency = Frequency(time_frequency)
    if time_frequency == Frequency.MONTHLY:
        table = "Amon"
    else:
        msg = "Currently only monthly time frequency available!"
        raise ValueError(msg)

    gcfs = connect_to_gcfs()
    df_cmip6 = _get_cmip6_catalog(DataProduct.CMIP6.url)
    a = []

    search_string = f"""
        activity_id == '{activity}' &
        experiment_id == '{experiment}' &
        institution_id == '{institution}' &
        source_id == '{source}' &
        member_id == '{member}' &
        table_id == '{table}' &
        grid_label == '{grid_label}'
        """

    for exp in ["historical", experiment]:
        activity = "CMIP" if exp == "historical" else "ScenarioMIP"
        for _var in variable:
            df_ta = df_cmip6.query("variable_id == '{_var}' &" + search_string)
            zstore = df_ta.zstore.to_numpy()[-1]
            mapper = gcfs.get_mapper(zstore)
            a.append(xr.open_zarr(mapper, consolidated=True))
    ds = xr.merge(a)
    ds["time"] = np.sort(ds["time"].values)

    dminsmaxs = [
        split_period(period)
        for period in [baseline_year, evaluation_year, projection_year]
    ]
    dmin = min(dminsmaxs, key=lambda x: x[0])[0]
    dmax = max(dminsmaxs, key=lambda x: x[1])[1]
    ds = ds.sel(time=slice(dmin, dmax))
    ds = ds.chunk(chunks=chunks)
    ds = prep_dataset(ds, "cmip6")

    for i, period in enumerate([baseline_year, evaluation_year, projection_year]):
        dmin, dmax = dminsmaxs[i]
        ds_clim = get_monthly_climatology(ds.sel(time=slice(dmin, dmax)))
        for aoi_n in aoi_name:
            baseline_file = (
                f"./results/{baseline}/{aoi_n}_chelsa2_{aggregation.value}_*.nc"
            )
            base = xr.open_dataset(baseline_file[0])
            regridder = xe.Regridder(ds_clim, base, "bilinear")
            ds_r = regridder(ds_clim, keep_attrs=True)
            output_file = f"""{output_directory}/{aoi_n}_CMIP6_global_{institution}_{source}_{experiment}_{member}_none_none_{baseline}_{aggregation.value}_{period[0]}-{period[1]}.nc"""
            ds_r.to_netcdf(output_file)

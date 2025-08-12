from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from pydantic import BaseModel, Field, field_validator

from ..aoi import get_aoi_informations
from .connectors import connect_to_gcfs
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    get_monthly_climatology,
    prep_dataset,
    split_period,
)

logger = logging.getLogger(__name__)


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

    project: list[str] | None = Field(
        default=["ScenarioMIP", "CMIP"],
        examples=[["ScenarioMIP", "CMIP"], ("ScenarioMIP", "CMIP")],
        description="Name of the CMIP6 activity",
    )
    institute: list[str] | None = Field(
        default=None,
        examples=[["IPSL", "NCAR"]],
        description="Institute name that produced the data.",
    )
    source: list[str] | None = Field(
        default=None,
        examples=[["IPSL-CM6A-LR", "CMCC-CM2-HR4"]],
        description="Global climate model name",
    )
    experiment: list[str] | None = Field(
        default=["ssp245", "historical"],
        examples=[["ssp245", "historical"]],
        description="Name of the experiment type of the simulation",
    )
    ensemble: list[str] | None = Field(
        default=["r1i1p1f1"],
        examples=["r1i1p1f1", ["r1i1p1f1", "r2i1p1f1"]],
        description="Ensemble member"
    )
    frequency: Frequency = Field(
        default=Frequency.MONTHLY,
        examples=[Frequency.MONTHLY, "mon"], # type: ignore[assignment]
        description="Time frequency of the data",
    )
    variable: list[str] | None = Field(
        default=["tas", "pr"],
        examples=[["tas", "tasmin", "tasmax", "pr"], "tas"],
        description="Variables name",
    )
    grid_label: str | None = Field(default=None, description="Grid label")

    class Config:
        """Pydantic configuration for the DownClimContext class."""

        # arbitrary_types_allowed = True # Whether arbitrary types are allowed for field types.
        extra = "forbid"  # Forbid extra data during model initialization.

    @classmethod
    def to_list(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return [v]
        return v

    @field_validator(
        "experiment",
        "institute",
        "source",
        "variable",
        "project",
        "ensemble",
        mode="before",
    )
    @classmethod
    def validate_list(cls, v: Any) -> list[Any]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, tuple | set):
            if all(isinstance(e, str) for e in v):
                return list(v)
            msg = f"Value {v} is not valid. Please provide a string, a tuple, set or list of string."
            raise ValueError(msg)
        if isinstance(v, list):
            if all(isinstance(e, str) for e in v):
                return v
            msg = f"Value {v} is not valid. Please provide a string, a tuple, set or list of string."
            raise ValueError(msg)
        msg = f"Value {v} is not valid. Please provide a string, a tuple or a list."
        raise ValueError(msg)

    @field_validator("experiment", mode="before")
    @classmethod
    def validate_experiment_id(cls, v: str | Iterable[str] | None) -> list[str]:
        if not v:
            v = []
        if isinstance(v, str):
            v = [v]
        if not any(exp == "historical" for exp in v):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
            return [*v, "historical"]
        return list(v)

    def list_available_simulations(
        self,
        cmip6_catalog_url: str = DataProduct.CMIP6.url
    ) -> pd.DataFrame:
        """List all available CMIP6 simulations available on Google Cloud Storage for a given set of context.

        Parameters
        ----------
        cmip6_catalog_url: str (default: DataProduct.CMIP6.url)
            URL to the CMIP6 catalog on the Google Cloud File System.

        Returns:
        -------
        pd.DataFrame: DataFrame containing information about the available datasets matching
        """

        context = self.model_dump()
        # gcfs connection
        # gcfs_connector = connect_to_gcfs()
        # list CMIP6 datasets matching context
        cmip6_simulations = inspect_cmip6(context, cmip6_catalog_url)
        cmip6_simulations = cmip6_simulations.assign(domain="GLOBAL")
        cmip6_simulations = cmip6_simulations.assign(product="output")

        # filter simulations that don't have all variables requested
        cmip6_simulations = cmip6_simulations.groupby(
            ["source", "experiment", "ensemble"]
        ).filter(lambda x: set(context["variable"]) == (set(x["variable"])))
        # filter simulations that don't have both historical & projection
        cmip6_simulations = cmip6_simulations.groupby(["source", "ensemble"]).filter(
            lambda x: set(context["experiment"]).issubset(set(x["experiment"]))
        )
        if cmip6_simulations.empty:
            msg = "No CMIP6 simulations found for the given context."
            warnings.warn(msg, stacklevel=1)
            return cmip6_simulations
        return cmip6_simulations.reset_index().drop("index", axis=1)


@lru_cache
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

def _get_filename_from_cmip6_context(
    output_dir: str,
    aoi_n: str,
    data_product: DataProduct,
    institute: str,
    source: str,
    experiment: str,
    ensemble: str,
    aggregation: Aggregation,
    tmin: int,
    tmax: int,
) -> str:
    """Internal function. Get the name of the output file given a search context."""
    return f"{output_dir}/{aoi_n}_{data_product.product_name}_{institute}_{source}_{experiment}_{ensemble}_{aggregation.value}_{tmin}_{tmax}.nc"


def get_cmip6_context_from_filename(filename: str) -> dict[str, str]:
    """Get CMIP6 context given a simulation filename.

    Parameters
    ----------
    filename: str
        Filename containing CMIP6 context of the simulation.

    Returns
    -------
    dict[str, str]
        List of main CMIP6 context information, including:
            - output_dir
            - aoi_n
            - data_product name
            - institute
            - source
            - experiment
            - ensemble
            - aggregation
            - tmin
            - tmax

    """
    context_items = ["output_dir", "aoi_n", "data_product", "institute", "source", "experiment", "ensemble", "aggregation", "tmin", "tmax"]
    context_elements = [str(Path(filename).parent), *Path(filename).name.split(".nc")[0].split("_")]
    return dict(zip(context_items, context_elements, strict=False))


def inspect_cmip6(
    context: dict[str, str | Iterable[str]],
    cmip6_catalog_url: str = DataProduct.CMIP6.url,
) -> pd.DataFrame:
    """
    Inspects Google Cloud File System to get information about the available CMIP6 datasets provided the context.

    Parameters
    ----------
    context: dict([str, str | Iterable[str]])
        Dictionary containing information about the query on the CMIP6 dataset. Entries of the dictionary can be
        either `str` or `Iterables` (e.g. `list`) if multiple values are provides.
        These following keys are available (none are mandatory):
            - activity_id: str, e.g "ScenarioMIP", "CMIP"
            - institution_id: str, e.g "IPSL", "NCAR"
            - source_id: str, e.g "IPSL-CM6A-LR", "CMCC-CM2-HR4"
            - experiment_id: str, e.g "ssp126", "historical"
            - member_id: str, e.g "r1i1p1f1"
            - table_id: str, e.g "Amon", "day"
            - variable_id: str, e.g "tas", "pr"
            - grid_label: str, e.g "gn", "gr"
            - zstore: str, e.g "gs://cmip6/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Amon/tas/gr/v20190903"
            - dcpp_init_year: str, e.g "1850", "2015"
            - version: str, e.g "20190903"
    cmip6_catalog_url: str (default: DataProduct.CMIP6.url)
        URL to the CMIP6 catalog on the Google Cloud File System.

    Returns
    -------
    pd.DataFrame: DataFrame containing information about the available datasets matching the query
    """

    # name mapping between context / CMIP6 catalog and output
    cmip6_name_mapping = {
        "project": "activity_id",
        "institute": "institution_id",
        "source": "source_id",
        "experiment": "experiment_id",
        "ensemble": "member_id",
        "frequency": "table_id",
        "variable": "variable_id",
        "grid_label": "grid_label",
    }
    inverse_cmip6_name_mapping = {v: k for k, v in cmip6_name_mapping.items()}
    inverse_cmip6_name_mapping["zstore"] = "datanode"
    inverse_cmip6_name_mapping["table_id"] = "table"

    if context["frequency"] == Frequency.MONTHLY:
        context["frequency"] = "Amon"

    cmip6_catalog = _get_cmip6_catalog(cmip6_catalog_url)

    search_string_parts = []
    for k, v in context.items():
        if v is not None:
            if isinstance(v, str):
                search_string_parts.append(f"{cmip6_name_mapping[k]} == '{v}'")
            else:
                search_string_parts.append(
                    "("
                    + " | ".join([f"{cmip6_name_mapping[k]} == '{w}'" for w in v])
                    + ")"
                )
    search_string = " & ".join(search_string_parts)

    return cmip6_catalog.query(search_string).rename(columns=inverse_cmip6_name_mapping)


def get_cmip6(
    aoi: list[gpd.GeoDataFrame],
    cmip6_simulations: pd.DataFrame,
    baseline_period: tuple[int, int] = (1980, 2005),
    evaluation_period: tuple[int, int] = (2006, 2019),
    projection_period: tuple[int, int] | None = (2071, 2100),
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,  # type: ignore[assignment]
    output_dir: str | None = None,
    chunks: dict[str, int] | None = None,
) -> None:
    """
    Get CMIP6 data for given regions, variables and periods. Uses google cloud storage to retrieve data.
    It also regrids the data to the given baseline dataset.

    You have one file gathering all requested files per:
    - area of interest,
    - period
    - institute / model / experiment / ensemble.

    Parameters
    ----------
    aoi: list[gpd.GeoDataFrame]
        List of GeoDataFrames defining the areas of interest.
    cmip6_simulations: pd.DataFrame
        DataFrame containing the CMIP6 simulations to retrieve. Typically the output of the `list_available_simulations` method from the `CMIP6Context` class.
    baseline_period: tuple[int, int]
        Interval of years to use for the baseline period.
    evaluation_period: tuple[int, int]
        Interval of years to use for the evaluation period.
    projection_period: tuple[int, int] | None
        Interval of years to use for the projection period. If None, no projection period will be used (can be used if you need to do only evaluation)
    aggregation: Aggregation
        Aggregation method to use for aggregating the data.
    output_dir: str | None
        Directory to save the output files.
    chunks: dict[str, int] | None
        Chunking strategy to use for the data. Keys must be one / combination of "time", "lat", "lon".

    Returns
    -------
    None
    """

    data_product = DataProduct.CMIP6

    # Default values of chunks
    if chunks is None:
        chunks = {"time": 100, "lat": 400, "lon": 400}

    # Create output directory
    if output_dir is None:
        output_dir = f"./results/{data_product.product_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)

    gcfs = connect_to_gcfs()

    # Retrieve data
    all_ds = {}
    cmip6_simulations_grouped = cmip6_simulations.groupby(
        ["institute", "source", "ensemble", "experiment"]
    )
    for group_name, group in cmip6_simulations_grouped:
        logger.info("Preparing CMIP6 data for %s.", group_name)
        ds_group = []
        for _, row in group.iterrows():
            # Todo
            # First check if the final dataset already exists
            # For a given time period, we need to check if ALL aoi are present;
            # if this is not the case, then we need to download the data
            # Todo
            mapper = gcfs.get_mapper(row.datanode)
            ds_group.append(
                xr.open_zarr(mapper, consolidated=True).assign_coords(
                    {
                        "source": row.source,
                        "institute": row.institute,
                        "experiment": row.experiment,
                        "ensemble": row.ensemble,
                    }
                )
            )
        all_ds[group_name] = prep_dataset(
            xr.merge(ds_group).chunk(chunks=chunks), DataProduct.CMIP6
        )

    # Define time periods
    periods_years = [baseline_period, evaluation_period, projection_period]
    periods_names = ["baseline", "evaluation", "projection"]

    for period_year, period_name in zip(periods_years, periods_names, strict=False):
        tmin, tmax = split_period(period_year)
        for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
            logger.info("Extracting CMIP6 data for %s period, years %s to %s, for the area of interest '%s'.",
                        period_name, tmin, tmax, aoi_n)
            # Extend the AOI to avoid edge effects
            bounds = aoi_b.copy()
            bounds["minx"] -= 2
            bounds["miny"] -= 2
            bounds["maxx"] += 2
            bounds["maxy"] += 2
            for (institute, source, ensemble, experiment), ds in all_ds.items():
                output_file = _get_filename_from_cmip6_context(
                    output_dir, aoi_n, data_product, institute, source, experiment, ensemble, aggregation, tmin, tmax
                )
                if Path(output_file).is_file():
                    logger.info("CMIP6 data for %s already exists.", output_file)
                    continue
                ds_period = ds.sel(time=slice(tmin, tmax))
                if ds_period.sizes["time"] == 0:
                    continue
                ds_aoi = ds_period.rio.write_crs("epsg:4326").rio.clip_box(*bounds.to_numpy()[0])
                if aggregation != Aggregation.MONTHLY_MEAN:
                    msg = "Currently only monthly-means aggregation available!"
                    raise ValueError(msg)
                ds_clim = get_monthly_climatology(ds_aoi)
                ds_clim.to_netcdf(output_file)


############################################################################################################
# DEPRECATED
############################################################################################################


def get_cmip6_old(
    aoi: Iterable[gpd.GeoDataFrame],
    variable: Iterable[str] = ("pr", "tas", "tasmin", "tasmax"),
    baseline_period: tuple[int, int] = (1980, 2005),
    evaluation_period: tuple[int, int] = (2006, 2019),
    projection_period: tuple[int, int] = (2071, 2100),
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
    output_dir: str = "./results/cmip6",
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
    output_dir: str
        Output directory where the CMIP6 climatology will be stored.
        Default is "./results/cmip6".
    """

    if chunks is None:
        chunks = {"time": 100, "lat": 400, "lon": 400}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        for period in [baseline_period, evaluation_period, projection_period]
    ]
    dmin = min(dminsmaxs, key=lambda x: x[0])[0]
    dmax = max(dminsmaxs, key=lambda x: x[1])[1]
    ds = ds.sel(time=slice(dmin, dmax))
    ds = ds.chunk(chunks=chunks)
    ds = prep_dataset(ds, "cmip6")

    for i, period in enumerate([baseline_period, evaluation_period, projection_period]):
        dmin, dmax = dminsmaxs[i]
        ds_clim = get_monthly_climatology(ds.sel(time=slice(dmin, dmax)))
        for aoi_n in aoi_name:
            baseline_file = (
                f"./results/{baseline}/{aoi_n}_chelsa2_{aggregation.value}_*.nc"
            )
            base = xr.open_dataset(baseline_file[0])
            regridder = xe.Regridder(ds_clim, base, "bilinear")
            ds_r = regridder(ds_clim, keep_attrs=True)
            output_file = f"""{output_dir}/{aoi_n}_CMIP6_global_{institution}_{source}_{experiment}_{member}_none_none_{baseline}_{aggregation.value}_{period[0]}-{period[1]}.nc"""
            ds_r.to_netcdf(output_file)

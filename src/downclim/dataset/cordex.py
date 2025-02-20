from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import multiprocess as mp
import pandas as pd
import pyesgf
import xarray as xr
import xesmf as xe
from pydantic import BaseModel, Field, field_validator

from .aoi import get_aoi_informations
from .connectors import connect_to_esgf
from .utils import (Aggregation, DataProduct, Frequency,
                    get_monthly_climatology, prep_dataset, split_period)


class CORDEXContext(BaseModel):
    """Context about the query on the CORDEX dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ ."""

    project: list[str] = Field(default="CORDEX", description="Name of the project")
    product: list[str] | None = Field(
        default="output",
        description="Name of the product. You probably want to keep the default value.",
    )
    domain: list[str] | None = Field(
        default=None,
        examples=["EUR-11", "AFR-22"],
        description="CORDEX domain(s) to look for",
    )
    institute: list[str] | None = Field(
        default=None,
        example=["IPSL", "NCAR"],
        description="Institute name that produced the data",
    )
    driving_model: list[str] | None = Field(
        default=None,
        example=["IPSL-CM6A-LR", "CMCC-CM2-HR4"],
        description="Name of the global climate model used to drive the RCM",
    )
    experiment: list[str] | None = Field(
        default=["historical", "rcp26"],
        example=["historical", "rcp26"],
        description="Name of the experiment type of the simulation",
    )
    experiment_family: list[str] | None = Field(
        default=None,
        example=["RCP", "Historical"],
        description="Type of experiment : weither 'RCP' or 'Historical' or 'All'",
    )
    experiment: list[str] | None = Field(
        default="r1i1p1", description="Ensemble member"
    )
    rcm_model: list[str] | None = Field(
        default=None,
        example=["WRF381P", "RCA4"],
        description="Name of the regional climate model (RCM)",
    )
    downscaling_realisation: list[str] | None = Field(
        default=None,
        example=["v1", "v2"],
        description="Version of the downscaling realisation",
    )
    frequency: Frequency = Field(
        default=Frequency.MONTHLY,
        example="mon",
        description="Time frequency of the data",
    )
    variable: list[str] | None = Field(
        default=("tas", "pr"),
        example=["tas", "tasmin", "tasmax", "pr"],
        description="Variables name"
    )
    variable_long_name: list[str] | None = Field(
        default=None, example=[], description="Long name of the variables"
    )

    class Config:
        """Pydantic configuration for the DownClimContext class."""

        #arbitrary_types_allowed = True
        extra = "forbid"  # Forbid extra data during model initialization.

    @classmethod
    def to_list(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return [v]
        return v

    @field_validator(
        "experiment",
        "institute",
        "driving_model",
        "rcm_model",
        "downscaling_realisation",
        "variable",
        "variable_long_name",
        mode="before")
    @classmethod
    def validate_list(cls, v: Any) -> list[Any] | None:
        msg = f"Value {v} is not valid. Please provide a string, a tuple, set or list of string."
        if v is None:
            return v
        if isinstance(v, str):
            return [v]
        if isinstance(v, tuple | set):
            if all(isinstance(e, str) for e in v):
                return list(v)
            raise ValueError(msg)
        if isinstance(v, list):
            if all(isinstance(e, str) for e in v):
                return v
            raise ValueError(msg)
        raise ValueError(msg)

    @field_validator("experiment", mode="before")
    @classmethod
    def validate_experiment(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            msg = "No experiment provided, defaulting to ['historical', 'rcp26']"
            warnings.warn(msg, stacklevel=1)
            return ["historical", "rcp26"]
        if isinstance(v, str):
            return [v]
        if not any(exp == "historical" for exp in v):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=1)
            return [*v, "historical"]
        return v

    @field_validator("project", mode="after")
    @classmethod
    def validate_project(
        cls, project: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not project:
            msg = "No project provided, defaulting to 'CORDEX'"
            warnings.warn(msg, stacklevel=1)
            return ["CORDEX"]
        if isinstance(project, str):
            return [project]
        return project

    @field_validator("product", mode="after")
    @classmethod
    def validate_product(
        cls, product: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not product:
            msg = "No product provided, defaulting to 'output'"
            warnings.warn(msg, stacklevel=1)
            return ["output"]
        if isinstance(product, str):
            return [product]
        return product


def _get_wget_download_lines(wget_file: str) -> list[str]:
    with Path(wget_file).open(encoding="utf-8") as reader:
        return [
            num for num, line in enumerate(reader, 1) if re.match(r".*\.(nc)", line)
        ]


def _get_cordex_wget(
    script: str,
    i: int,
    periods: list[tuple[(int, int)]],
    tmp_dir: str="./results/tmp/cordex",
) -> bool:

    script_name = f"{tmp_dir}/wget_{i}.sh"

    # Write script to file
    # Select only required files corresponding to the required periods
    with Path(script_name).open(mode="w") as writer:
        for line in script.split("\n"):
            if re.match(r".*\.(nc)", line):
                tmin, tmax = (
                    int(year[:4])
                    for year in line.split(".nc")[0].split("_")[-1].split("-")
                )
                if any(period[0] <= tmax and period[1] >= tmin for period in periods):
                    writer.write(line + "\n")
            else:
                writer.write(line + "\n")

    Path.chmod(script_name, 0o750)
    subprocess.check_output(["/bin/bash", Path(script_name).name], cwd=tmp_dir)
    return True

@lru_cache
def _get_cordex_domains(
    url: str = "https://raw.githubusercontent.com/WCRP-CORDEX/domain-tables/main/CORDEX-CMIP5_rotated_grids.csv"
) -> pd.DataFrame:
    """Get the CORDEX domains boundaries.

    Args:
        url (_type_, optional): URL to the CORDEX domains boundaries.
            Defaults to "https://raw.githubusercontent.com/WCRP-CORDEX/domain-tables/main/CORDEX-CMIP5_rotated_grids.csv".

    Returns:
        pd.DataFrame: DataFrame containing the boundaries of the CORDEX domains.
    """
    return pd.read_csv(url)[["CORDEX_domain", "ll_lon", "ll_lat", "ur_lon", "ur_lat"]]

def _aoi_in_domain(aoi_bounds: pd.DataFrame, domain: pd.DataFrame) -> bool:
    """Check if the AOI is in the domain.

    Args:
        aoi_bounds (pd.DataFrame): Boundaries of the AOI.
        domain (pd.DataFrame): Boundaries of the CORDEX domain.

    Returns:
        bool: True if the AOI is in the domain, False otherwise.
    """
    return (
        aoi_bounds["minx"].values >= domain["ll_lon"].values
        and aoi_bounds["miny"].values >= domain["ll_lat"].values
        and aoi_bounds["maxx"].values <= domain["ur_lon"].values
        and aoi_bounds["maxy"].values <= domain["ur_lat"].values
    )[0] is np.True_

def list_available_cordex_simulations(
    context: dict[str, str | Iterable[str]] | CORDEXContext,
    esgf_credential: str = "config/esgf_credential.yaml",
    server: str = DataProduct.CORDEX.url,
) -> pd.DataFrame:
    """List all available CORDEX simulations available on esgf node for a given context.

    Parameters
    ----------
        context(dict[str, str | Iterable[str]] | CORDEXContext):
            Class containing information about the query on the CORDEX dataset, or dictionary.
            Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided.
            These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ . None are mandatory:
                - "project": str, Name of the project,
                    e.g "CORDEX", "CORDEX-Adjust". If not provided, "CORDEX" is used.
                - "product": str, Name of the product,
                    e.g "output", "output-adjust". If not provided, "output" is used.
                - "domain": str, CORDEX domain to look for,
                    e.g "SAM-22", "AFR-22", "AUS-22".
                - "institute": str, Institute name that produced the data,
                    e.g "IPSL", "NCAR"
                - "driving_model": str, global climate model that provided the boundary conditions to the RCM,
                    e.g "CNRM-CM5", "MPI-ESM-LR"
                - "experiment": str, experiment name, historical or futur scenarios to search for,
                    e.g "rcp26", "rcp85", "evaluation", "historical"
                - "experiment_family": str, type of experiment,
                    either "RCP" or "Historical"
                - "ensemble": str, ensemble member,
                    e.g "r1i1p1", "r3i1p1"
                - "rcm_model": str, name of the regional climate model,
                    e.g "CCLM4-8-17", "RACMO22E"
                - "downscaling_realisation": str, version of the downscaling realisation,
                    e.g "v1", "v2"
                - "time_frequency": str, time frequency of the data,
                    e.g "mon"
                - "variable": str, name of the variable to search for,
                    e.g "tas", "pr"
                - "variable_long_name": str, long name of the variable,
                    e.g "near-surface air temperature", "precipitation"
        esgf_credential (str, optional): Path to the ESGF credentials file.
            Keys expected in the files are "openid" and "password".
            Defaults to "config/esgf_credential.yaml".
        server (str, optional): URL to the ESGF node. Defaults to "https://esgf-node.ipsl.upmc.fr/esg-search".

    Returns:
        pd.DataFrame: List of CORDEX simulations available on esgf node meeting the search criteria.
    """

    if isinstance(context, CORDEXContext):
        context = context.model_dump()

    # esgf connection
    conn = connect_to_esgf(esgf_credential, server)
    # list CORDEX datasets matching context
    cordex_simulations = inspect_cordex(context=context, connector=conn)
    # filter simulations that don't have all variables requested
    cordex_simulations = cordex_simulations.groupby(
        ["institute", "driving_model", "rcm_model", "experiment", "ensemble"]
    ).filter(lambda x: set(context["variable"]) == (set(x["variable"])))
    # filter simulations that don't have all experiment requested
    cordex_simulations = cordex_simulations.groupby(
        ["institute", "driving_model", "rcm_model", "ensemble"]
    ).filter(lambda x: set(context["experiment"])==(set(x["experiment"])))

    if cordex_simulations.empty:
        msg = "No CORDEX simulations found for the given context"
        warnings.warn(msg, stacklevel=1)
    return cordex_simulations

def inspect_cordex(
    context: dict[str, str | Iterable[str]],
    connector: pyesgf.SearchConnection,
) -> pd.DataFrame:
    """
    Inspects ESGF server to get information about the available datasets provided the context.

    Parameters
    ----------
    context: dict
        Dictionary containing information about the query on the ESGF server. It must contain the following keys:
            - domain: str
            - gcm: str
            - rcm: str
            - time_frequency: str
            - experiment: str
            - variable: str
    connector: pyesgf.SearchConnection
        Connector to the ESGF server. Obtained using the `connect_to_esgf()` function.

    Returns
    -------
    pd.DataFrame: DataFrame containing information about the available datasets matching the query
    """

    # name mapping between context / CORDEX dataset
    cordex_name_mapping = {
        "frequency": "time_frequency",
    }

    if context["frequency"] == Frequency.MONTHLY:
        context["frequency"] = "mon"
    context_cleaned = {k: v for k, v in context.items() if v}
    for k in cordex_name_mapping:
        context_cleaned[cordex_name_mapping[k]] = context_cleaned.pop(k, None)
    facets_list = [*context_cleaned.keys()]
    facets = ", ".join(facets_list)
    ctx = connector.new_context(facets=facets, **context_cleaned)
    if ctx.hit_count == 0:
        msg = "The query has no results"
        raise SystemExit(msg)
    df_cordex = pd.DataFrame(
        [re.split("[|.]", res.dataset_id, maxsplit=12) for res in ctx.search()]
    ).drop_duplicates()
    df_cordex.columns = [
        "project",
        "product",
        "domain",
        "institute",
        "driving_model",
        "experiment",
        "ensemble",
        "rcm_model",
        "downscaling_realisation",
        "time_frequency",
        "variable",
        "version",
        "datanode",
    ]
    df_cordex.project = df_cordex.project.str.upper()
    cordex_scripts = [res.file_context().get_download_script() for res in ctx.search()]
    df_cordex["download_script"]=cordex_scripts
    return df_cordex

def get_context_from_filename(filename: str) -> dict[str, str]:
    """
    Get the cordex context from a filename parsing.

    Parameters
    ----------
    filename: str
        Filename to parse.

    Returns
    -------
    dict: Dictionary containing the context.
    """
    only_name = Path(filename).name
    keys = ["variable", "domain", "driving_model", "experiment", "ensemble", "rcm_model", "downscaling_realisation", "time_frequency", "period"]
    return dict(zip(keys, only_name.split(".")[0].split("_"), strict=False))


def get_cordex_from_list(
    aoi: list[gpd.GeoDataFrame],
    cordex_simulations: pd.DataFrame,
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    projection_year: tuple[int, int] = (2071, 2100),
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    output_dir: str = "./results/cordex",
    tmp_dir: str = "./results/tmp/cordex",
    nb_threads: int = 1,
    keep_tmp_dir: bool = False,
    esgf_credential: str | None = None,
) -> None:
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)
    domains = cordex_simulations["domain"].unique()
    all_cordex_domains = _get_cordex_domains()
    cordex_domains = all_cordex_domains[all_cordex_domains["CORDEX_domain"].isin(domains)]
    aoi_in_domain = {}
    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        aoi_in_domain[aoi_n] = {
            domain: _aoi_in_domain(aoi_b, cordex_domains.loc[cordex_domains["CORDEX_domain"]==domain]) for domain in domains
                }

    # connect
    _ = connect_to_esgf(esgf_credential, server=DataProduct.CORDEX.url)

    # Define time periods
    periods_years = [baseline_year, evaluation_year, projection_year]
    periods_names = ["baseline", "evaluation", "projection"]

    # download
    pool = mp.Pool(nb_threads)
    pool.starmap_async(
        _get_cordex_wget,
        [(script, i, periods_years, tmp_dir) for i, script in enumerate(cordex_simulations["download_script"])],
    ).get()
    pool.close()

    # rearrange and sort files downloaded
    files = [f"{tmp_dir}/{f}" for f in os.listdir(tmp_dir) if re.match(r".*\.(nc)", f)]
    df_files = []
    for f in files:
        df_f = pd.DataFrame.from_dict(get_context_from_filename(f), orient='index').T
        df_f['filename'] = f
        df_files.append(df_f)
    df_files = pd.concat(df_files, ignore_index=True)

    # group files for each simulation
    group_context = ["domain", "driving_model", "ensemble", "rcm_model", "downscaling_realisation"]
    for group_name, group in df_files.groupby(group_context):
        # read & prepare
        ds = xr.open_mfdataset(group["filename"], parallel=True)
        ds = prep_dataset(ds, DataProduct.CORDEX)
        # For each aoi
        for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
            # Extend the AOI to avoid edge effects
            aoi_b["minx"] -= 2
            aoi_b["miny"] -= 2
            aoi_b["maxx"] += 2
            aoi_b["maxy"] += 2
            ds_aoi = ds.sel(lon=slice(aoi_b["minx"], aoi_b["maxx"]), lat=slice(aoi_b["miny"], aoi_b["maxy"]))
            # write per aoi and period
            for period_year, period_name in zip(periods_years, periods_names, strict=False):
                tmin, tmax = split_period(period_year)

                print(f"""Extracting CORDEX data for {period_name}, years {tmin} to {tmax},
                  for the area of interest {aoi_n}.""")

                if aggregation == Aggregation.MONTHLY_MEAN:
                    ds_clim = get_monthly_climatology(ds_aoi.sel(time=slice(tmin, tmax)))
                else:
                    msg = "Currently only monthly-means aggregation available!"
                    raise ValueError(msg)
                for aoi_n in aois_names:
                    output_file = f"{output_dir}/{aoi_n}_CORDEX_{domain}_{institution}_{source}_{experiment}_{member}_none_none_{baseline}_{aggregation.value}_{period[0]}-{period[1]}.nc"
                    ds_clim.to_netcdf(output_file)

    if not keep_tmp_dir:
        shutil.rmtree(tmp_dir)



############################################################################################################
# DEPRECATED
############################################################################################################


def get_cordex(
    aois: list[gpd.GeoDataFrame],
    variables: list[str],
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    projection_year: tuple[int, int] = (2071, 2100),
    # time_frequency: str = "mon",
    aggregation: str = "monthly-means",
    domain: str = "EUR-11",
    institute: str = "IPSL",
    model: str = "IPSL-IPSL-CM5A-MR",
    rcm: str = "WRF381P",
    experiment: str = "rcp45",
    ensemble: str = "r1i1p1",
    baseline: str = "chelsa2",
    downscaling: str = "v1",
    esgf_credential: str = "config/credentials_esgf.yml",
    nb_threads: int = 1,
    output_dir: str = "./results/cordex",
    tmp_dir: str = "./results/tmp/cordex",
    keep_tmp_directory: bool = False,
) -> None:
    """
    Get CORDEX data for a given region and period.

    Parameters
    ----------
    base_file: list
        List of baseline files.
    domain: str
        Domain of the region.
    areas: list
        List of areas.
    institute: str
        Institute.
    model: str
        Model.
    experiment: str
        Experiment.
    ensemble: str
        Ensemble.
    baseline: str
        Baseline.
    rcm: str
        RCM.
    downscaling: str
        Downscaling.
    variables: list
        List of variables.
    time_frequency: str
        Time frequency.
    esgf_credential: str
        ESGF credentials.
    threads: int
        Number of threads for downloading in parallel. Default is 1 (i.e. no parallel downloading).
    periods: list
        List of periods.
    aggregation: str
        Aggregation.
    tmp_dir: str
        Temporary directory.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path.is_dir(tmp_dir):
        context = {
            "domain": domain,
            "gcm": model,
            "rcm": rcm,
            "time_frequency": "mon",
            "experiment": experiment,
            "var": variables,
        }
        # connect
        connector = connect_to_esgf(esgf_credential, server=DataProduct.CORDEX.url)
        ctx = connector.new_context(facets="*", **context)
        all_scripts = [res.file_context().get_download_script() for res in ctx.search()]

        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        # download
        pool = mp.Pool(nb_threads)
        pool.starmap_async(
            _get_cordex_wget,
            [(script, i, tmp_dir) for i, script in enumerate(all_scripts)],
        ).get()
        pool.close()
    else:
        print(
            f"Folder '{tmp_dir}' already present, assuming already downloaded data. If this is not the case, "
        )

    # read & prepare
    files = [f"{tmp_dir}/{f}" for f in os.listdir(tmp_dir) if re.match(r".*\.(nc)", f)]
    files_hist = [f for f in files if re.search("_historical", f)]
    files_rcp = [f for f in files if f not in files_hist]
    ds_hist = xr.open_mfdataset(files_hist, parallel=True)
    ds_rcp = xr.open_mfdataset(files_rcp, parallel=True)
    ds_hist = prep_dataset(ds_hist)
    ds_rcp = prep_dataset(ds_rcp)

    check_file = "toto.nc"
    periods = [baseline_year, evaluation_year, projection_year]
    # regrid and write per country
    for aoi in aois:
        base_file = f"{Path(check_file).parent}/{aoi}_base.nc"
        base = xr.open_dataset(base_file)
        regridder_hist = xe.Regridder(ds_hist, base, "bilinear", ignore_degenerate=True)
        ds_hist_r = regridder_hist(ds_hist, keep_attrs=True)
        regridder_rcp = xe.Regridder(ds_rcp, base, "bilinear", ignore_degenerate=True)
        ds_rcp_r = regridder_rcp(ds_rcp, keep_attrs=True)
        for period in periods:
            dmin, dmax = split_period(period)
            if dmax <= "2005-01-01":
                ds_a = (
                    ds_hist_r.sel(time=slice(dmin, dmax))
                    .groupby("time.month")
                    .mean("time")
                )
            else:
                ds_a = (
                    ds_rcp_r.sel(time=slice(dmin, dmax))
                    .groupby("time.month")
                    .mean("time")
                )
            path = f"{Path(check_file).parent}/{aoi}_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"
            ds_a.to_netcdf(path)

    if not keep_tmp_directory:
        shutil.rmtree(tmp_directory)

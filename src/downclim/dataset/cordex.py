from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import multiprocess as mp
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field, field_validator
from pyesgf.search.connection import SearchConnection

from ..aoi import get_aoi_informations
from ..logging_config import get_logger
from .connectors import connect_to_esgf
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    _check_output_dir,
    get_monthly_climatology,
    prep_dataset,
    sel_period,
    split_period,
)

logger = get_logger(__name__)

simulations_columns = [
    "project",
    "product",
    "domain",
    "institute",
    "driving_model",
    "experiment",
    "ensemble",
    "rcm_name",
    "rcm_version",
    "time_frequency",
    "variable",
    "version",
    "datanode",
]


class CORDEXContext(BaseModel):
    """Context about the query on the CORDEX dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ ."""

    project: list[str] = Field(
        default=["CORDEX"], examples=["CORDEX"], description="Name of the project"
    )
    product: list[str] | None = Field(
        default=["output"],
        examples=["output"],
        description="Name of the product. You probably want to keep the default value.",
    )
    domain: list[str] | None = Field(
        default=None,
        examples=[["EUR-11", "AFR-22"], ["EUR-11", "EUR-44"]],
        description="CORDEX domain(s) to look for",
    )
    institute: list[str] | None = Field(
        default=None,
        examples=[["IPSL", "NCAR"]],
        description="Institute name that produced the data",
    )
    driving_model: list[str] | None = Field(
        default=None,
        examples=[["IPSL-CM6A-LR", "CMCC-CM2-HR4"]],
        description="Name of the global climate model used to drive the RCM",
    )
    experiment: list[str] | None = Field(
        default=["historical", "rcp26"],
        examples=[["historical", "rcp26"]],
        description="Name of the experiment type of the simulation",
    )
    experiment_family: list[str] | None = Field(
        default=None,
        examples=[["RCP", "Historical"]],
        description="Family of experiment : weither 'RCP' or 'Historical' or 'All'",
    )
    ensemble: list[str] | None = Field(
        default=["r1i1p1"], description="Ensemble member"
    )
    rcm_name: list[str] | None = Field(
        default=None,
        examples=[["WRF381P", "RCA4"]],
        description="Name of the regional climate model (RCM)",
    )
    rcm_version: list[str] | None = Field(
        default=None,
        examples=[["v1", "v2"]],
        description="Version of the downscaling realisation",
    )
    frequency: Frequency = Field(
        default=Frequency.MONTHLY,  # type: ignore[assignment]
        examples=[Frequency.MONTHLY, "mon"],
        description="Time frequency of the data",
    )
    variable: list[str] | None = Field(
        default=["tas", "pr"],
        examples=[("tas", "pr"), ["tas", "tasmin", "tasmax", "pr"]],
        description="Variables name",
    )
    variable_long_name: list[str] | None = Field(
        default=None,
        examples=["Near-Surface Air Temperature"],
        description="Long name of the variables",
    )

    class Config:
        """Pydantic configuration for the DownClimContext class."""

        # arbitrary_types_allowed = True
        extra = "forbid"  # Forbid extra data during model initialization.

    @field_validator(
        "domain",
        "ensemble",
        "experiment",
        "institute",
        "driving_model",
        "rcm_name",
        "rcm_version",
        "variable",
        "variable_long_name",
        mode="before",
    )
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

    @field_validator("experiment", mode="after")
    @classmethod
    def validate_experiment(cls, v: str | Iterable[str] | None) -> list[str]:
        if v is None:
            msg = "No experiment provided, defaulting to ['historical', 'rcp26']"
            logger.warning(msg)
            return ["historical", "rcp26"]
        if isinstance(v, str):
            return [v]
        if not any(exp == "historical" for exp in v):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            logger.warning(msg)
            return [*v, "historical"]
        return list(v)

    @field_validator("project", mode="after")
    @classmethod
    def validate_project(cls, project: str | Iterable[str] | None) -> list[str]:
        if not project:
            msg = "No project provided, defaulting to 'CORDEX'"
            logger.warning(msg)
            return ["CORDEX"]
        if isinstance(project, str):
            return [project]
        return list(project)

    @field_validator("product", mode="after")
    @classmethod
    def validate_product(cls, product: str | Iterable[str] | None) -> list[str]:
        if not product:
            msg = "No product provided, defaulting to 'output'"
            logger.warning(msg)
            return ["output"]
        if isinstance(product, str):
            return [product]
        return list(product)

    @field_validator("frequency", mode="after")
    @classmethod
    def validate_frequency(cls, frequency: str | Frequency) -> Frequency:
        if isinstance(frequency, str):
            frequency = Frequency(frequency)
        if frequency is not Frequency.MONTHLY:
            msg = f"""Frequency {frequency} is not valid. Please provide a valid string or a Frequency object.
            Right now only 'monthly' is implemented."""
            raise ValueError(msg)
        return frequency

    def clean_context(self) -> dict[str, Any]:
        """
        From the CORDEXContext object, returns a dictionary directly usable for ESGF search.

        Returns
        -------
        dict[str, Any]: dictionary containing esgf keywords request.
        """
        context_cleaned = {k: v for k, v in dict(self).items() if v}
        if context_cleaned["frequency"] == Frequency.MONTHLY:
            context_cleaned["time_frequency"] = "mon"
        del context_cleaned["frequency"]

        return context_cleaned

    def inspect_cordex(
        self,
        context_cleaned: dict[str, Any],
        connector: SearchConnection | None = None,
    ) -> pd.DataFrame:
        """
        Inspects ESGF server to get information about the available datasets provided the cordexcontext.

        Parameters
        ----------
        context_cleaned: dict[str, Any]
            Dictionary containing the cleaned context obtained from self.clean_context().
        connector: pyesgf.search.connection.SearchConnection | None = None
            Pyesgf connect object.
            Defaults to None, in which case the function will fail.

        Returns
        -------
        pd.DataFrame: DataFrame containing information about the available datasets matching the query
        """

        # connect
        if connector is None:
            msg = (
                "No connector provided, please provide one to get the download scripts."
            )
            raise KeyError(msg)

        facets_list = [*context_cleaned.keys()]
        facets = ", ".join(facets_list)
        ctx = connector.new_context(facets=facets, **context_cleaned)
        if ctx.hit_count == 0:
            msg = "The query has no results"
            logger.warning(msg)
            return pd.DataFrame()
        df_cordex = pd.DataFrame(
            [re.split("[|.]", res.dataset_id, maxsplit=12) for res in ctx.search()]
        ).drop_duplicates()
        df_cordex.columns = simulations_columns
        df_cordex.project = df_cordex.project.str.upper()
        # cordex_scripts = [res.file_context().get_download_script() for res in ctx.search(ignore_facet_check=True)]
        # df_cordex["download_script"]=cordex_scripts
        return df_cordex

    def list_available_simulations(
        self,
        esgf_credential: str = "config/esgf_credential.yaml",
        server: str = DataProduct.CORDEX.url,
        save_simulations: str | None = None,
    ) -> pd.DataFrame:
        """List all available CORDEX simulations available on esgf node for a given context.

        Parameters
        ----------
            esgf_credential (str, optional): Path to the ESGF credentials file.
                Keys expected in the files are "openid" and "password".
                Defaults to "config/esgf_credential.yaml".
            server (str, optional): URL to the ESGF node. Defaults to "https://esgf-node.ipsl.upmc.fr/esg-search".
            save_simulations: str | None (default: None)
                Filepath to save the dataframe to a csv file. Not saved by default.

        Returns:
            pd.DataFrame: List of CORDEX simulations available on esgf node meeting the search criteria.
        """
        context = self.clean_context()
        # esgf connection
        conn = connect_to_esgf(esgf_credential, server)
        # list CORDEX datasets matching context
        cordex_simulations = self.inspect_cordex(context, connector=conn)
        # filter simulations that don't have all variables requested
        cordex_simulations = cordex_simulations.groupby(
            ["institute", "driving_model", "rcm_name", "experiment", "ensemble"]
        ).filter(lambda x: set(context["variable"]) == (set(x["variable"])))
        # filter simulations that don't have all experiment requested
        cordex_simulations = cordex_simulations.groupby(
            ["institute", "driving_model", "rcm_name", "ensemble"]
        ).filter(lambda x: set(context["experiment"]) == (set(x["experiment"])))

        if cordex_simulations.empty:
            msg = "No CORDEX simulations found for the given context"
            logger.warning(msg)
        if save_simulations:
            cordex_simulations.to_csv(save_simulations, index=False)
        return cordex_simulations


def _get_cordex_wget(
    script: str,
    i: int,
    periods: list[tuple[(int, int)]],
    tmp_dir: str = "./results/tmp/cordex",
) -> bool:
    script_path = Path(f"{tmp_dir}/wget_{i}.sh")

    # Write script to file
    # Check if the file already exists
    # Select only required files corresponding to the required periods
    with script_path.open(mode="w", encoding="utf-8") as writer:
        for line in script.split("\n"):
            if re.match(r".*\.(nc)", line):
                if (
                    not Path(tmp_dir)
                    .joinpath(line.split()[0].replace("'", ""))
                    .is_file()
                ):
                    tmin, tmax = (
                        int(year[:4])
                        for year in line.split(".nc")[0].split("_")[-1].split("-")
                    )
                    if any(
                        period[0] <= tmax and period[1] >= tmin for period in periods
                    ):
                        writer.write(line + "\n")
            else:
                writer.write(line + "\n")

    script_path.chmod(0o750)
    subprocess.check_output(["/bin/bash", script_path.name], cwd=tmp_dir)
    return True


@lru_cache
def _get_cordex_domains(
    url: str = "https://raw.githubusercontent.com/WCRP-CORDEX/domain-tables/main/CORDEX-CMIP5_rotated_grids.csv",
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
        aoi_bounds["minx"].to_numpy() >= domain["ll_lon"].to_numpy()
        and aoi_bounds["miny"].to_numpy() >= domain["ll_lat"].to_numpy()
        and aoi_bounds["maxx"].to_numpy() <= domain["ur_lon"].to_numpy()
        and aoi_bounds["maxy"].to_numpy() <= domain["ur_lat"].to_numpy()
    )[0] is np.True_


def _check_cordex_download(
    simulations: pd.DataFrame,
    tmp_dir: str = "./results/tmp/cordex",
) -> pd.DataFrame:
    """Check if the download of the CORDEX data has been successful.

    Args:
        simulations (pd.DataFrame): DataFrame containing the list of simulations.
        tmp_dir (str, optional): Path to the temporary directory. Defaults to "./results/tmp/cordex".

    Returns:
        pd.DataFrame: DataFrame containing the list of simulations and an additional column indicating if the download has been successful.
    """

    return None  # TODO: Implement this function


def get_download_scripts(
    simulations: pd.DataFrame,
    esgf_credential: str | None = None,
    server: str = DataProduct.CORDEX.url,
) -> pd.DataFrame:
    """Get the esgf download scripts for the simulations described in the DataFrame.

    Args:
        simulations (pd.DataFrame): DataFrame containing the simulations.
        esgf_credential (str, optional): Path to the ESGF credentials file, if needs to reconnect.
        server (str, optional): URL to the ESGF node. Defaults to "https://esgf-node.ipsl.upmc.fr/esg-search".

    Returns:
       pd.DataFrame: same DataFrame as the input with the download scripts added.
    """

    # connect
    if esgf_credential is None:
        msg = "No esgf_credential provided, please provide one to get the download scripts."
        raise KeyError(msg)
    connector = connect_to_esgf(esgf_credential, server=server)

    facets = ", ".join(simulations_columns)
    cordex_scripts = []
    for _, row in simulations.iterrows():
        logger.info("Getting download script for simulation:")
        logger.info(row)
        context = row.to_dict()
        context["version"] = context["version"][1:]
        context["data_node"] = context.pop("datanode")
        ctx = connector.new_context(facets=facets, **context)
        cordex_scripts.append(
            ctx.search(ignore_facet_check=True)[0].file_context().get_download_script()
        )
    return simulations.assign(download_script=cordex_scripts)


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
    keys = [
        "variable",
        "domain",
        "driving_model",
        "experiment",
        "ensemble",
        "rcm_name",
        "rcm_version",
        "time_frequency",
        "period",
    ]
    return dict(zip(keys, only_name.split(".")[0].split("_"), strict=False))


def _get_filename_from_cordex_context(
    output_dir: str,
    aoi_n: str,
    data_product: DataProduct,
    domain: str,
    driving_model: str,
    rcm_name: str,
    ensemble: str,
    rcm_version: str,
    aggregation: Aggregation,
    tmin: int,
    tmax: int,
) -> str:
    """Internal function. Get the name of the output file for the simulation given a Cordex context."""
    return f"{output_dir}/{aoi_n}_{data_product.product_name}_{domain}_{driving_model}_{rcm_name}_{ensemble}_\
                    {rcm_version}_{aggregation.value}_{tmin}-{tmax}.nc"


def get_cordex_context_from_filename(filename: str) -> dict[str, str]:
    """Get CORDEX context from a filename.

    Parameters
    ----------
    filename: str
        Filename containing CORDEX context of the simulation.

    Returns
    -------
    dict[str, str]
        List of main CORDEX context information, including:
            - output_dir
            - aoi_n
            - data_product
            - domain
            - driving_model
            - rcm_name
            - ensemble
            - rcm_version
            - aggregation
            - tmin
            - tmax

    """
    context_items = [
        "output_dir",
        "aoi_n",
        "data_product",
        "domain",
        "driving_model",
        "rcm_name",
        "ensemble",
        "rcm_version",
        "aggregation",
        "tmin",
        "tmax",
    ]
    context_elements = [
        str(Path(filename).parent),
        *Path(filename).name.split(".nc")[0].split("_"),
    ]
    return dict(zip(context_items, context_elements, strict=False))


def get_cordex(
    aoi: list[gpd.GeoDataFrame],
    cordex_simulations: pd.DataFrame,
    historical_period: tuple[int, int] = (1980, 2005),
    evaluation_period: tuple[int, int] = (2006, 2019),
    projection_period: tuple[int, int] = (2071, 2100),
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,  # type: ignore[assignment]
    output_dir: str | None = None,
    tmp_dir: str | None = None,
    nb_threads: int = 2,
    keep_tmp_dir: bool = False,
    esgf_credentials: str | None = None,
    server: str = DataProduct.CORDEX.url,
) -> None:
    """Download CORDEX data for the given area of interest (AOI) and provided simulations list.

    Parameters
    ----------
    aoi: list[gpd.GeoDataFrame]
        List of areas of interest (AOI) to download data for. Typically the output of :func:`~downclim.aoi.get_aoi` function.
    cordex_simulations: pd.DataFrame
        DataFrame containing the list of CORDEX simulations to download. It can be obtained using the : func:`~downclim.dataset.CordexContext.list_available_simulations` method.
    historical_period: tuple[int, int], optional
        Baseline period for the data, as a tuple of (start year, end year). Defaults to (1980, 2005).
    evaluation_period: tuple[int, int], optional
        Evaluation period for the data, as a tuple of (start year, end year). Defaults to (2006, 2019).
    projection_period: tuple[int, int], optional
        Projection period for the data, as a tuple of (start year, end year). Defaults to (2071, 2100).
    aggregation: Aggregation, optional
        Aggregation method to apply to the data. Defaults to Aggregation.MONTHLY_MEAN.
    output_dir: str | None, optional
        Directory where the output files will be saved. If None, set to "./results/CORDEX". Defaults to None.
    tmp_dir: str | None, optional
        Directory where the temporary files will be saved. If None, set to "./results/tmp/CORDEX". Defaults to None.
    nb_threads: int, optional
        Number of threads to use for downloading the data. Defaults to 2.
    keep_tmp_dir: bool, optional
        If True, keep the temporary directory after the download. Defaults to False.
    esgf_credentials: str | None, optional
        Path to the ESGF credentials file. If None, the function will fail. Defaults to None.
    server: str, optional
        URL to the ESGF node. Defaults to "https://esgf-node.ipsl.upmc.fr/esg-search".

    Raises
    ------
    KeyError: If the 'download_script' column is not present in the `cordex_simulations` DataFrame.
    ValueError: If the aggregation method is not supported (currently only MONTHLY_MEAN is implemented).
    Warning: If the AOI is not in the domain of the simulation selected.

    Returns
    -------
    None: The function does not return anything, but saves the downloaded data to the specified output directory.
    """

    data_product = DataProduct.CORDEX

    # Create output and tmp directories
    output_dir = _check_output_dir(output_dir, f"./results/{data_product.product_name}")
    tmp_dir = _check_output_dir(tmp_dir, f"./results/tmp/{data_product.product_name}")
    # if output_dir is None:
    #     output_dir = f"./results/{data_product.product_name}"
    # if tmp_dir is None:
    #     tmp_dir = f"./results/tmp/{data_product.product_name}"
    # Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    # Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)
    domains = cordex_simulations["domain"].unique()
    all_cordex_domains = _get_cordex_domains()
    cordex_domains = all_cordex_domains[
        all_cordex_domains["CORDEX_domain"].isin(domains)
    ]
    aoi_in_domain = {}
    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        aoi_in_domain[aoi_n] = {
            domain: _aoi_in_domain(
                aoi_b, cordex_domains.loc[cordex_domains["CORDEX_domain"] == domain]
            )
            for domain in domains
        }

    # check if need to et the download scripts
    if "download_script" not in cordex_simulations.columns:
        if esgf_credentials is not None:
            cordex_simulations = get_download_scripts(
                cordex_simulations, esgf_credentials, server
            )
        else:
            msg = """To retrieve the desired CORDEX data, you need to get the 'download_scripts' from ESGF.
            Please provide 'esgf_credentials' as argument to the function to do so, or run the 'get_download_scripts'
            to populate the DataFrame with the adequate 'download_scripts' column."""
            raise KeyError(msg)

    # Define time periods
    periods_years = [historical_period, evaluation_period, projection_period]
    periods_names = ["baseline", "evaluation", "projection"]

    # download
    with mp.Pool(nb_threads) as pool:
        pool.starmap_async(
            _get_cordex_wget,
            [
                (script, i, periods_years, tmp_dir)
                for i, script in enumerate(cordex_simulations["download_script"])
            ],
        ).get()

    # rearrange and sort files downloaded
    files = [str(f) for f in Path(tmp_dir).iterdir() if re.match(r".*\.(nc)", f.name)]
    df_files = []
    for f in files:
        df_f = pd.DataFrame.from_dict(get_context_from_filename(f), orient="index").T
        df_f["filename"] = f
        df_files.append(df_f)
    df_files = pd.concat(df_files, ignore_index=True)

    # group files for each simulation
    group_context = ["domain", "driving_model", "ensemble", "rcm_name", "rcm_version"]
    for group_name, group in df_files.groupby(group_context):
        domain, driving_model, ensemble, rcm_name, rcm_version = group_name
        # read & prepare
        ds = xr.open_mfdataset(group["filename"], parallel=True)
        ds = prep_dataset(ds, DataProduct.CORDEX)
        # For each aoi
        for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
            # check if aoi is in domain
            if not aoi_in_domain[aoi_n][domain]:
                msg = f"AOI {aoi_n} is not in domain {domain}. Skipping."
                logger.warning(msg)
                continue
            # Extend the AOI to avoid edge effects
            ds_aoi = ds.sel(
                rlon=slice(aoi_b["minx"] - 2, aoi_b["maxx"] + 2),
                rlat=slice(aoi_b["miny"] - 2, aoi_b["maxy"] + 2),
            )
            # write per aoi and period
            for period_year, period_name in zip(
                periods_years, periods_names, strict=False
            ):
                tmin, tmax = split_period(period_year)
                logger.info(
                    """Extracting CORDEX data for %s, years %s to %s,
                  for the area of interest \"%s\", and simulation %s_%s_%s_%s_%s""",
                    period_name,
                    tmin,
                    tmax,
                    aoi_n,
                    domain,
                    driving_model,
                    ensemble,
                    rcm_name,
                    rcm_version,
                )
                if aggregation == Aggregation.MONTHLY_MEAN:
                    ds_clim = get_monthly_climatology(sel_period(ds_aoi, tmin, tmax))
                else:
                    msg = "Currently only monthly-means aggregation available!"
                    raise ValueError(msg)
                output_file = _get_filename_from_cordex_context(
                    output_dir,
                    aoi_n,
                    data_product,
                    domain,
                    driving_model,
                    rcm_name,
                    ensemble,
                    rcm_version,
                    aggregation,
                    tmin,
                    tmax,
                )
                ds_clim.to_netcdf(output_file)

    if not keep_tmp_dir:
        shutil.rmtree(tmp_dir)

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pyesgf
from pydantic import BaseModel, Field, field_validator

from .getters.connectors import connect_to_esgf, data_urls


# CORDEX
# ------
class CORDEXContext(BaseModel):
    """Context about the query on the CORDEX dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ ."""

    project: str | list[str] | None = Field(
        default="CORDEX", example="CORDEX", description="Name of the project"
    )
    product: str | list[str] | None = Field(
        default="output", description="Name of the product"
    )
    domain: str | list[str] | None = Field(description="CORDEX domain(s) to look for")
    institute: str | list[str] | None = Field(
        default=None, description="Institute name that produced the data"
    )
    driving_model: str | list[str] | None = Field(
        default=None,
        description="Name of the global climate model used to drive the RCM",
    )
    experiment: str | list[str] | None = Field(
        default=["historical", "rcp26"],
        description="Name of the experiment type of the simulation",
    )
    experiment_family: str | list[str] | None = Field(
        default=None,
        description="Type of experiment : weither 'RCP' or 'Historical' or 'All'",
    )
    ensemble: str | list[str] | None = Field(
        default="r1i1p1", description="Ensemble member"
    )
    rcm_model: str | list[str] | None = Field(
        default=None, description="Name of the regional climate model"
    )
    downscaling_realisation: str | list[str] | None = Field(
        default=None, description="Version of the downscaling realisation"
    )
    time_frequency: str | None = Field(
        default="mon", description="Time frequency of the data"
    )
    variable: str | list[str] | None = Field(
        default=["tas", "tasmin", "tasmax", "pr"], description="Variables requested"
    )
    variable_long_name: str | list[str] | None = Field(
        default=None, description="Long name of the variables"
    )

    @field_validator("project")
    @classmethod
    def validate_project(
        cls, project: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not project:
            msg = "No project provided, defaulting to 'CORDEX'"
            warnings.warn(msg, stacklevel=2)
            return "CORDEX"
        return project

    @field_validator("product")
    @classmethod
    def validate_product(
        cls, product: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not product:
            msg = "No product provided, defaulting to 'output'"
            warnings.warn(msg, stacklevel=2)
            return "output"
        return product

    @field_validator("experiment", mode="before")
    @classmethod
    def validate_experiment(cls, experiment: str | list[str] | None) -> list[str]:
        if not any(exp == "historical" for exp in experiment):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
            if isinstance(experiment, str):
                return [experiment, "historical"]
            return [*experiment, "historical"]
        return experiment


def list_available_cordex_simulations(
    context: CORDEXContext,
    esgf_credential: str = "config/credentials_esgf.yml",
    server: str = data_urls["esgf"],
) -> pd.DataFrame:
    """List all available CORDEX simulations available on esgf node for a given context.

    Args:
        context (dict[str, str | Iterable[str]]): Dictionary containing information about the query on the CORDEX dataset.
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
                - "experiment": str, experiment name,
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
                - "variable": str, name of the variable,
                    e.g "tas", "pr"
                - "variable_long_name": str, long name of the variable,
                    e.g "near-surface air temperature", "precipitation"
        domain (Iterable[str], optional): List of CORDEX domains to search for.
            Defaults to ["SAM-22", "AFR-22", "AUS-22"].
        experiment (Iterable[str], optional): List of CORDEX historical or futur scenarios to search for.
            Defaults to ["rcp26", "rcp85"].
        variable (Iterable[str], optional): List of variables to search for.
            Defaults to ["tas", "tasmin", "tasmax", "pr"].
        time_frequency (str, optional): Time frequency of the data.
            Defaults to "mon".

    Returns:
        pd.DataFrame: List of CORDEX simulations available on esgf node meeting the search criteria.
    """

    # esgf connection
    conn = connect_to_esgf(esgf_credential, server)
    # list CORDEX datasets matching context
    cordex_simulations = inspect_cordex(context=context, connector=conn)
    # filter simulations that don't have both historical & projection
    return cordex_simulations.groupby(
        ["driving_model", "rcm_model", "ensemble"]
    ).filter(lambda x: set(context.experiment).issubset(set(x["experiment"])))


def inspect_cordex(
    context: dict[str, str | Iterable[str]] | CORDEXContext,
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

    if isinstance(context, CORDEXContext):
        context = context.model_dump()

    ctx = connector.new_context(facets="*", **context)
    if ctx.hit_count == 0:
        msg = "The query has no results"
        raise SystemExit(msg)
    df_cordex = pd.DataFrame(
        [re.split("[|.]", res.dataset_id, maxsplit=12) for res in ctx.search()]
    )
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
    return df_cordex.drop_duplicates()


# CMIP6
# -----


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
    def validate_experiment_id(cls, experiment: str | list[str] | None) -> list[str]:
        if not any(exp == "historical" for exp in experiment):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
            if isinstance(experiment, str):
                return [experiment, "historical"]
            return [*experiment, "historical"]
        return experiment


def list_available_cmip6_simulations(
    context: CMIP6Context,
) -> pd.DataFrame:
    """List all available CMIP6 simulations available on Google Cloud Storage for a given set of context.

    Parameters
    ----------
        context (CMIP6Context): Object containing information about the query on the CMIP6 dataset. Entries of the dictionary can be
        either `str` or `Iterables` (e.g. `list`) if multiple values are provides.

        These following keys are available. None are mandatory):
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


    Returns:
    -------
        pd.DataFrame: DataFrame containing information about the available datasets matching
    """

    # gcfs connection
    # gcfs_connector = connect_to_gcfs()
    # list CMIP6 datasets matching context
    cmip6_simulations = inspect_cmip6(context)
    cmip6_simulations = cmip6_simulations.assign(domain="GLOBAL")
    cmip6_simulations = cmip6_simulations.assign(product="output")
    cmip6_simulations["time_frequency"] = np.where(
        cmip6_simulations["table_id"] == "Amon", "mon", None
    )

    # filter simulations that don't have both historical & projection
    cmip6_simulations = cmip6_simulations.groupby(["source_id", "member_id"]).filter(
        lambda x: set(context.experiment_id).issubset(set(x["experiment_id"]))
    )

    return (
        cmip6_simulations.rename(
            columns={
                "activity_id": "project",
                "institution_id": "institute",
                "source_id": "model",
                "experiment_id": "experiment",
                "member_id": "ensemble",
                "variable_id": "variable",
                "zstore": "datanode",
                "table_id": "table",
            }
        )
        .reset_index()
        .drop("index", axis=1)
    )


def inspect_cmip6(
    context: dict[str, str | Iterable[str]] | CMIP6Context,
    cmip6_catalog_url: str = data_urls["gcsfs_cmip6"],
) -> pd.DataFrame:
    """
    Inspects Google Cloud File System to get information about the available CMIP6 datasets provided the context.

    Parameters
    ----------
    context: dict([str, str | Iterable[str]] | CMIP6Context)
        Dictionary or CMIP6Context object containing information about the query on the CMIP6 dataset. Entries of the dictionary can be
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
    cmip6_catalog_url: str (default: data_urls["gcsfs_cmip6"])
        URL to the CMIP6 catalog on the Google Cloud File System.

    Returns
    -------
    pd.DataFrame: DataFrame containing information about the available datasets matching the query
    """
    cmip6_catalog = pd.read_csv(cmip6_catalog_url)

    if isinstance(context, CMIP6Context):
        context = context.model_dump()

    search_string_parts = []
    for k, v in context.items():
        if v is not None:
            if isinstance(v, str):
                search_string_parts.append(f"{k} == '{v}'")
            else:
                search_string_parts.append(
                    "(" + " | ".join([f"{k} == '{w}'" for w in v]) + ")"
                )
    search_string = " & ".join(search_string_parts)

    return cmip6_catalog.query(search_string)


# ---------------
# save available simulations
def save_projections(
    cordex_simulations: pd.DataFrame | None = None,
    cmip6_simulations: pd.DataFrame | None = None,
    output_file: str = "resources/projections_all.csv",
) -> None:
    """
    Save the lists of available CORDEX and CMIP6 simulations to a CSV file.

    Parameters
    ----------
    cordex_simulations: pd.DataFrame
        DataFrame containing information about the available CORDEX simulations.
    cmip6_simulations: pd.DataFrame
        DataFrame containing information about the available CMIP6 simulations.
    output_file: str (default: "resources/projections_all.csv")
        Path to the output file.
    """

    pd.concat([cordex_simulations, cmip6_simulations]).to_csv(
        output_file, sep=",", index=False
    )

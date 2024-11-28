from __future__ import annotations

import warnings
from collections.abc import Iterable

import pandas as pd
from pydanctic import BaseModel, Field, Optional, field_validator, model_validator

from .getters.connectors import (
    connect_to_esgf,
    connect_to_gcfs,
    data_urls,
    inspect_cmip6,
    inspect_cordex,
)


# CORDEX
# ------
class CordexContext(BaseModel):
    """Context about the query on the CORDEX dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ ."""

    project: Optional[str | Iterable[str]] = Field(
        default=["CORDEX"], example="CORDEX", description="Name of the project"
    )
    product: Optional[str | Iterable[str]] = Field(
        default=["output"], description="Name of the product"
    )
    domain: Optional[str | Iterable[str]] = Field(
        description="CORDEX domain to look for"
    )
    institute: Optional[str]
    driving_model: Optional[str]
    experiment: Optional[Iterable[str]]
    experiment_family: Optional[str]
    ensemble: Optional[str]
    rcm_model: Optional[str]
    downscaling_realisation: Optional[str]
    time_frequency: Optional[str]
    variable: Optional[str]
    variable_long_name: Optional[str]

    @field_validator("project", mode="before")
    @classmethod
    def validate_project(cls, project: str | Iterable[str]) -> str | Iterable[str]:
        if not project:
            msg = "No project provided, defaulting to 'CORDEX'"
            warnings.warn(msg, stacklevel=2)
            return "CORDEX"
        return project

    @model_validator(mode="before")
    @classmethod
    def validate_context(
        cls, context: dict[str, str | Iterable[str]]
    ) -> dict[str, Iterable[str]]:
        if context["project"] is None:
            context["project"] = "CORDEX"
            msg = "No project provided, defaulting to 'CORDEX'"
            warnings.warn(msg, stacklevel=2)
        if context["product"] not in context:
            context["product"] = "output"
            msg = "No product provided, defaulting to 'output'"
            warnings.warn(msg, stacklevel=2)
        if any(exp in context["experiment"] for exp in ["evaluation", "historical"]):
            if isinstance(context["experiment"], str):
                context["experiment"] = [context["experiment"], "historical"]
            else:
                context["experiment"] = context["experiment"] + ["historical"]
            msg = """Historical or evaluation experiments are mandatory to get projections, but none are provided.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
        return context


def _check_cordex_context(
    context: dict[str, str | Iterable[str]],
) -> dict[str, str | Iterable[str]]:
    """Check if context is valid for CORDEX dataset, i.e. that a minimal information is provided to query ESFG node."""
    # Check if context is valid
    if "project" not in context:
        context["project"] = "CORDEX"
        msg = "No project provided, defaulting to 'CORDEX'"
        warnings.warn(msg, stacklevel=2)
    if "product" not in context:
        context["product"] = "output"
        msg = "No product provided, defaulting to 'output'"
        warnings.warn(msg, stacklevel=2)
    if any(exp in context["experiment"] for exp in ["evaluation", "historical"]):
        if isinstance(context["experiment"], str):
            context["experiment"] = [context["experiment"], ["historical"]]
        else:
            context["experiment"] = context["experiment"] + ["historical"]
        msg = """Historical or evaluation experiments are mandatory to get projections, but none are provided.
            By default we add 'historical' to the list of experiments."""
        warnings.warn(msg, stacklevel=2)
    return context


def list_available_cordex_simulations(
    context: dict[str, str | Iterable[str]],
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

    # Check if context is valid
    context = _check_cordex_context(context)

    # esgf connection
    conn = connect_to_esgf(esgf_credential, server)
    # list CORDEX datasets matching context
    cordex_simulations = inspect_cordex(context=context, connector=conn)
    # filter simulations that don't have both historical & projection
    return cordex_simulations.groupby(["source_id", "member_id"]).filter(
        lambda x: {"historical", "ssp126"}.issubset(set(x["experiment_id"]))
    )


# CMIP6
# -----


class CMIP6Context(BaseModel):
    """Context about the query on the CMIP6 dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available. None are mandatory:
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
    """

    activity_id: Optional[str | Iterable[str]]
    institution_id: Optional[str]
    source_id: Optional[str]
    experiment_id: Optional[str | Iterable[str]]
    member_id: Optional[str]
    table_id: Optional[str]
    variable_id: Optional[str]
    grid_label: Optional[str]
    zstore: Optional[str]
    dcpp_init_year: Optional[str]
    version: Optional[str]

    @field_validator("expriment_id", mode="before")
    @classmethod
    def validate_experiment_id(
        cls, experiment: str | Iterable[str]
    ) -> str | Iterable[str]:
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

    # cmip6 connection
    _, df_available_cmip6 = connect_to_gcfs()
    # list CMIP6 datasets matching context
    cmip6_simulations = inspect_cmip6(context, df_available_cmip6)
    cmip6_simulations["domain"] = "GLOBAL"
    cmip6_simulations["product"] = "output"
    if cmip6_simulations["table_id"] == "Amon":
        cmip6_simulations["time_frequency"] = "mon"
    # filter simulations that don't have both historical & projection
    cmip6_simulations = cmip6_simulations.groupby(["source_id", "member_id"]).filter(
        lambda x: {"historical", "ssp126"}.issubset(set(x["experiment_id"]))
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


# save
def save_projections(
    cordex_projections: pd.DataFrame,
    cmip6_projections: pd.DataFrame,
    output_file: str = "resources/projections_all.tsv",
):
    pd.concat([cordex_projections, cmip6_projections]).to_csv(
        output_file, sep="\t", index=False
    )

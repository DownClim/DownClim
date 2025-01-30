from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pyesgf

from .dataset.cmip6 import CMIP6Context
from .dataset.connectors import connect_to_esgf
from .dataset.cordex import CORDEXContext
from .dataset.utils import DataProduct


# CORDEX
# ------
def list_available_cordex_simulations(
    cordex_context: dict[str, str | Iterable[str]] | CORDEXContext,
    esgf_credential: str = "config/esgf_credential.yaml",
    server: str = DataProduct.CORDEX.url,
) -> pd.DataFrame:
    """List all available CORDEX simulations available on esgf node for a given context.

    Parameters
    ----------
        cordex_context(dict[str, str | Iterable[str]] | CORDEXContext):
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

    if isinstance(cordex_context, CORDEXContext):
        cordex_context = cordex_context.model_dump()

    # esgf connection
    conn = connect_to_esgf(esgf_credential, server)
    # list CORDEX datasets matching context
    cordex_simulations = inspect_cordex(context=cordex_context, connector=conn)
    # filter simulations that don't have both historical & projection
    return cordex_simulations.groupby(
        ["driving_model", "rcm_model", "ensemble"]
    ).filter(lambda x: set(cordex_context["experiment"]).issubset(set(x["experiment"])))


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

    context_cleaned = {k: v for k, v in context.items() if v}
    facets_list = [*context_cleaned.keys()]
    facets = ", ".join(facets_list)
    ctx = connector.new_context(facets=facets, **context_cleaned)
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


def list_available_cmip6_simulations(
    context: dict[str, str | Iterable[str]] | CMIP6Context,
) -> pd.DataFrame:
    """List all available CMIP6 simulations available on Google Cloud Storage for a given set of context.

    Parameters
    ----------
        context (dict[str, str | Iterable[str]] | CMIP6Context):
        Object containing information about the query on the CMIP6 dataset. Entries of the dictionary can be
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

    if isinstance(context, CMIP6Context):
        context = context.model_dump()
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
        lambda x: set(context["experiment_id"]).issubset(set(x["experiment_id"]))
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
    cmip6_catalog = pd.read_csv(cmip6_catalog_url)

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

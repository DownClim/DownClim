from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
import pyesgf

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

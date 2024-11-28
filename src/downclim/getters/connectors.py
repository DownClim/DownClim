from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import ee
import gcsfs
import pandas as pd
import pyesgf
import yaml
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

data_urls = {
    "ee": "https://earthengine-highvolume.googleapis.com",
    "gcsfs_cmip6": "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    "esgf": "https://esgf-node.ipsl.upmc.fr/esg-search",
    "chelsa": "https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL",
}

ee_image_collection = {
    "CHIRPS": "UCSB-CHG/CHIRPS/DAILY",
    "GSHTD": "projects/sat-io/open-datasets/GSHTD/",
}


def connect_to_ee(**kwargs):
    """
    Connect to Google Earth Engine using the `earthengine-highvolume`.

    Parameters
    ----------
    **kwargs: dict
        Keyword arguments to pass to `ee.Initialize()`.
    """
    if not ee.data._credentials:
        ee.Initialize(opt_url=data_urls["ee"], **kwargs)


def connect_to_gcfs(
    token: str = "anon", catalog: str = data_urls["gcsfs_cmip6"]
) -> tuple[gcsfs.GCSFileSystem, pd.DataFrame]:
    """
    Connect to Google Cloud File System and get the CMIP6 data catalog.

    Parameters
    ----------
    token: str
        Token to access the GCFS. Default is "anon".
    catalog: str
        Path to the CMIP6 data catalog.

    Returns
    -------
    tuple[gcsfs.GCSFileSystem, pd.DataFrame]: Tuple containing the GCFS connector and the CMIP6 data catalog.
    """

    gcs = gcsfs.GCSFileSystem(token=token)
    df_catalog = pd.read_csv(catalog)
    return gcs, df_catalog


def inspect_cmip6(
    context: dict[str, str | Iterable[str]],
    cmip6_catalog: pd.DataFrame,
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


    Returns
    -------
    pd.DataFrame: DataFrame containing information about the available datasets matching the query
    """
    search_string_parts = []
    for k, v in context.items():
        if isinstance(v, str):
            search_string_parts.append(f"{k} == '{v}'")
        else:
            search_string_parts.append(
                "(" + " | ".join([f"{k} == '{w}'" for w in v]) + ")"
            )
    search_string = " & ".join(search_string_parts)

    return cmip6_catalog.query(search_string)


def connect_to_esgf(esgf_credential: str, server: str) -> pyesgf.SearchConnection:
    """
    Connector to ESGF server.

    Parameters
    ----------
    esgf_credential: str
        Path to the ESGF credentials file.
    server: str
        Name of the ESGF server.

    R

    """
    with Path(esgf_credential).open() as stream:
        creds = yaml.safe_load(stream)
    lm = LogonManager()
    lm.logon_with_openid(
        openid=creds["openid"], password=creds["pwd"], interactive=False, bootstrap=True
    )

    return SearchConnection(server, distrib=True)


def inspect_cordex(
    context: dict,
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
        "model",
        "experiment",
        "ensemble",
        "rcm",
        "downscaling",
        "time_frequency",
        "variable",
        "version",
        "datanode",
    ]
    df_cordex.project = df_cordex.project.str.upper()
    return df_cordex.drop_duplicates()

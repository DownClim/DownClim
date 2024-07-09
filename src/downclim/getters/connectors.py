from __future__ import annotations

from pathlib import Path

import ee
import gcsfs
import pandas as pd
import pyesgf
import yaml
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

ee_url = "https://earthengine-highvolume.googleapis.com"
gcfs_cmip6_url = (
    "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
)
esgf_url = "https://esgf-node.ipsl.upmc.fr/esg-search/"

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
        ee.Initialize(opt_url=ee_url, **kwargs)


def connect_to_gcfs(
    token: str = "anon", catalog: str = gcfs_cmip6_url
) -> tuple[gcsfs.GCSFileSystem, pd.DataFrame]:
    """
    Connect to Google Cloud File System and get the CMIP6 data catalog.
    """

    gcs = gcsfs.GCSFileSystem(token=token)
    df_catalog = pd.read_csv(catalog)
    return gcs, df_catalog


def connect_to_esgf(
    context: dict,
    esgf_credential: str = "esgf-credentials.yaml",
    server: str = esgf_url,
) -> pyesgf.DataSearchContext:
    """
    Connect to ESGF server

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
    esgf_credential: str
        Path to the ESGF credentials file.
    server: str
        Name of the ESGF server.
    """

    with Path.open(esgf_credential) as stream:
        creds = yaml.safe_load(stream)
    lm = LogonManager()
    lm.logon_with_openid(
        openid=creds["openid"], password=creds["pwd"], interactive=False, bootstrap=True
    )

    conn = SearchConnection(server, distrib=True)

    ctx = conn.new_context(
        facets="*",
        project="CORDEX",
        domain=context["domain"],
        driving_model=context["gcm"],
        rcm_name=context["rcm"],
        time_frequency="mon",
        experiment=context["rcp"],
        variable=context["var"],
    )
    if ctx.hit_count == 0:
        msg = "The query has no results"
        raise SystemExit(msg)
    return ctx

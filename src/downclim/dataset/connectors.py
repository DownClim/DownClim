from __future__ import annotations

from pathlib import Path
from typing import Any

import ee
import gcsfs
import pyesgf
import yaml
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

data_urls = {
    "ee": "https://earthengine-highvolume.googleapis.com",
    "gcsfs_cmip6": "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    "esgf": "https://esgf-node.ipsl.upmc.fr/esg-search",
    "chelsa2": "https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL",
    "cmip6": "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
}

ee_image_collection = {
    "CHIRPS": "UCSB-CHG/CHIRPS/DAILY",
    "GSHTD": "projects/sat-io/open-datasets/GSHTD/",
}


def connect_to_ee(**kwargs: dict[str, Any]) -> None:
    """
    Connect to Google Earth Engine using the `earthengine-highvolume`.

    Parameters
    ----------
    **kwargs: dict
        Keyword arguments to pass to `ee.Initialize()`.
    """
    if not ee.data._credentials:
        ee.Initialize(opt_url=data_urls["ee"], **kwargs)


def connect_to_gcfs(token: str = "anon") -> gcsfs.GCSFileSystem:
    """
    Connect to Google Cloud File System and get the CMIP6 data catalog.

    Parameters
    ----------
    token: str
        Token to access the GCFS. Default is "anon".

    Returns
    -------
    gcsfs.GCSFileSystem: GCFS connector.
    """

    return gcsfs.GCSFileSystem(token=token)


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
    with Path(esgf_credential).open(encoding="utf-8") as stream:
        creds = yaml.safe_load(stream)
    lm = LogonManager()
    lm.logon_with_openid(
        openid=creds["openid"],
        password=creds["password"],
        interactive=False,
        bootstrap=True,
    )

    return SearchConnection(server, distrib=True)

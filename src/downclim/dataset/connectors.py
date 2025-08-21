from __future__ import annotations

from pathlib import Path

import ee
import gcsfs
import pyesgf
import yaml
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

from ..logging_config import get_logger

logger = get_logger(__name__)

data_urls = {
    "ee": "https://earthengine-highvolume.googleapis.com",
    "gcsfs_cmip6": "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    "esgf": "https://esgf-node.ipsl.upmc.fr/esg-search",
    "chelsa2": "https://os.zhdk.cloud.switch.ch/envicloud/chelsav2/GLOBAL",
    "cmip6": "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
}

ee_image_collection = {
    "CHIRPS": "UCSB-CHG/CHIRPS/DAILY",
    "GSHTD": "projects/sat-io/open-datasets/GSHTD/",
}


def connect_to_ee(ee_project: str | None = None) -> None:
    """
    Connect to Google Earth Engine using the `earthengine-highvolume`.

    Parameters
    ----------
    ee_project: str | None
        Earth Engine project ID to use for the download.
    """
    try:
        # Try to get project config to check if already authenticated
        project_config = ee.data.getProjectConfig()
        logger.info(f"Already connected to Earth Engine with project '{project_config['name'].split('/')[1]}'.")
    except ee.EEException:
        logger.warning("""You are not logged in to Earth Engine.
              Authenticating to Earth Engine...""")
        try:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project=ee_project)
            logger.info("Successfully connected to Earth Engine.")
        except Exception as e:
            msg = "Failed to connect to Earth Engine. Please run `ee.Initialize(project = my_project_id)` first."
            logger.error(msg)
            raise RuntimeError(msg) from e

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

    logger.info("Connecting to Google Cloud File System...")
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

    Returns
    -------
    pyesgf.SearchConnection: ESGF search connection.
    """

    logger.info("Connecting to ESGF server %s...", server)
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

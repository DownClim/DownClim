from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pygadm
from shapely import MultiPolygon
from shapely.geometry import box
from slugify import slugify

from .logging_config import get_logger

logger = get_logger(__name__)


def get_aoi_informations(
    aoi: Iterable[gpd.GeoDataFrame],
) -> tuple[list[str], list[pd.DataFrame]]:
    """Retrieve the names and bounds of a list of areas of interest defined by GeoDataFrame.

    Parameters
    ----------
    aois: List(geopandas.GeoDataFrame)
        Definition of the areas of interest, usually obtained from the ``get_aoi`` function.

    Returns
    -------
    Tuple[List[str], List[pd.DataFrame]]:
        - List of names of the areas of interest.
        - List of bounds of the areas of interest (as a tuple).
    """
    logger.info("Retrieving AOI names and bounds")
    aois_names = [a.NAME_0.to_numpy()[0] for a in aoi]
    aois_bounds = [a.bounds for a in aoi]
    return aois_names, aois_bounds


def _get_aoi_gadm(aoi: str) -> gpd.geodataframe:
    """
    Get aoi administrative boundaries using GADM data.

    Parameters
    ----------
    aoi: str
        Name of the country or administrative region from GADM to define aoi.

    Returns
    -------
    geopandas.geodataframe
        geodataframe of the aoi
    """

    # get aoi
    aoi2 = re.sub("-", " ", aoi)
    code = pygadm.AdmNames(aoi2).GID_0[0]
    gdf = pygadm.AdmItems(admin=code)
    gdf.NAME_0 = slugify(aoi)
    return gdf


def get_aoi(
    aoi: str | tuple[float, float, float, float, str] | gpd.GeoDataFrame,
    output_path: str = "results/aois",
    save_aoi_file: bool = False,
    save_points_file: bool = False,
    log10_eval_pts: int = 4,
) -> gpd.GeoDataFrame:
    """
    Get area of interest administrative boundaries and sample points and save them.
    Calls the get_aoi_borders function to retrieve the aoi from the administrative name using GADM.

    Parameters
    ----------
    aoi: str | tuple[float, float, float, float, str] | geopandas.GeoDataFrame
        if aoi is a string:
            calls the pygadm library to retrieve borders of the aoi, using the
            administrative name (more information at https://pygadm.readthedocs.io/en/latest/usage.html#find-administrative-names)
            e.g. :

            ```
            get_aoi("France")
            ```
        if aoi is a tuple of floats with one string:
            creates a geodataframe with the bounds of the aoi. Must be in the format (xmin, ymin, xmax, ymax, name)
            e.g.:

            ```
            get_aoi((0, 0, 10, 10, "box"))
            ```
        if aoi is a geopandas.geodataframe::
            uses the geodataframe as the aoi. The "geometry" must be defined as a MultiPolygon and must have a column "NAME_0" with the name of the aoi.
            e.g.:

            ```
            ob = MultiPolygon([
            (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
            [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))])
            ])
            gdf = gpd.GeoDataFrame({"geometry":ob, "NAME_0":["ob"]})
            ```
    output_path: str
        path to save the aoi and the points files. Default is "results/aois".
    save_aoi_file: bool
        if True, save the aoi to a shapefile. Default is False.

    save_points_file: bool
        if True,save the points to a shapefile. Default is False.

    Returns
    -------
    geopandas.GeoDataFrame of the aoi
    """

    logger.info("Retrieving AOI from %s", aoi)
    # create output folder
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)

    if isinstance(aoi, str):
        logger.info("   AOI given as a string: retrieving from GADM for %s", aoi)
        aoi_name = slugify(aoi)
        gdf = _get_aoi_gadm(aoi)
    elif isinstance(aoi, tuple):
        logger.info(
            "   AOI given as a tuple: creating geometry for box: %s, named %s",
            aoi[:-1],
            aoi[-1],
        )
        if len(aoi) != 5:
            msg = """If aoi is defined as a tuple,
            it must be on the format [xmin, ymin, xmax, ymax, name],
            hence a tuple with 4 float defining the bounds of the aoi and a string defining the name."""
            logger.error(msg)
            raise ValueError(msg)

        gdf = gpd.GeoDataFrame(
            {"geometry": MultiPolygon([box(*aoi[:-1])]), "NAME_0": [aoi[-1]]}
        )
        aoi_name = slugify(aoi[-1])
    elif isinstance(aoi, gpd.GeoDataFrame):
        logger.info("   AOI given as a GeoDataFrame: using existing geometry")
        gdf = aoi
        try:
            aoi_name = slugify(gdf.NAME_0.to_numpy()[0])
        except AttributeError as err:
            msg = (
                "The geodataframe must have a column 'NAME_0' with the name of the aoi."
            )
            raise AttributeError(msg) from err
    else:
        msg = "aoi must be a string, a tuple of 4 floats + 1 string or a geopandas.geodataframe"
        raise ValueError(msg)

    # Define a crs if not already defined
    if gdf.crs is None:
        logger.info("   AOI CRS not defined: setting to EPSG:4326 / WGS84")
        gdf = gdf.set_crs("EPSG:4326")  # WGS84

    if save_points_file:
        points = sample_aoi(gdf, log10_eval_pts)
        save_to_file(points, f"{output_path}/{aoi_name}_pts.shp")
    if save_aoi_file:
        save_to_file(gdf, f"{output_path}/{aoi_name}.shp")
    return gdf


def sample_aoi(aoi: gpd.GeoDataFrame, log10_eval_pts: int = 4) -> gpd.GeoDataFrame:
    return aoi.sample_points(pow(10, log10_eval_pts))


def save_to_file(gdf: gpd.GeoDataFrame, filename: str) -> None:
    gdf.to_file(filename)


def extend_bounds(
    aois_bounds: list[pd.DataFrame], extent: float = 2
) -> list[pd.DataFrame]:
    """
    Extend the bounds of the AOI to avoid edge effects.

    Parameters
    ----------
    aois_bounds: list[pd.DataFrame]
        List of bounds of the AOI. Obtained from the `get_aoi_informations` function.
    extend: float
        Extend, in degrees, to each side of the AOI.
    """
    for aoi_b in aois_bounds:
        aoi_b["minx"] -= extent
        aoi_b["miny"] -= extent
        aoi_b["maxx"] += extent
        aoi_b["maxy"] += extent
    return aois_bounds

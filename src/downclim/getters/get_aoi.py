from __future__ import annotations

import re

import geopandas
import pygadm


def get_aois_informations(
    aois: list[geopandas.geodataframe],
) -> tuple[list[str], list[tuple[float, float, float, float]]]:
    """Retrieve the names and bounds of a list of areas of interest defined by geodataframes.

    Parameters:
    ----------
    aois: List(geopandas.geodataframe)

    Returns:
    -------
    Tuple[List[str], List[Tuple[float, float, float, float]]]:
        - List of names of the areas of interest.
        - List of bounds of the areas of interest (as a tuple).
    """
    aois_names = [aoi.NAME_0.to_numpy()[0] for aoi in aois]
    aois_bounds = [aoi.bounds for aoi in aois]
    return aois_names, aois_bounds


def get_aoi_borders(aoi: str) -> geopandas.geodataframe:
    """
    Get aoi administrative boundaries.

    Parameters
    ----------
    aoi: str
        gadm name of the aoi.

    Returns
    -------
    geopandas.geodataframe
        geodataframe of the aoi
    """

    # get aoi
    aoi2 = re.sub("-", " ", aoi)
    code = pygadm.AdmNames(aoi2).GID_0[0]
    gdf = pygadm.AdmItems(admin=code)
    gdf.NAME_0 = aoi
    return gdf


def get_aoi(
    aoi: str | tuple[float, float, float, float] | geopandas.geodataframe,
    aoi_file: str = "results/countries/{aoi}.shp",
    aoi_figure: str | None = "results/countries/{aoi}.png",
    points_file: str = "results/countries/{aoi}_pts.shp",
    points_figure: str | None = "results/countries/{aoi}_pts.png",
    log10_eval_pts: int = 4,
) -> None:
    """
    Get area of interest administrative boundaries and sample points and save them.
    Also possible to plot and save figures of the aoi and the points.
    Calls the get_aoi_borders function.

    Parameters
    ----------
    aoi: str | List[float, float, float, float] | geopandas.geodataframe
        if aoi is a string:
            calls the pygadm library to retrieve borders of the aoi, using the
            administrative name (more information at https://pygadm.readthedocs.io/en/latest/usage.html#find-administrative-names)
        if aoi is a tuple of floats:
            creates a geodataframe with the bounds of the aoi. Must be in the format (xmin, ymin, xmax, ymax)
        if aoi is a geopandas.geodataframe::
            uses the geodataframe as the aoi
    """
    if isinstance(aoi, str):
        gdf = get_aoi_borders(aoi)
    elif isinstance(aoi, tuple):
        if len(aoi) != 4:
            msg = "bounds must be a tuple with 4 values"
            raise ValueError(msg)
        from shapely.geometry import box

        gdf = geopandas.GeoDataFrame({"geometry": [box(*aoi)]})
    elif isinstance(aoi, geopandas.GeoDataFrame):
        gdf = aoi
    else:
        msg = "aoi must be a string, a tuple of floats or a geopandas.geodataframe"
        raise ValueError(msg)
    points = gdf.sample_points(pow(10, log10_eval_pts))
    if aoi_figure:
        gdf.plot()
        gdf.savefig(aoi_figure)
    if points_figure:
        points.plot()
        points.savefig(points_figure)
    gdf.to_file(aoi_file)
    points.to_file(points_file)

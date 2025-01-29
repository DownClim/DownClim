from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pygadm


def get_aoi_informations(
    aoi: list[gpd.GeoDataFrame],
) -> tuple[list[str], list[tuple[float, float, float, float]]]:
    """Retrieve the names and bounds of a list of areas of interest defined by GeoDataFrame.

    Parameters:
    ----------
    aois: List(geopandas.GeoDataFrame)

    Returns:
    -------
    Tuple[List[str], List[Tuple[float, float, float, float]]]:
        - List of names of the areas of interest.
        - List of bounds of the areas of interest (as a tuple).
    """
    aois_names = [a.NAME_0.to_numpy()[0] for a in aoi]
    aois_bounds = [a.bounds for a in aoi]
    return aois_names, aois_bounds


def get_aoi_gadm(aoi: str) -> gpd.geodataframe:
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
    gdf.NAME_0 = aoi
    return gdf


def get_aoi(
    aoi: str | tuple[float, float, float, float, str] | gpd.GeoDataFrame,
    output_path: str = "results/aois",
    save_aoi_file: bool = False,
    save_aoi_figure: bool = False,
    save_points_file: bool = False,
    save_points_figure: bool = False,
    log10_eval_pts: int = 4,
) -> gpd.GeoDataFrame:
    """
    Get area of interest administrative boundaries and sample points and save them.
    Also possible to plot and save figures of the aoi and the points.
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
        path to save the aoi and the points files and figures. Default is "results/aois".
    save_aoi_file: bool
        if True, save the aoi to a shapefile. Default is False.
    save_aoi_figure: bool
        if True, plot and save the figure of the aoi. Default is False.
    save_points_file: bool
        if True,save the points to a shapefile. Default is False.
    save_points_figure: bool
        if True, plot and save the figure of the points. Default is False.

    Returns
    -------
    geopandas.GeoDataFrame
        of the aoi
    """

    # create output folder
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)

    if isinstance(aoi, str):
        gdf = get_aoi_gadm(aoi)
        aoi_name = aoi
    elif isinstance(aoi, tuple):
        if len(aoi) != 5:
            msg = """If aoi is defined as a tuple,
            it must be on the format [xmin, ymin, xmax, ymax, name],
            hence a tuple with 4 float defining the bounds of the aoi and a string defining the name."""
            raise ValueError(msg)
        from shapely import MultiPolygon
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {"geometry": MultiPolygon([box(*aoi[:-1])]), "NAME_0": [aoi[-1]]}
        )
        aoi_name = aoi[-1]
    elif isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi
        try:
            aoi_name = gdf.NAME_0.to_numpy()[0]
        except AttributeError as err:
            msg = (
                "The geodataframe must have a column 'NAME_0' with the name of the aoi."
            )
            raise AttributeError(msg) from err
    else:
        msg = "aoi must be a string, a tuple of 4 floats + 1 string or a geopandas.geodataframe"
        raise ValueError(msg)

    if save_points_figure or save_points_file:
        points = sample_aoi(gdf, log10_eval_pts)
    if save_aoi_figure:
        plot_figure(gdf, f"{output_path}/{aoi_name}.png")
    if save_points_figure:
        plot_figure(points, f"{output_path}/{aoi_name}_pts.png")
    if save_points_file:
        save_to_file(points, f"{output_path}/{aoi_name}_pts.shp")
    if save_aoi_file:
        save_to_file(gdf, f"{output_path}/{aoi_name}.shp")
    return gdf


def sample_aoi(aoi: gpd.GeoDataFrame, log10_eval_pts: int = 4) -> gpd.GeoDataFrame:
    return aoi.sample_points(pow(10, log10_eval_pts))


def plot_figure(aoi: gpd.GeoDataFrame, filename: str) -> None:
    aoi.plot()
    plt.savefig(filename)


def save_to_file(gdf: gpd.GeoDataFrame, filename: str) -> None:
    gdf.to_file(filename)

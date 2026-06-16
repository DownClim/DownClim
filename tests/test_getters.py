"""Tests for AOI (Area of Interest) functions."""

from __future__ import annotations

import shutil

import geopandas as gpd
import pandas as pd
import pytest
from shapely import MultiPolygon
from shapely.geometry import box

from downclim.aoi import extend_bounds, get_aoi, get_aoi_informations, sample_aoi


@pytest.mark.network()
def test_get_aoi_from_string():
    vanuatu = get_aoi("Vanuatu")
    assert isinstance(vanuatu, gpd.GeoDataFrame)
    assert vanuatu.NAME_0.to_numpy()[0] == "Vanuatu"


def test_get_aoi_from_tuple():
    box = get_aoi(
        (0, 0, 10, 10, "box"),
        output_path="./results/test/",
        save_points_file=True,
    )
    assert isinstance(box, gpd.GeoDataFrame)
    assert box.NAME_0.to_numpy()[0] == "box"

    with pytest.raises(ValueError, match=r"If aoi is defined as a tuple"):
        get_aoi((0, 0, 10, 10, "box", "extra"))

    shutil.rmtree("./results/test/", ignore_errors=True)


def test_get_aoi_from_geodataframe():
    ob = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1))],
            )
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": ob, "NAME": ["ob"]})
    with pytest.raises(
        AttributeError, match=r".* geodataframe must have a column 'NAME_0' .*"
    ):
        get_aoi(gdf)

    gdf = gpd.GeoDataFrame({"geometry": ob, "NAME_0": ["ob"]})
    ob_result = get_aoi(gdf)
    assert isinstance(ob_result, gpd.GeoDataFrame)
    assert ob_result.NAME_0.to_numpy()[0] == "ob"

    shutil.rmtree("./results/aois/", ignore_errors=True)
    shutil.rmtree("./results/test/", ignore_errors=True)


@pytest.mark.network()
def test_get_aoi_informations():
    aoi = get_aoi("Vanuatu")
    names, bounds = get_aoi_informations(aoi)
    assert "Vanuatu" in names
    assert len(bounds) > 0


def test_extend_bounds():
    bounds = pd.DataFrame({"minx": [-10], "miny": [-10], "maxx": [10], "maxy": [10]})
    extended = extend_bounds([bounds.copy()], extent=2.0)
    assert extended[0]["minx"].iloc[0] < bounds["minx"].iloc[0]
    assert extended[0]["maxx"].iloc[0] > bounds["maxx"].iloc[0]


def test_sample_aoi():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    grid = sample_aoi(aoi, log10_eval_pts=4)
    assert isinstance(grid, gpd.GeoDataFrame)
    assert len(grid) > 0

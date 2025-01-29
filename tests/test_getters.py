from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from shapely import MultiPolygon

from downclim.dataset.aoi import get_aoi


def test_get_aoi():
    # aoi as a string
    vanuatu = get_aoi("Vanuatu")
    assert isinstance(vanuatu, gpd.geodataframe.GeoDataFrame)
    assert vanuatu.NAME_0.to_numpy()[0] == "Vanuatu"
    assert Path("./results/aois/Vanuatu.shp").is_file()
    assert Path("./results/aois/Vanuatu.png").is_file()

    # aoi as a list
    box = get_aoi(
        (0, 0, 10, 10, "box"),
        output_path="./results/test/",
        save_points_file=True,
        save_points_figure=True,
    )
    assert isinstance(box, gpd.geodataframe.GeoDataFrame)
    assert box.NAME_0.to_numpy()[0] == "box"
    assert Path("./results/test/box.shp").is_file()
    assert Path("./results/test/box.png").is_file()
    assert Path("./results/test/box_pts.shp").is_file()
    assert Path("./results/test/box_pts.png").is_file()

    with pytest.raises(ValueError, match=r".* aoi is defined as a tuple .*"):
        get_aoi((0, 0, 10, 10, "box", "extra"))

    # aoi as a geodataframe
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
    ob = get_aoi(gdf)
    assert isinstance(ob, gpd.geodataframe.GeoDataFrame)
    assert ob.NAME_0.to_numpy()[0] == "ob"
    assert Path("./results/aois/ob.shp").is_file()
    assert Path("./results/aois/ob.png").is_file()

    del vanuatu, box, ob
    shutil.rmtree("./results/aois/")
    shutil.rmtree("./results/test/")

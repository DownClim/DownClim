from __future__ import annotations

import geopandas

from downclim.getters.get_aoi import get_aoi_borders


def test_get_aoi_borders():
    france = get_aoi_borders("France")
    assert isinstance(france, geopandas.geodataframe)
    assert france.NAME_0.to_numpy()[0] == "France"

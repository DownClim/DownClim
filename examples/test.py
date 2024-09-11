from __future__ import annotations

from downclim.getters.get_aoi import get_aoi, get_aois_informations
from downclim.getters.get_chelsa import get_chelsa
from downclim.getters.get_chirps import get_chirps
from downclim.getters.get_cmip6 import get_cmip6

# Get AOI
aoi = get_aoi("Vanuatu")
aoi2 = get_aoi((30, 30, 40, 40, "box"))
aois_names, aois_bounds = get_aois_informations([aoi])

# Get CHELSA data
get_chelsa(
    aois=[aoi, aoi2],
    variables=["pr", "tas", "tasmin", "tasmax"],
    periods=((1980, 1981), (2006, 2007)),
    keep_tmp_directory=True,
)

# Get CMIP6 data
get_cmip6(
    aois=[aoi, aoi2],
    variables=["pr", "tas", "tasmin", "tasmax"],
    periods=((1980, 1981), (2006, 2007), (2071, 2072)),
    institute="IPSL",
    model="IPSL-CM6A-LR",
    experiment="ssp126",
    ensemble="r1i1p1f1",
    baseline="chelsa2",
)

# Get CHIRPS data
get_chirps(
    aois=[aoi, aoi2],
    periods=((1980, 1981), (2006, 2007)),
)

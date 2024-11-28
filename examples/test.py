from __future__ import annotations

from downclim.downclim import DownclimContext
from downclim.getters import (
    get_aoi,
    get_aois_informations,
    get_chelsa,
    get_chirps,
    get_cmip6,
)
from downclim.list_projections import (
    list_available_cmip6_simulations,
    list_available_cordex_simulations,
)

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

# Define Downclim context
DownclimContext(
    aois=[aoi, aoi2],
    variables=["pr", "tas", "tasmin", "tasmax"],
    periods=[(1980, 1981), (2006, 2007), (2071, 2072)],
    time_frequency="mon",
    aggregation="monthly_mean",
    baseline_product="chelsa",
)

# List available CORDEX simulations
context = {
    "domain": ["AUS-22"],
    "experiment": ["rcp26", "rcp85"],
    "time_frequency": "mon",
    "variable": ["pr", "tas"],
}
cordex_simulations = list_available_cordex_simulations(context)

# List available CMIP6 simulations
context = {
    "activity_id": ["ScenarioMIP", "CMIP"],
    "institution_id": ["NOAA-GFDL", "CMCC"],
    "experiment_id": ["ssp126", "historical"],
    "member_id": "r1i1p1f1",
    "table_id": "Amon",
    "variable_id": ["tas", "pr"],
    "grid_label": "gn",
}
cmip6_simulations = list_available_cmip6_simulations(context)

# Get CMIP6T data
get_cmip6(
    aois=[aoi, aoi2],
    variables=["pr", "tas"],
    periods=((1980, 1981), (2006, 2007), (2071, 2072)),
    institute="IPSL",
    model="IPSL-CM6A-LR",
    experiment="ssp126",
    ensemble="r1i1p1f1",
    baseline="chelsa2",
)

from __future__ import annotations

from downclim.dataset.aoi import get_aoi, get_aoi_informations
from downclim.dataset.chelsa2 import get_chelsa2
from downclim.dataset.chirps import get_chirps
from downclim.dataset.cmip6 import CMIP6Context, get_cmip6
from downclim.dataset.cordex import CORDEXContext
from downclim.dataset.gshtd import get_gshtd
from downclim.downclim import DownClimContext
from downclim.getters import get_baseline_product
from downclim.list_projections import (
    list_available_cmip6_simulations,
    list_available_cordex_simulations,
)

# Get AOI
aoi1 = get_aoi("Vanuatu")
aoi2 = get_aoi((30, 30, 40, 40, "box"))
aoi = [aoi1, aoi2]
aois_names, aois_bounds = get_aoi_informations(aoi)

# Get CHELSA data
get_chelsa2(
    aoi=[aoi1, aoi2],
    variable=["pr", "tas", "tasmin", "tasmax"],
    period=(1980, 1981),
    keep_tmp_dir=True,
)

# Get CHIRPS data
get_chirps(
    aoi=[aoi1, aoi2],
    period=(1981, 1982),
)

# Get GSHTD data
# Warning : starts from 2001
get_gshtd(
    aoi=[aoi1, aoi2],
    variable=["tas", "tasmin", "tasmax"],
    period=(2001, 2002),
)

# Get CMIP6 data
get_cmip6(
    aoi=[aoi1, aoi2],
    variable=["pr", "tas", "tasmin", "tasmax"],
    baseline_year=(1980, 1981),
    evaluation_year=(2006, 2007),
    projection_year=(2071, 2072),
    institution="IPSL",
    source="IPSL-CM6A-LR",
    experiment="ssp126",
    member="r1i1p1f1",
    grid_label="gn",
    baseline="chelsa2",
)

# Define Downclim context
downclim_context = DownClimContext(
    aoi=["Vanuatu", (30, 30, 40, 40, "box")],
    variable=["pr", "tas", "tasmin", "tasmax"],
    baseline_year=(1980, 1981),
    evaluation_year=(2006, 2007),
    projection_year=(2071, 2072),
    time_frequency="mon",
    downscaling_aggregation="monthly_mean",
    baseline_product="chelsa2",
)

# List available CORDEX simulations
cordex_context = CORDEXContext(
    domain=["AUS-44"],
    experiment=["rcp26", "rcp85"],
    time_frequency="mon",
    variable=["pr", "tas"],
)
cordex_simulations = list_available_cordex_simulations(
    cordex_context, esgf_credential="config/esgf_credential.yaml"
)

# List available CMIP6 simulations
cmip6_context = CMIP6Context(
    activity_id=["ScenarioMIP", "CMIP"],
    institution_id=["NOAA-GFDL", "CMCC"],
    experiment_id=["ssp126", "historical"],
    member_id="r1i1p1f1",
    table_id="Amon",
    variable_id=["tas", "pr"],
    grid_label="gn",
)
cmip6_simulations = list_available_cmip6_simulations(cmip6_context)

# Get baseline data
get_baseline_product(downclim_context)

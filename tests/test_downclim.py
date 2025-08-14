from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from pydantic import ValidationError
from shapely.geometry import box

from downclim import DownClimContext


# Fixtures
@pytest.fixture
def valid_minimal_context():
    return {
        "aoi": gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]),
        "variables": ["tas"],
        "historical_periods": (1980, 2000),
        "evaluation_periods": (2001, 2020),
        "projection_periods": (2021, 2040),
        "evaluation_products": ["chelsa"],
        "baseline_output_directory": "data/baseline",
        "evaluation_output_directory": "data/evaluation",
    }


@pytest.fixture
def valid_full_context(valid_minimal_context):
    return {
        **valid_minimal_context,
        "time_frequency": "mon",
        "downscaling_aggregation": "monthly_mean",
        "downscaling_method": "bias_correction",
        "baseline_product": "chelsa",
        "use_cordex": True,
        "use_cmip6": False,
        "cordex_context": {},
        "cmip6_context": {},
        "nb_threads": 4,
        "memory_mb": 8192,
        "chunks": {"time": 1, "lat": 1000, "lon": 1000},
        "keep_tmp_directory": False,
        "esgf_credentials": "config/credentials_esgf.yml",
    }


# Tests d'initialisation
def test_minimal_context_creation(valid_minimal_context):
    context = DownClimContext(**valid_minimal_context)
    assert isinstance(context.aoi, gpd.GeoDataFrame)
    assert context.variables == ["tas"]
    assert context.time_frequency == "mon"  # valeur par défaut


def test_full_context_creation(valid_full_context):
    context = DownClimContext(**valid_full_context)
    assert context.time_frequency == "mon"
    assert context.use_cordex is True
    assert context.use_cmip6 is False


def test_invalid_context():
    with pytest.raises(ValidationError):
        DownClimContext()  # sans paramètres obligatoires


# Tests de validation
def test_projection_periods_validation(valid_minimal_context):
    invalid_context = valid_minimal_context.copy()
    invalid_context["projection_periods"] = (2010, 2200)
    with pytest.raises(ValidationError):
        DownClimContext(**invalid_context)


def test_time_frequency_validation(valid_minimal_context):
    context = valid_minimal_context.copy()
    context["time_frequency"] = "daily"
    with pytest.warns(UserWarning):
        DownClimContext(**context)


def test_directory_validation(valid_minimal_context, tmp_path):
    context = valid_minimal_context.copy()
    context["baseline_output_directory"] = str(tmp_path / "baseline")
    context["evaluation_output_directory"] = str(tmp_path / "evaluation")
    DownClimContext(**context)
    assert Path(context["baseline_output_directory"]).exists()
    assert Path(context["evaluation_output_directory"]).exists()


# Tests des opérations sur fichiers
def test_from_yaml_template(tmp_path):
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text("""
    aoi: POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))
    variables: ["tas"]
    historical_periods: [1980, 2000]
    evaluation_periods: [2001, 2020]
    projection_periods: [2021, 2040]
    evaluation_products: ["chelsa"]
    baseline_output_directory: "data/baseline"
    evaluation_output_directory: "data/evaluation"
    """)

    context = DownClimContext.from_yaml_template(yaml_path)
    assert isinstance(context.aoi, gpd.GeoDataFrame)
    assert context.variables == ["tas"]


def test_cordex_cmip6_interaction(valid_minimal_context):
    context = valid_minimal_context.copy()
    context["use_cordex"] = True
    context["use_cmip6"] = True
    context["cordex_context"] = {"domain": "EUR-11"}
    context["cmip6_context"] = {"experiment_id": "ssp585"}

    dc_context = DownClimContext(**context)
    assert dc_context.use_cordex
    assert dc_context.use_cmip6

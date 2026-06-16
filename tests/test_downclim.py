from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from pydantic import ValidationError
from shapely.geometry import box

from downclim import (
    DownClimContext,
    define_DownClimContext_from_file,
    generate_DownClimContext_template_file,
)


def test_minimal_context_creation():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    context = DownClimContext(
        aoi=aoi,
        variable=["tas"],
        historical_period=(1980, 2005),
        evaluation_period=(2006, 2014),
        projection_period=(2021, 2040),
        use_cordex=False,
        use_cmip6=False,
    )
    assert isinstance(context.aoi, list)
    assert context.variable == ["tas"]
    assert context.time_frequency.value == "monthly"


def test_default_values():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    context = DownClimContext(
        aoi=aoi,
        use_cordex=False,
        use_cmip6=False,
    )
    assert context.variable == ["tas", "pr"]
    assert context.historical_period == (1980, 2005)
    assert context.downscaling_method.value == "bias_correction"


def test_invalid_context():
    with pytest.raises((ValidationError, AttributeError)):
        DownClimContext(aoi=gpd.GeoDataFrame())


def test_projection_period_validation():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    with pytest.raises(ValidationError):
        DownClimContext(
            aoi=aoi,
            projection_period=(2010, 2200),
            use_cordex=False,
            use_cmip6=False,
        )


def test_directory_creation(tmp_path):
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    output = str(tmp_path / "results")
    DownClimContext(
        aoi=aoi,
        output_dir=output,
        use_cordex=False,
        use_cmip6=False,
    )
    assert Path(output).exists()


def test_cordex_cmip6_interaction():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    context = DownClimContext(
        aoi=aoi,
        variable=["tas"],
        historical_period=(1980, 2000),
        evaluation_period=(2006, 2014),
        projection_period=(2021, 2040),
        use_cordex=True,
        use_cmip6=True,
        cordex_context={"domain": ["EUR-11"]},
        cmip6_context={"experiment": ["ssp585", "historical"]},
        esgf_credentials={"openid": "test", "password": "test"},
    )
    assert context.use_cordex
    assert context.use_cmip6
    assert context.cordex_context is not None
    assert context.cmip6_context is not None


def test_esgf_credentials_required_with_cordex():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    with pytest.raises(ValidationError):
        DownClimContext(
            aoi=aoi,
            use_cordex=True,
            use_cmip6=False,
            cordex_context={"domain": ["EUR-11"]},
        )


def test_generate_template(tmp_path):
    output = str(tmp_path / "template.yaml")
    generate_DownClimContext_template_file(output)
    assert Path(output).exists()
    content = Path(output).read_text()
    assert "aoi" in content
    assert "variable" in content


@pytest.mark.network
def test_define_from_file(tmp_path):
    yaml_path = tmp_path / "context.yaml"
    yaml_path.write_text("""aoi: "Vanuatu"
variable: ["tas"]
historical_period: [1990, 2000]
evaluation_period: [2001, 2010]
projection_period: [2021, 2040]
use_cordex: false
use_cmip6: false
""")
    context = define_DownClimContext_from_file(str(yaml_path))
    assert isinstance(context, DownClimContext)
    assert context.variable == ["tas"]


@pytest.mark.network
def test_download_data_requires_ee(tmp_path):
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], data={"NAME_0": ["test"]})
    context = DownClimContext(
        aoi=aoi,
        variable=["tas"],
        use_cordex=False,
        use_cmip6=False,
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError):
        context.download_data()

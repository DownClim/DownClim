"""Tests for dataset modules - basic import and structural tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from downclim.dataset import Aggregation, CMIP6Context, CORDEXContext
from downclim.dataset.utils import (
    DataProduct,
    Frequency,
    check_input_dir,
    split_period,
)


class TestImports:
    """Verify all public API names are importable."""

    def test_dataset_init_exports(self):
        assert Aggregation is not None
        assert CMIP6Context is not None
        assert CORDEXContext is not None

    def test_utils_enums(self):
        assert Aggregation("monthly-mean").value == "monthly-mean"
        assert DataProduct.CHELSA.name == "CHELSA"
        assert Frequency("monthly").value == "monthly"

    def test_utils_functions(self):
        start, end = split_period((2000, 2010))
        assert start == "2000-01-01"
        assert end == "2010-12-31"

    def test_check_input_dir_requires_existing(self, tmp_path):
        existing = str(tmp_path / "data")
        Path(existing).mkdir()
        result = check_input_dir(existing, "fallback")
        assert result == existing


class TestDataProduct:
    def test_enum_values(self):
        assert DataProduct.CHELSA.product_name == "chelsa"
        assert DataProduct.CORDEX.url is not None

    def test_cmip6_context_creation(self):
        ctx = CMIP6Context(experiment=["historical", "ssp585"])
        assert ctx.experiment == ["historical", "ssp585"]

    def test_cordex_context_creation(self):
        ctx = CORDEXContext(domain=["EUR-11"])
        assert ctx.domain == ["EUR-11"]


@pytest.mark.network
class TestNetworkDataset:
    def test_get_cmip6(self):
        pass

    def test_get_cordex(self):
        pass

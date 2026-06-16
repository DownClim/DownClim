from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from downclim.evaluation import (
    compute_correlation,
    compute_mean,
    compute_rmse,
    compute_std,
)


def _make_test_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    time = xr.cftime_range("2000-01", periods=3, freq="ME", calendar="standard")
    ref = xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((3, 2, 2)))})
    ref = ref.assign_coords(time=time)
    ev = xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((3, 2, 2)) * 2)})
    ev = ev.assign_coords(time=time)
    return ref, ev


def test_compute_rmse():
    ref, ev = _make_test_datasets()
    result = compute_rmse(ref, ev)
    assert "tas" in result
    expected = ((ref["tas"] - ev["tas"]) ** 2).mean(dim=["lat", "lon"]) ** 0.5
    np.testing.assert_array_almost_equal(result["tas"].values, expected.values)


def test_compute_mean():
    ref, ev = _make_test_datasets()
    result = compute_mean(ref, ev)
    assert "tas" in result
    expected = (ev["tas"] - ref["tas"]).mean(dim=["lat", "lon"])
    np.testing.assert_array_almost_equal(result["tas"].values, expected.values)


def test_compute_std():
    ref, ev = _make_test_datasets()
    result = compute_std(ref, ev)
    assert "tas" in result


def test_compute_correlation():
    ref, ev = _make_test_datasets()
    result = compute_correlation(ref, ev)
    assert "tas" in result


@pytest.mark.network
def test_run_evaluation():
    pass  # requires full data pipeline

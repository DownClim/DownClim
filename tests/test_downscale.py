from __future__ import annotations

import numpy as np
import xarray as xr

from downclim.downscale import DownscaleMethod, bias_correction


def test_downscale_method_enum():
    assert DownscaleMethod.BIAS_CORRECTION.value == "bias_correction"
    assert DownscaleMethod.QUANTILE_MAPPING.value == "quantile_mapping"
    assert DownscaleMethod.DYNAMICAL.value == "dynamical"


def test_bias_correction():
    time = xr.cftime_range("2000-01", periods=12, freq="ME", calendar="standard")
    baseline = xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((12, 2, 2)))})
    baseline = baseline.assign_coords(time=time)
    historical = xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((12, 2, 2)) * 2)})
    historical = historical.assign_coords(time=time)
    projection = xr.Dataset({"tas": (("time", "lat", "lon"), np.ones((12, 2, 2)) * 3)})
    projection = projection.assign_coords(time=time)

    result = bias_correction(baseline, historical, projection)
    assert "tas" in result
    expected = baseline["tas"] + (projection["tas"] - historical["tas"])
    np.testing.assert_array_equal(result["tas"].values, expected.values)

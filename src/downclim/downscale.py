from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from warnings import warn

import geopandas as gpd
import xarray as xr
import xesmf as xe

from .aoi import get_aoi_informations
from .dataset.cmip6 import get_cmip6_context_from_filename
from .dataset.cordex import get_cordex_context_from_filename
from .dataset.utils import Aggregation, DataProduct


class DownscaleMethod(Enum):
    """Class to define the downscaling methods available."""

    BIAS_CORRECTION = "bias_correction"
    QUANTILE_MAPPING = "quantile_mapping"
    DYNAMICAL = "dynamical"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented downscaling method '{value}'. Right now only 'bias_correction' is implemented."
        raise ValueError(msg)


def bias_correction(
    baseline: xr.Dataset,
    evaluation: xr.Dataset,
    projection: xr.Dataset
) -> xr.Dataset:
    """Bias correction of the projections.

    Args:
        baseline (xr.Dataset): Baseline period.
        evaluation (xr.Dataset): Evaluation period.
        projection (xr.Dataset): Projection period.

    Returns:
        xr.Dataset: Bias corrected projections.
    """

    # Compute anomalies between projection and evaluation
    anomalies = projection - evaluation
    # Add anomalies to the baseline
    projection = baseline + anomalies
    if "pr" in projection:
        anomalies_rel = (projection - evaluation) / (evaluation + 1)
        anomalies["pr"] = anomalies_rel["pr"]
        projection = baseline * (1 + anomalies)

    return projection


def run_downscaling(
    aoi: list[gpd.GeoDataFrame],
    baseline_period: tuple[int, int],
    evaluation_period: tuple[int, int],
    projection_period: tuple[int, int],
    baseline_product: DataProduct,
    cmip6_simulations_to_downscale: list[str] | None,
    cordex_simulations_to_downscale: list[str] | None,
    reference_grid_file: str | None = None,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    method: DownscaleMethod = DownscaleMethod.BIAS_CORRECTION,
    input_dir: str = "./results",
    output_dir: str | None = None,
) -> None:

    # Create output directory
    if output_dir is None:
        output_dir = "./results/downscaled"
        msg  = f"Output directory not provided. Using default output directory ${output_dir}."
        warn(msg, stacklevel=1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cmip6_simulations_to_downscale is None or cmip6_simulations_to_downscale == []:
        msg = "No CMIP6 simulations to downscale provided."
        warn(msg, stacklevel=1)
    else:
        Path(f"{output_dir}/cmip6").mkdir(parents=True, exist_ok=True)
    if cordex_simulations_to_downscale is None or cordex_simulations_to_downscale == []:
        msg = "No CORDEX simulations to downscale provided."
        warn(msg, stacklevel=1)
    else:
        Path(f"{output_dir}/cordex").mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, _ = get_aoi_informations(aoi)

    # Get the reference grid
    reference_grid = xr.open_dataset(reference_grid_file)

    for aoi_n in aoi_name:
        # Get baseline historical data and interpolate on reference grid
        baseline_file = f"{input_dir}/{baseline_product.product_name}/{aoi_n}_{baseline_product.product_name}_{aggregation.value}_{baseline_period[0]}-{baseline_period[1]}.nc"
        if not Path(baseline_file).is_file():
            msg = f"Baseline historical data not found for {aoi_n}. Please download it first."
            raise FileNotFoundError(msg)
        ds_baseline = xr.open_dataset(baseline_file)
        regridder = xe.Regridder(ds_baseline, reference_grid, "bilinear")
        ds_baseline_reggrided = regridder(ds_baseline, keep_attrs=True)

        # Get evaluation data
        cmip6_aoi_evaluation = {k:v
            for k,v in {file:get_cmip6_context_from_filename(file) for file in cmip6_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and evaluation_period[0] == int(v["tmin"][:4]) and evaluation_period[1] == int(v["tmax"][:4])
        }
        cordex_aoi_evaluation = {k:v
            for k,v in {file:get_cordex_context_from_filename(file) for file in cordex_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and evaluation_period[0] == int(v["tmin"][:4]) and evaluation_period[1] == int(v["tmax"][:4])
        }

        # Get projections data
        cmip6_aoi_projection = {k:v
            for k,v in {file:get_cmip6_context_from_filename(file) for file in cmip6_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and projection_period[0] == int(v["tmin"][:4]) and projection_period[1] == int(v["tmax"][:4])
        }
        cordex_aoi_projection = {k:v
            for k,v in {file:get_cordex_context_from_filename(file) for file in cordex_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and projection_period[0] == int(v["tmin"][:4]) and projection_period[1] == int(v["tmax"][:4])
        }

        for k,v in {**cmip6_aoi_evaluation, **cordex_aoi_evaluation}.items():
            # Open evaluation and projection datasets
            ds_evaluation = xr.open_dataset(k)
            if v["data_product"] == DataProduct.CMIP6.product_name:
                ds_projection_file = [f for f,d in cmip6_aoi_projection.items()
                                      if d["institute"] == v["institute"] and d["source"] == v["source"] and d["ensemble"] == v["ensemble"]
                                    ]
            else:
                ds_projection_file = [f for f,d in cordex_aoi_projection.items()
                                      if d["domain"] == v["domain"] and
                                      d["driving_model"] == v["driving_model"] and
                                      d["rcm_name"] == v["rcm_name"] and
                                      d["ensemble"] == v["ensemble"] and
                                      d["rcm_version"] == v["rcm_version"]
                                    ]
            ds_projection = xr.open_dataset(ds_projection_file[0])

            # Interpolate the data onto reference grid
            regridder = xe.Regridder(ds_evaluation, reference_grid, "bilinear")
            ds_evaluation_reggrided = regridder(ds_evaluation, keep_attrs=True)
            ds_projection_reggrided = regridder(ds_projection, keep_attrs=True)

            # Downscale
            if method == DownscaleMethod.BIAS_CORRECTION:
                ds_projection_downscaled = bias_correction(ds_baseline_reggrided, ds_evaluation_reggrided, ds_projection_reggrided)
            else:
                msg = "Method not implemented yet, only bias_correction is available."
                raise ValueError(msg)

            # prep and write
            ds_projection_downscaled.to_netcdf(
                f"{output_dir}/{Path(k).stem}_downscaled.nc"
            )

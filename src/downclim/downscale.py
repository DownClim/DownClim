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
from .dataset.utils import Aggregation, DataProduct, climatology_filename, get_grid


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
    historical: xr.Dataset,
    projection: xr.Dataset
) -> xr.Dataset:
    """Bias correction of the projections.

    Args:
        baseline (xr.Dataset): Baseline product dataset.
        historical (xr.Dataset): Historical period dataset.
        projection (xr.Dataset): Projection period dataset.

    Returns:
        xr.Dataset: Bias corrected projections.
    """

    # Compute anomalies between projection and historical
    anomalies = projection - historical
    # Add anomalies to the baseline
    projection = baseline + anomalies

    if "pr" in projection:
        rel_anomalies_pr = anomalies["pr"] / (historical["pr"] + 1)
        projection["pr"] = baseline["pr"] * (1 + rel_anomalies_pr)

    return projection


def run_downscaling(
    aoi: list[gpd.GeoDataFrame],
    baseline_period: tuple[int, int],
    projection_period: tuple[int, int],
    baseline_product: DataProduct,
    cmip6_simulations_to_downscale: list[str] | None = None,
    cordex_simulations_to_downscale: list[str] | None = None,
    reference_grid_file: str | None = None,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    method: DownscaleMethod = DownscaleMethod.BIAS_CORRECTION,
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Run the downscaling process.

    Args:
        aoi (list[gpd.GeoDataFrame]): List of areas of interest.
        baseline_period (tuple[int, int]): Baseline period (start, end).
        projection_period (tuple[int, int]): Projection period (start, end).
        baseline_product (DataProduct): Baseline product to use.
        cmip6_simulations_to_downscale (list[str] | None): List of CMIP6 simulations to downscale. Defaults to None,
        which means all available CMIP6 simulations in "<input_dir>".
        cordex_simulations_to_downscale (list[str] | None): List of CORDEX simulations to downscale. Defaults to None,
        which means all available CORDEX simulations in "<input_dir>".
        reference_grid_file (str | None, optional): Path to the reference grid file. Defaults to None.
        aggregation (Aggregation, optional): Aggregation method to use. Defaults to Aggregation.MONTHLY_MEAN.
        method (DownscaleMethod, optional): Downscaling method to use. Defaults to DownscaleMethod.BIAS_CORRECTION.
        input_dir (str, optional): Input directory for the data. Defaults to "./results".
        output_dir (str | None, optional): Output directory for the results. Defaults to None.

    Raises:
        FileNotFoundError: If a required file is not found.
        ValueError: If a required parameter is invalid.
    """

    # Check input directory
    if input_dir is None:
        msg = "Input directory not provided. Using default input directory './results'."
        warn(msg, stacklevel=1)
        input_dir = "./results/"
    if not Path(input_dir).is_dir():
        msg = f"Input directory {input_dir} not found."
        raise FileNotFoundError(msg)

    # Create output directory
    if output_dir is None:
        output_dir = "./results/downscaled"
        msg  = f"Output directory not provided. Using default output directory {output_dir}."
        warn(msg, stacklevel=1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cmip6").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cordex").mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, _ = get_aoi_informations(aoi)

    for aoi_n in aoi_name:
        # Check and populate CMIP6 simulations if needed
        if cmip6_simulations_to_downscale is None:
            cmip6_simulations_to_downscale = [str(p) for p in Path(f"{input_dir}/cmip6").glob(f"{aoi_n}_cmip6*.nc")]
            msg = f"CMIP6 simulations to downscale not provided. Using all files found in {input_dir}/cmip6."
            warn(msg, stacklevel=1)
        if cmip6_simulations_to_downscale == []:
            msg = "No CMIP6 simulations to downscale found."
            warn(msg, stacklevel=1)

        # Check and populate CORDEX simulations if needed
        if cordex_simulations_to_downscale is None:
            cordex_simulations_to_downscale = [str(p) for p in Path(f"{input_dir}/cordex").glob(f"{aoi_n}_cordex*.nc")]
            msg = f"CORDEX simulations to downscale not provided. Using all files found in {input_dir}/cordex."
            warn(msg, stacklevel=1)
        if cordex_simulations_to_downscale == []:
            msg = "No CORDEX simulations to downscale found."
            warn(msg, stacklevel=1)

        # Get the reference grid
        if reference_grid_file is None:
            reference_grid_file = f"{input_dir}/{baseline_product.product_name}/{baseline_product.product_name}_{aoi_n}_grid.nc"
            msg = f"Reference grid file not provided. Using default grid file {reference_grid_file} which is extracted from {baseline_product.product_name}."
            warn(msg, stacklevel=1)
        if not Path(reference_grid_file).is_file():
            msg = f"Reference grid file {reference_grid_file} not found. Please provide a valid reference grid file."
            raise FileNotFoundError(msg)
        reference_grid = xr.open_dataset(reference_grid_file)

        # Get baseline historical data and interpolate on reference grid (if needed)
        baseline_file = climatology_filename(f"{input_dir}/{baseline_product.product_name}", aoi_n, baseline_product, aggregation, baseline_period)
        if not Path(baseline_file).is_file():
            msg = f"Baseline historical data not found for {aoi_n}. Please download it first."
            raise FileNotFoundError(msg)
        ds_baseline = xr.open_dataset(baseline_file)
        baseline_grid = get_grid(ds_baseline, baseline_product)
        if baseline_grid.equals(reference_grid):
            ds_baseline_reggrided = ds_baseline
            ds_baseline_reggrided = ds_baseline_reggrided.rename({baseline_product.lon_lat_names['lon']:'lon', baseline_product.lon_lat_names['lat']:'lat'})
        else:
            regridder = xe.Regridder(ds_baseline, reference_grid, "bilinear")
            ds_baseline_reggrided = regridder(ds_baseline, keep_attrs=True)

        # Get historical data
        cmip6_aoi_historical = {k:v
            for k,v in {file:get_cmip6_context_from_filename(file) for file in cmip6_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and baseline_period[0] == int(v["tmin"][:4]) and baseline_period[1] == int(v["tmax"][:4])
        }
        cordex_aoi_historical = {k:v
            for k,v in {file:get_cordex_context_from_filename(file) for file in cordex_simulations_to_downscale}.items()
            if aoi_n == v["aoi_n"] and baseline_period[0] == int(v["tmin"][:4]) and baseline_period[1] == int(v["tmax"][:4])
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

        for k,v in {**cmip6_aoi_historical, **cordex_aoi_historical}.items():
            # Open historical datasets
            ds_historical = xr.open_dataset(k)
            if v["data_product"] == DataProduct.CMIP6.product_name:
                ds_projection_file = [f for f,d in cmip6_aoi_projection.items()
                                      if d["institute"] == v["institute"] and
                                      d["source"] == v["source"] and
                                      d["ensemble"] == v["ensemble"]
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
            regridder = xe.Regridder(ds_historical, reference_grid, "bilinear")
            ds_historical_reggrided = regridder(ds_historical, keep_attrs=True)
            ds_projection_reggrided = regridder(ds_projection, keep_attrs=True)

            # Downscale
            if method == DownscaleMethod.BIAS_CORRECTION:
                ds_projection_downscaled = bias_correction(ds_baseline_reggrided, ds_historical_reggrided, ds_projection_reggrided)
            else:
                msg = "Method not implemented yet, only bias_correction is available."
                raise ValueError(msg)

            # prep and write
            ds_projection_downscaled.to_netcdf(
                f"{output_dir}/{Path(k).stem}_downscaled.nc"
            )

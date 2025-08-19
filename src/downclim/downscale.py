from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import geopandas as gpd
import xarray as xr
import xesmf as xe

from .aoi import get_aoi_informations
from .dataset.cmip6 import get_cmip6_context_from_filename
from .dataset.cordex import get_cordex_context_from_filename
from .dataset.utils import Aggregation, DataProduct, climatology_filename, get_regridder

logger = logging.getLogger(__name__)

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

def get_simulations_to_downscale(
    aoi_n:str,
    historical_period: tuple[int, int],
    evaluation_period: tuple[int, int],
    projection_period: tuple[int, int],
    cmip6_simulations_to_downscale: list[str],
    cordex_simulations_to_downscale: list[str]
    ) -> dict[str, dict[str, dict[str, dict[str, str]]]]:

    simulations_to_downscale: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    simulations_to_downscale["cmip6"] = {}
    simulations_to_downscale["cordex"] = {}
    simulations_to_downscale["cmip6"]["historical"], simulations_to_downscale["cmip6"]["evaluation"], simulations_to_downscale["cmip6"]["projection"] = _get_simulations_per_period(
        aoi_n, historical_period, evaluation_period, projection_period, cmip6_simulations_to_downscale, DataProduct.CMIP6
        )
    simulations_to_downscale["cordex"]["historical"], simulations_to_downscale["cordex"]["evaluation"], simulations_to_downscale["cordex"]["projection"] = _get_simulations_per_period(
        aoi_n, historical_period, evaluation_period, projection_period, cordex_simulations_to_downscale, DataProduct.CORDEX
        )
    logger.info("CMIP6 simulations to downscale: %s", simulations_to_downscale["cmip6"])
    logger.info("CORDEX simulations to downscale: %s", simulations_to_downscale["cordex"])

    return simulations_to_downscale

def _get_simulations_per_period(
    aoi_n: str,
    historical_period: tuple[int, int],
    evaluation_period: tuple[int, int],
    projection_period: tuple[int, int],
    simulations_list: list[str],
    dataproduct: DataProduct,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """ Given a list of CMIP6 or CORDEX already downloaded for historical, evaluation and projection periods, will return the list of simulations per period with their corresponding context.
    """
    historical = {}
    evaluation = {}
    projection = {}

    context_getter = get_cmip6_context_from_filename if dataproduct == DataProduct.CMIP6 else get_cordex_context_from_filename

    for k,v in {file:context_getter(file) for file in simulations_list}.items():
        tmin = int(v["tmin"][:4])
        tmax = int(v["tmax"][:4])
        if aoi_n == v["aoi_n"]:
            if historical_period[0] == tmin and historical_period[1] == tmax:
                historical[k] = v
            elif evaluation_period[0] == tmin and evaluation_period[1] == tmax:
                evaluation[k] = v
            elif projection_period[0] == tmin and projection_period[1] == tmax:
                projection[k] = v
            else:
                msg = f"Unknown period for {k}: {v['tmin']} - {v['tmax']}. Does not corresponds to historical, evaluation or projection period."
                logger.warning(msg)
                continue
    return (historical, evaluation, projection)

def _check_populate_simulations(
    simulations: list[str] | None,
    aoi_n: str,
    input_dir: str,
    dataproduct: DataProduct
) -> list[str]:
    """Check and populate simulations for a specific AOI and data product.

    Args:
        simulations (list[str] | None): List of simulations to downscale.
        aoi_n (str): AOI name.
        input_dir (str): Input directory where the simulation files are located.
        dataproduct (DataProduct): Data product.

    Returns:
        list[str]: List of populated simulations for the AOI and data product.
    """
    if simulations is None:
        simulations = [str(p) for p in Path(f"{input_dir}/{dataproduct.product_name}").glob(f"{aoi_n}_{dataproduct.product_name}*.nc")]
        msg = f"{dataproduct.product_name.upper()} simulations to downscale not provided. Using all files found in {input_dir}/{dataproduct.product_name}."
        logger.warning(msg)
    if simulations == []:
        msg = f"No {dataproduct.product_name.upper()} simulations to downscale found."
        logger.warning(msg)
    return simulations


def _matching_files(
    v: dict[str, str],
    period: str,
    simulations_to_downscale: dict[str, dict[str, dict[str, dict[str, str]]]]
    ) -> xr.Dataset:
    """Find matching files for a given simulation context and returns the associated dataset.

    Args:
        v (dict[str, str]): Simulation context.
        period (str): Period to match (historical, evaluation, projection).
        simulations_to_downscale (dict[str, dict[str, dict[str, str]]]): Simulations to downscale.

    Returns:
        xr.Dataset: Dataset of matching file.
    """
    # Define keys for CMIP6 and CORDEX to identify simulations
    simulations_keys = {
        "cmip6" : ["institute", "source", "ensemble"],
        "cordex" : ["domain", "driving_model", "rcm_name", "ensemble", "rcm_version"]
    }
    ds_files = [f for f,d in simulations_to_downscale[v["data_product"]][period].items()
        if all(d[key] == v[key] for key in simulations_keys[v["data_product"]])]
    if len(ds_files) == 0:
        msg = f"No matching files found for {v} in {period}"
        raise FileNotFoundError(msg)
    if len(ds_files) > 1:
        msg = f"Multiple conflicting matching files found for {v} in {period}: {ds_files}"
        raise FileExistsError(msg)
    return xr.open_dataset(ds_files[0])


def run_downscaling(
    aoi: list[gpd.GeoDataFrame],
    historical_period: tuple[int, int],
    evaluation_period: tuple[int, int],
    projection_period: tuple[int, int],
    baseline_product: DataProduct,
    cmip6_simulations_to_downscale: list[str] | None = None,
    cordex_simulations_to_downscale: list[str] | None = None,
    downscaling_grid_file: str | None = None,
    periods_to_downscale: list[str] | None = None,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN, # type: ignore[assignment]
    method: DownscaleMethod = DownscaleMethod.BIAS_CORRECTION,
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Run the downscaling process.

    Args:
        aoi (list[gpd.GeoDataFrame]): List of areas of interest.
        historical_period (tuple[int, int]): Baseline period (start, end).
        evaluation_period (tuple[int, int]): Evaluation period (start, end).
        projection_period (tuple[int, int]): Projection period (start, end).
        baseline_product (DataProduct): Baseline product to use.
        cmip6_simulations_to_downscale (list[str] | None): List of CMIP6 simulations to downscale. Defaults to None,
        which means all available CMIP6 simulations in "<input_dir>".
        cordex_simulations_to_downscale (list[str] | None): List of CORDEX simulations to downscale. Defaults to None,
        which means all available CORDEX simulations in "<input_dir>".
        downscaling_grid_file (str | None, optional): Path to the grid file on which to downscale. Defaults to None, meaning
        the grid will be extracted from the baseline product.
        periods_to_downscale (list[str] | None): List of periods to downscale. Can be any combination of ['evaluation', 'projection']. Defaults to None, meaning all periods will be downscaled.
        aggregation (Aggregation, optional): Aggregation method to use. Defaults to Aggregation.MONTHLY_MEAN.
        method (DownscaleMethod, optional): Downscaling method to use. Defaults to DownscaleMethod.BIAS_CORRECTION.
        input_dir (str, optional): Input directory for the data. Only used if "<cmip6_simulations_to_downscale>" or "<cordex_simulations_to_downscale>" are None. Defaults to "./results".
        output_dir (str, optional): Output directory for the results. Defaults to "./results/downscaled".

    Raises:
        FileNotFoundError: If a required file is not found.
        ValueError: If a required parameter is invalid.
    """

    # Check input directory
    if input_dir is None:
        msg = "Input directory not provided. Using default input directory './results'."
        logger.warning(msg)
        input_dir = "./results"
    if not Path(input_dir).is_dir():
        msg = f"Input directory {input_dir} not found."
        raise FileNotFoundError(msg)

    # Create output directory
    if output_dir is None:
        output_dir = "./results/downscaled"
        msg  = f"Output directory not provided. Using default output directory {output_dir}."
        logger.warning(msg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cmip6").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cordex").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/../regridder").mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, _ = get_aoi_informations(aoi)

    # Define periods to downscale
    if periods_to_downscale is None:
        periods_to_downscale = ['evaluation', 'projection']
        msg = f"Periods to downscale not provided. Using default periods {periods_to_downscale}."
        logger.warning(msg)
    if not all(period in ['evaluation', 'projection'] for period in periods_to_downscale):
        msg = f"Invalid periods found in {periods_to_downscale}. Please provide valid periods : ['evaluation', 'projection']."
        raise ValueError(msg)

    for aoi_n in aoi_name:
        # Check and populate simulations to downscale
        if not cmip6_simulations_to_downscale:
            cmip6_simulations_to_downscale = _check_populate_simulations(cmip6_simulations_to_downscale, aoi_n, input_dir, DataProduct.CMIP6)
        if not cordex_simulations_to_downscale:
            cordex_simulations_to_downscale = _check_populate_simulations(cordex_simulations_to_downscale, aoi_n, input_dir, DataProduct.CORDEX)

        # Get the downscaling grid
        if downscaling_grid_file is None:
            downscaling_grid_file = f"{input_dir}/{baseline_product.product_name}/{baseline_product.product_name}_{aoi_n}_grid.nc"
            msg = f"Downscaling grid file not provided. Using default grid file {downscaling_grid_file} which is extracted from {baseline_product.product_name}"
            logger.warning(msg)
        if not Path(downscaling_grid_file).is_file():
            msg = f"Downscaling grid file {downscaling_grid_file} not found. Please provide a valid downscaling grid file."
            logger.error(msg)
            raise FileNotFoundError(msg)
        downscaling_grid = xr.open_dataset(downscaling_grid_file)

        # Get baseline historical data and interpolate on downscaling grid (if needed)
        baseline_file = climatology_filename(f"{input_dir}/{baseline_product.product_name}", aoi_n, baseline_product, aggregation, historical_period)
        if not Path(baseline_file).is_file():
            msg = f"""Baseline historical data not found: {baseline_product.product_name} for {aoi_n} should be located in {baseline_file}.
            Please download it first by using the `downclim.downclim.DownClimContext.download_data` method."""
            logger.error(msg)
            raise FileNotFoundError(msg)
        ds_baseline = xr.open_dataset(baseline_file)
        baseline_grid_file = f"{input_dir}/{baseline_product.product_name}/{baseline_product.product_name}_{aoi_n}_grid.nc"
        baseline_grid = xr.open_dataset(baseline_grid_file)
        if baseline_grid.equals(downscaling_grid):
            ds_baseline_downscaling_grid = ds_baseline.rename(
                    {baseline_product.lon_lat_names['lon']:'lon', baseline_product.lon_lat_names['lat']:'lat'}
                )
        else:
            logger.info("Regridding baseline data %s, period %s, for AOI: %s.",
                baseline_product.product_name, historical_period, aoi_n)
            regridder = get_regridder(
                ds_baseline,
                downscaling_grid,
                baseline_grid_file,
                downscaling_grid_file,
                f"{output_dir}/..")
            ds_baseline_downscaling_grid = regridder(ds_baseline, keep_attrs=True)

        # Get historical / evaluation / projection data sets
        simulations_to_downscale = get_simulations_to_downscale(
            aoi_n,
            historical_period,
            evaluation_period,
            projection_period,
            cmip6_simulations_to_downscale,
            cordex_simulations_to_downscale
            )

        for k,v in {**simulations_to_downscale["cmip6"]["historical"], **simulations_to_downscale["cordex"]["historical"]}.items():
            # Open historical datasets and interpolate the data onto downscaling grid
            logger.info("Regridding historical data %s, period %s, for AOI: %s.", k, historical_period, aoi_n)
            ds_historical = xr.open_dataset(k)
            regridder = xe.Regridder(ds_historical, downscaling_grid, "bilinear")
            historical_regridded_file = f"{output_dir}/{v['data_product']}/{Path(k).stem}-{Path(downscaling_grid_file).stem}.nc"
            if not Path(historical_regridded_file).is_file():
                logger.warning("Regridded historical dataset for %s already exists: %s. No action taken.", k, historical_regridded_file)
            else:
                logger.info("Regridding historical dataset for %s: %s.", k, historical_regridded_file)
                ds_historical_regridded = regridder(ds_historical, keep_attrs=True)
                ds_historical_regridded.to_netcdf(historical_regridded_file)

            for period in periods_to_downscale:
                logger.info("Regridding dataset for %s, period %s, for AOI: %s.", k, period, aoi_n)
                ds_to_downscale = _matching_files(v, period, simulations_to_downscale)
                ds_to_downscale_regridded = regridder(ds_to_downscale, keep_attrs=True)

                # Downscale
                logger.info("Downscaling dataset %s, period %s, for AOI: %s using method: %s.", k, period, aoi_n, method.value)
                if method == DownscaleMethod.BIAS_CORRECTION:
                    ds_to_downscale_downscaled = bias_correction(ds_baseline_downscaling_grid, ds_historical_regridded, ds_to_downscale_regridded)
                else:
                    msg = "Method not implemented yet, only bias_correction is available."
                    raise ValueError(msg)

                # prep and write

                ds_to_downscale_downscaled.to_netcdf(
                    f"{output_dir}/{v['data_product']}/{Path(k).stem}-downscaled-{baseline_product.product_name}_baseline-{Path(downscaling_grid_file).stem}.nc"
                )

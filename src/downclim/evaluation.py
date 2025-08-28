from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
import xesmf as xe

from .dataset.cmip6 import get_cmip6_context_from_filename
from .dataset.cordex import get_cordex_context_from_filename
from .dataset.utils import (
    Aggregation,
    DataProduct,
    check_input_dir,
    check_output_dir,
    climatology_filename,
)
from .logging_config import get_logger

logger = get_logger(__name__)


def _check_populate_simulations(
    simulations: list[str] | None,
    aoi_n: str,
    input_dir: str,
    dataproduct: DataProduct,
    period: tuple[int, int],
    evaluation_grid: str
) -> list[str]:
    """Check and populate simulations for a specific AOI, data product and period.

    Args:
        simulations (list[str] | None): List of simulations to evaluate.
        aoi_n (str): AOI name.
        input_dir (str): Input directory where the simulation files are located.
        dataproduct (DataProduct): Data product.
        period (tuple[int, int]): Period to consider for the simulations.
        evaluation_grid (str): Evaluation grid name.

    Returns:
        list[str]: List of populated simulations for the AOI and data product.
    """
    if simulations is None:
        simulations_ok = [
            str(p) \
            for p in Path(f"{input_dir}/{dataproduct.product_name}").glob(f"{aoi_n}_{dataproduct.product_name}*{period[0]}*{period[1]}*{evaluation_grid}.nc") \
            ]
        logger.warning("%s simulations to evaluate not provided. Using all files found in %s/%s.", dataproduct.product_name.upper(), input_dir, dataproduct.product_name)
    else:
        simulations_ok = [
            p for p in simulations
            if Path(p).is_file() and
                p.startswith(f"{input_dir}/{dataproduct.product_name}/{aoi_n}_{dataproduct.product_name}") and
                p.endswith(f"{period[0]}*{period[1]}*{evaluation_grid}.nc")
            ]
    if simulations_ok == []:
        logger.warning("No %s simulations to evaluate found.", dataproduct.product_name.upper())
        logger.warning("Files that are evaluated must have the format %s.", f"{input_dir}/{dataproduct.product_name}/{aoi_n}_{dataproduct.product_name}*{period[0]}*{period[1]}*{evaluation_grid}.nc")
    return simulations_ok


def compute_evaluation(ds_reference: xr.Dataset, ds_evaluated: xr.Dataset) -> xr.Dataset:
    """
    Computes evaluation metrics between reference and evaluated datasets.
    """
    return xr.concat([
        compute_correlation(ds_reference, ds_evaluated),
        compute_rmse(ds_reference, ds_evaluated),
        compute_std(ds_reference, ds_evaluated),
        compute_mean(ds_reference, ds_evaluated),
        ], pd.Index(["correlation", "rmse", "std", "mean"], name="indicator"))


def compute_correlation(ds_reference: xr.Dataset, ds_evaluated: xr.Dataset) -> xr.Dataset:
    """
    Computes correlation between reference and evaluated datasets.
    """
    logger.info("   Computing correlation...")
    correlations = {}

    for var in ds_reference.data_vars:
        if var in ds_evaluated.data_vars:
            # Calculate correlation for each month
            corr_by_month = xr.corr(
                ds_reference[var],
                ds_evaluated[var],
                dim=['lat', 'lon']  # Spatial correlation for each month
            )
            correlations[var] = corr_by_month

    return xr.Dataset(correlations)

def compute_rmse(ds_reference: xr.Dataset, ds_evaluated: xr.Dataset) -> xr.Dataset:
    """
    Computes RMSE between reference and evaluated datasets.
    """
    logger.info("   Computing RMSE...")
    rmse = {}

    for var in ds_reference.data_vars:
        if var in ds_evaluated.data_vars:
            # Calculate RMSE for each month
            rmse_by_month = ((ds_reference[var] - ds_evaluated[var]) ** 2).mean(dim=['lat', 'lon'])** 0.5
            rmse[var] = rmse_by_month

    return xr.Dataset(rmse)

def compute_std(ds_reference: xr.Dataset, ds_evaluated: xr.Dataset) -> xr.Dataset:
    """
    Computes standard deviation between reference and evaluated datasets.
    """
    logger.info("   Computing standard deviation...")
    std = {}

    for var in ds_reference.data_vars:
        if var in ds_evaluated.data_vars:
            # Calculate standard deviation for each month
            diff = ds_evaluated[var] - ds_reference[var]
            std_by_month = diff.std(dim=['lat', 'lon'])  # Méthode de l'objet, pas du module
            std[var] = std_by_month

    return xr.Dataset(std)

def compute_mean(ds_reference: xr.Dataset, ds_evaluated: xr.Dataset) -> xr.Dataset:
    """
    Computes mean between reference and evaluated datasets.
    """
    logger.info("   Computing mean...")
    mean = {}

    for var in ds_reference.data_vars:
        if var in ds_evaluated.data_vars:
            # Calculate mean for each month
            diff = ds_evaluated[var] - ds_reference[var]
            mean_by_month = diff.mean(dim=['lat', 'lon'])  # Méthode de l'objet, pas du module
            mean[var] = mean_by_month

    return xr.Dataset(mean)

def run_evaluation(
    aoi: list[gpd.GeoDataFrame],
    evaluation_period: tuple[int, int],
    evaluation_product: list[DataProduct],
    cmip6_simulations_to_evaluate: list[str] | None = None,
    cordex_simulations_to_evaluate: list[str] | None = None,
    evaluation_grid_file: list[str] | None = None,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN, # type: ignore[assignment]
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> xr.Dataset:
    """Run the evaluation of the simulations (CMIP6 and / or CORDEX) on the evaluation period.

    Parameters
    ----------
        aoi: list[gpd.GeoDataFrame]
            List of areas of interest (AOI). Obtained from `get_aoi` function
        evaluation_period: tuple[int, int]
            Evaluation period as a tuple of (start_year, end_year)
        evaluation_product: list[DataProduct]
            List of evaluation products to use, e.g. [DataProduct.CHELSA, DataProduct.CHIRPS]
        cmip6_simulations_to_evaluate: list[str] | None
            List of CMIP6 simulations to evaluate.
            If None, all available simulations located in `<input_dir>/cmip6` will be used.
            Default is None.
        cordex_simulations_to_evaluate: list[str] | None
            List of CORDEX simulations to evaluate.
            If None, all available simulations located in `<input_dir>/cordex` will be used.
            Default is None.
        evaluation_grid_file: list[str] | None
            List of evaluation grid files to use. Has to be the same length as `evaluation_product`.
            If None, grid from the evaluation product will be used.
            Default is None.
        aggregation: Aggregation
            Aggregation method to use. Default is Aggregation.MONTHLY_MEAN.
        input_dir: str | None
            Input directory for the evaluation files. If None, taken as `results/downscaled`.
            Default is None.
        output_dir: str | None
            Output directory for the evaluation results. If None, taken as `results/evaluation`.
            Default is None.

    Returns
    -------
        xr.Dataset: Dataset with the evaluation.
    """

    logger.info("Starting evaluation process...")
    # Check input directory
    input_dir = check_input_dir(input_dir, "./results/downscaled")
    # Create output directory
    output_dir = check_output_dir(output_dir, "./results/evaluation", ["cmip6", "cordex"])

    # Check evaluation grids
    if (evaluation_grid_file) and (len(evaluation_grid_file) != len(evaluation_product)):
        msg = f"Number of evaluation grid files provided ({len(evaluation_grid_file)}) do not match the number of evaluation products ({len(evaluation_product)})."
        logger.error(msg)
        raise ValueError(msg)

    for aoi_i in aoi:
        # Check and populate simulations to evaluate if needed
        aoi_n = aoi_i.NAME_0[0]
        aoi_g = aoi_i.geometry
        logger.info("   Start evaluation for AOI: %s", aoi_n)

        for i, product in enumerate(evaluation_product):
            logger.info("   Evaluating simulations over product: %s", product.product_name)

            # Get the evaluation grid
            if evaluation_grid_file is None:
                grid_file = f"{input_dir}/../{product.product_name}/{product.product_name}_{aoi_n}_grid.nc"
                logger.warning("    Evaluation grid file not provided. Using default grid file %s which is extracted from %s.", grid_file, product.product_name)
            else:
                grid_file = evaluation_grid_file[i]
            if not Path(grid_file).is_file():
                msg = f"Evaluation grid file {grid_file} not found. Please provide a valid evaluation grid file."
                logger.error(msg)
                raise FileNotFoundError(msg)
            logger.info("    Opening evaluation grid file: %s", grid_file)
            evaluation_grid = xr.open_dataset(grid_file)

            logger.info("    Opening evaluation product %s...", product.product_name)
            product_file = climatology_filename(f"{input_dir}/../{product.product_name}", aoi_n, product, aggregation, evaluation_period)
            logger.info("    Looking for evaluation product for evaluation period here : %s", product_file)
            if not Path(product_file).is_file():
                msg = f"""{product.product_name} file not found for {aoi_n} and period {evaluation_period}.
                It was expected here: {product_file}
                Please download it first."""
                logger.error(msg)
                raise FileNotFoundError(msg)
            ds_product = xr.open_dataset(product_file)

            # Check if evaluation product needs to be regridded onto evaluation grid
            product_grid_file = f"{input_dir}/../{product.product_name}/{product.product_name}_{aoi_n}_grid.nc"
            product_grid = xr.open_dataset(product_grid_file)
            if product_grid.equals(evaluation_grid):
                logger.info("       Evaluation grid and %s grid are the same. No need to regrid.", product.product_name)
                ds_product_reggrided = ds_product.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.clip(aoi_g.values)
            else:
                logger.info("       Regridding evaluation data onto the evaluation grid.")
                regridder = xe.Regridder(ds_product, evaluation_grid, "bilinear")
                ds_product_reggrided = regridder(ds_product, keep_attrs=True)
                ds_product_reggrided = ds_product_reggrided.rio.clip(aoi_g)

            logger.info("       Looking for downscaled simulations to evaluate...")
            if not cmip6_simulations_to_evaluate:
                cmip6_simulations = _check_populate_simulations(cmip6_simulations_to_evaluate, aoi_n, input_dir, DataProduct.CMIP6, evaluation_period, Path(grid_file).stem)
            if not cordex_simulations_to_evaluate:
                cordex_simulations = _check_populate_simulations(cordex_simulations_to_evaluate, aoi_n, input_dir, DataProduct.CORDEX, evaluation_period, Path(grid_file).stem)

            cmip6_aoi_evaluation = {file:get_cmip6_context_from_filename(file) for file in cmip6_simulations}
            cordex_aoi_evaluation = {file:get_cordex_context_from_filename(file) for file in cordex_simulations}

            for k, v in {**cmip6_aoi_evaluation, **cordex_aoi_evaluation}.items():
                # Get the dataset
                ds_evaluated = xr.open_dataset(k).rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.clip(aoi_g)

                logger.info("""
                            Evaluating %s against %s...
                            """,  k, product_file)
                ds_evaluation = compute_evaluation(ds_product_reggrided, ds_evaluated)
                # Save the evaluation metrics
                eval_file = f"{output_dir}/{v['data_product']}/{Path(k).stem}_evaluation.nc"
                ds_evaluation.to_netcdf(eval_file)

    return xr.Dataset()

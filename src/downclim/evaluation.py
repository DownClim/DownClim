from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from .aoi import get_aoi_informations
from .dataset.cmip6 import get_cmip6_context_from_filename
from .dataset.cordex import get_cordex_context_from_filename
from .dataset.utils import Aggregation, DataProduct, climatology_filename, get_grid

logger = logging.getLogger(__name__)

def get_hist(
    pred_ds: xr.Dataset,
    type: str,
    aoi: str,
    origin: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    baseline: str,
    aggregation: str,
    period_proj: str,
    period_eval: str,
    ds_method: str,
) -> pd.DataFrame:
    variables = list(pred_ds.keys())
    months = list(range(1, 13))
    a = []
    for v in variables:
        for m in months:
            if v == "pr":
                low = 0
                high = 2000
                step = 10
            else:
                low = 0
                high = 1000
                step = 1
            bins = np.arange(low, high, step)
            labels = np.arange(low + step / 2, high - step / 2, step)
            out = pd.cut(
                pred_ds.sel(month=m)[v].to_numpy().ravel(), bins=bins, labels=labels
            )
            res = out.value_counts().to_frame()
            res["bin"] = res.index
            res.insert(0, "month", m)
            res.insert(0, "variable", v)
            a.append(res)
    tab = pd.concat(a)
    tab.insert(0, "aoi", aoi)
    tab.insert(0, "origin", origin)
    tab.insert(0, "type", type)
    tab.insert(0, "domain", domain)
    tab.insert(0, "institute", institute)
    tab.insert(0, "model", model)
    tab.insert(0, "experiment", experiment)
    tab.insert(0, "ensemble", ensemble)
    tab.insert(0, "rcm", rcm)
    tab.insert(0, "downscaling", downscaling)
    tab.insert(0, "base", baseline)
    tab.insert(0, "aggregation", aggregation)
    tab.insert(0, "period_proj", period_proj)
    tab.insert(0, "period_eval", period_eval)
    tab.insert(0, "ds_method", ds_method)
    tab[
        [
            "area",
            "origin",
            "type",
            "domain",
            "institute",
            "model",
            "experiment",
            "ensemble",
            "rcm",
            "downscaling",
            "base",
            "aggregation",
            "period_proj",
            "period_eval",
            "ds_method",
            "month",
            "variable",
            "bin",
            "count",
        ]
    ]
    return tab


def get_eval(
    downscaled_ds: xr.Dataset,
    base_ds: xr.Dataset,
    type: str,
    aoi: str,
    origin: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    base: str,
    aggregation: str,
    period_proj: str,
    period_eval: str,
    ds_method: str,
    base_eval: str,
) -> pd.DataFrame:
    """
    Compute evaluation metrics (CC, RMSEP, SDE, Bias) for the projection before and after downscaling.

    Parameters
    ----------
    downscaled_ds: xr.Dataset
        Dataset with the downscaled data.
    base_ds: xr.Dataset
        Base dataset.
    type: str
        Type of data.

    Returns
    -------
    pd.DataFrame
        Dataframe.
    """

    variables = list(downscaled_ds.keys())
    months = list(range(1, 13))
    a = []
    for v in variables:
        for m in months:
            pred = downscaled_ds.sel(month=m)[v].to_numpy().ravel()
            pred = pred[~np.isnan(pred)]
            obs = base_ds.sel(month=m)[v].to_numpy().ravel()
            obs = obs[~np.isnan(obs)]
            d = {
                "metric": ["CC", "RMSE", "SDE", "bias"],
                "value": [
                    np.corrcoef(pred, obs)[1, 0],
                    np.sqrt(np.mean(pow(pred - obs, 2))),
                    np.std(pred - obs),
                    np.mean(pred - obs),
                ],
            }
            res = pd.DataFrame(data=d)
            res.insert(0, "month", m)
            res.insert(0, "variable", v)
            a.append(res)
    tab = pd.concat(a)
    tab.insert(0, "area", aoi)
    tab.insert(0, "origin", origin)
    tab.insert(0, "type", type)
    tab.insert(0, "domain", domain)
    tab.insert(0, "institute", institute)
    tab.insert(0, "model", model)
    tab.insert(0, "experiment", experiment)
    tab.insert(0, "ensemble", ensemble)
    tab.insert(0, "rcm", rcm)
    tab.insert(0, "downscaling", downscaling)
    tab.insert(0, "base", base)
    tab.insert(0, "aggregation", aggregation)
    tab.insert(0, "period_proj", period_proj)
    tab.insert(0, "period_eval", period_eval)
    tab.insert(0, "ds_method", ds_method)
    tab.insert(0, "base_eval", base_eval)
    tab[
        [
            "area",
            "origin",
            "type",
            "domain",
            "institute",
            "model",
            "experiment",
            "ensemble",
            "rcm",
            "downscaling",
            "base",
            "aggregation",
            "period_proj",
            "period_eval",
            "ds_method",
            "base_eval",
            "month",
            "variable",
            "metric",
            "value",
        ]
    ]
    return tab


def run_hist(
    aoi: str,
    ds_file: str,
    aoi_file: str,
    output_file: str,
    origin: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    baseline: str,
    aggregation: str,
    period_eval: str,
) -> pd.DataFrame:
    proj_file = f"results/projections/{aoi}_{origin}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period_eval}.nc"
    area_shp = gpd.read_file(aoi_file)
    ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    proj = xr.open_dataset(proj_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    pd.concat([get_hist(ds, "downscaled"), get_hist(proj, "raw")]).to_csv(
        output_file, sep="\t", index=False
    )

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


def run_evaluation(
    aoi: list[gpd.GeoDataFrame],
    historical_period: tuple[int, int],
    evaluation_period: tuple[int, int],
    evaluation_product: list[DataProduct],
    cmip6_simulations_to_evaluate: list[str] | None = None,
    cordex_simulations_to_evaluate: list[str] | None = None,
    evaluation_grid_file: list[str] | None = None,
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN, # type: ignore[assignment]
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Run the evaluation of the simulations (CMIP6 and / or CORDEX) on the evaluation period.

    Parameters
    ----------

    Returns
    -------
        pd.DataFrame: Dataframe with the evaluation.
    """

    # Check input directory
    if input_dir is None:
        input_dir = "./results/downscaled"
        msg = f"Input directory not provided. Using default input directory {input_dir}."
        logger.warning(msg)
    if not Path(input_dir).is_dir():
        msg = f"Input directory {input_dir} not found."
        raise FileNotFoundError(msg)

    # Create output directory
    if output_dir is None:
        output_dir = "./results/evaluation"
        msg  = f"Output directory not provided. Using default output directory {output_dir}."
        logger.warning(msg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cmip6").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/cordex").mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aoi_name, _ = get_aoi_informations(aoi)

    for aoi_n in aoi_name:
        # Check and populate simulations to evaluate if needed
        if not cmip6_simulations_to_evaluate:
            cmip6_simulations_to_evaluate = _check_populate_simulations(cmip6_simulations_to_evaluate, aoi_n, input_dir, DataProduct.CMIP6)
        if not cordex_simulations_to_evaluate:
            cordex_simulations_to_evaluate = _check_populate_simulations(cordex_simulations_to_evaluate, aoi_n, input_dir, DataProduct.CORDEX)

        for i, product in enumerate(evaluation_product):
            # Get the evaluation grid
            if evaluation_grid_file is None:
                grid_file = f"{input_dir}/{product.product_name}/{product.product_name}_{aoi_n}_grid.nc"
                msg = f"Evaluation grid file not provided. Using default grid file {grid_file} which is extracted from {product.product_name}."
                logger.warning(msg)
            else:
                grid_file = evaluation_grid_file[i]
            if not Path(grid_file).is_file():
                msg = f"Evaluation grid file {grid_file} not found. Please provide a valid evaluation grid file."
                raise FileNotFoundError(msg)
            evaluation_grid = xr.open_dataset(grid_file)

            product_file = climatology_filename(f"{input_dir}/{product.product_name}", aoi_n, product, aggregation, evaluation_period)
            if not Path(product_file).is_file():
                msg = f"""{product.product_name} file not found for {aoi_n} and period {evaluation_period}.
                It was expected here: {product_file}
                Please download it first."""
                raise FileNotFoundError(msg)
            ds_product = xr.open_dataset(product_file)
            product_grid = get_grid(ds_product, product)
            if product_grid.equals(evaluation_grid):
                ds_product_reggrided = ds_product
                ds_product_reggrided = ds_product_reggrided.rename({product.lon_lat_names['lon']:'lon', product.lon_lat_names['lat']:'lat'})
            else:
                msg = f"Need to regrid from {product_grid} grid to {evaluation_grid} grid... this might take a while."
                regridder = xe.Regridder(ds_product, evaluation_grid, "bilinear")
                ds_product_reggrided = regridder(ds_product, keep_attrs=True)

        # Get data for the evaluation period
        cmip6_aoi_evaluation = {k:v
            for k,v in {file:get_cmip6_context_from_filename(file) for file in cmip6_simulations_to_evaluate}.items()
            if aoi_n == v["aoi_n"] and evaluation_period[0] == int(v["tmin"][:4]) and evaluation_period[1] == int(v["tmax"][:4])
        }
        cordex_aoi_evaluation = {k:v
            for k,v in {file:get_cordex_context_from_filename(file) for file in cordex_simulations_to_evaluate}.items()
            if aoi_n == v["aoi_n"] and evaluation_period[0] == int(v["tmin"][:4]) and evaluation_period[1] == int(v["tmax"][:4])
        }

        for k, v in {**cmip6_aoi_evaluation, **cordex_aoi_evaluation}.items():
            # Get the dataset
            ds = xr.open_dataset(k)
            regridder = xe.Regridder(ds, evaluation_grid, "bilinear")
            ds_reggrided_path = f"{output_dir}/{Path(k).stem}-{Path(grid_file).stem}.nc"
            if not Path(ds_reggrided_path).is_file():
                msg = f"Regridded historical dataset {k} already exists. No action taken."
                logger.warning(msg)
            else:
                ds_reggrided = regridder(ds, keep_attrs=True)
                ds_reggrided.to_netcdf(ds_reggrided_path)


            # Get the evaluation metrics
            if product.product_name == DataProduct.CMIP6.product_name:
                eval_file = f"{output_dir}/cmip6/{v['aoi_n']}_{v['origin']}_{v['domain']}_{v['institute']}_{v['model']}_{v['experiment']}_{v['ensemble']}_{v['rcm']}_{v['downscaling']}_{v['baseline']}_{aggregation.value}_{evaluation_period[0]}_{evaluation_period[1]}.nc"
            else:
                eval_file = f"{output_dir}/cordex/{v['aoi_n']}_{v['origin']}_{v['domain']}_{v['institute']}_{v['model']}_{v['experiment']}_{v['ensemble']}_{v['rcm']}_{v['downscaling']}_{v['baseline']}_{aggregation.value}_{evaluation_period[0]}_{evaluation_period[1]}.nc"

            if not Path(eval_file).is_file():
                msg = f"Evaluation file {eval_file} not found. Please run the evaluation first."
                raise FileNotFoundError(msg)

            eval_ds = xr.open_dataset(eval_file)
            eval_ds = eval_ds.rio.clip(aoi.geometry.values, aoi.crs)

            # Save the evaluation metrics
            eval_ds.to_netcdf(eval_file)




    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        ds = xr.open_dataset(downscaled_projection_file).rio.clip(aoi.geometry.values)



    proj_file = f"results/projections/{parameters.area}_{parameters.origin}_{parameters.domain}_{parameters.institute}_{parameters.model}_{parameters.experiment}_{parameters.ensemble}_{parameters.rcm}_{parameters.downscaling}_{parameters.baseline}_{parameters.aggregation}_{parameters.period_eval}.nc"


    proj = xr.open_dataset(proj_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    base = xr.open_dataset(base_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    out_file = pd.concat(
        [get_eval(ds, base, "downscaled"), get_eval(proj, base, "raw")]
    )
    if save_file:
        out_file.to_csv(out_file, sep="\t", index=False)
    return out_file


def merge_eval(eval_files: list, merged_file: str) -> pd.DataFrame:
    """Merge the evaluation files.

    Args:
        eval_files (list): List of evaluation files.
        merged_file (str): File to save the merged evaluation.

    Returns:
        pd.DataFrame: Merged evaluation.
    """

    merged = pd.concat([pd.read_csv(file, sep="\t") for file in eval_files])
    merged.to_csv(merged_file, sep="\t", index=False)
    return merged


def merge_hist(
    base_hist_files: list[Path], proj_hist_files: list[Path], output_file: Path
) -> None:
    """Merges all baseline and projection historical files.

    Args:
        base_hist_files (list[Path]): list of baseline historical files
        proj_hist_files (list[Path]): list of projection historical files
        output_file (Path): name of the output file to save the merged historical files.
    """
    base = [pd.read_csv(base_file, sep="\t") for base_file in base_hist_files]
    proj = [pd.read_csv(proj_file, sep="\t") for proj_file in proj_hist_files]

    all_hist_files = pd.concat([proj, base])
    all_hist_files = all_hist_files[all_hist_files["count"] > 0]
    all_hist_files.to_csv(output_file, sep="\t", index=False)


def merge_bias(
    aois: list[str],
    bias_files: list[Path],
    origins: list[str],
    base_eval: list[str],
) -> None:
    for origin in origins:
        files_o = [f for f in bias_files if re.search("_" + origin + "_", f)]
        for eval in base_eval:
            files_b = [f for f in files_o if re.search("_" + eval + ".nc", f)]
            for aoi in aois:
                files_a = [f for f in files_b if re.search("/" + aoi + "_", f)]
                ds_all = [xr.open_dataset(f).expand_dims(file=[f]) for f in files_a]
                ds = xr.combine_by_coords(ds_all).mean("file")
                path = f"results/evaluation/bias/{aoi}_{origin}_{eval}.nc"
                ds.to_netcdf(path)


def map_bias(
    ds_file: Path,
    base_file: Path,
    area_file: Path,
    out_file: Path,
    baseline: str,
    base_eval: str,
) -> None:
    area_shp = gpd.read_file(area_file)
    ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    if base_eval == baseline:
        base = xr.open_dataset(base_file).rio.clip(
            area_shp.geometry.values, area_shp.crs
        )
    if base_eval != baseline:
        base = xr.open_dataset(base_file, decode_coords="all")
        regridder = xe.Regridder(base, ds, "bilinear")
        base = regridder(base, keep_attrs=True)
    bias = ds - base
    bias.to_netcdf(out_file)

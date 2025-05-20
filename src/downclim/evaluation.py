from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from .aoi import get_aoi_informations


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
    area_shp = gp.read_file(aoi_file)
    ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    proj = xr.open_dataset(proj_file).rio.clip(area_shp.geometry.values, area_shp.crs)
    pd.concat([get_hist(ds, "downscaled"), get_hist(proj, "raw")]).to_csv(
        output_file, sep="\t", index=False
    )


def run_eval(
    aoi: gpd.GeoDataFrame,
    variables: list[str],
    input_directory: str,
    output_directory: str,
    downscaled_projection_file: str,
    baseline_file: str,
    parameters: dict(any, any),
    save_file: str | None = None,
) -> pd.DataFrame:
    """Run the evaluation of the projections.

    Args:
        downscaled_projection_file (str): File with the downscaled data.
        baseline_file (str): File with the baseline data.
        parameters (dict): Parameters of the simulation to evaluate.
        save_file (str): File to save the evaluation.

    Returns:
        pd.DataFrame: Dataframe with the evaluation.
    """
    # Get AOIs information
    aois_names, aois_bounds = get_aoi_informations(aoi)

    for aoi_n, aoi_b in zip(aois_names, aois_bounds, strict=False):
        ds = xr.open_dataset(downscaled_projection_file).rio.clip(aoi.geometry.values)



    proj_file = f"results/projections/{parameters.area}_{parameters.origin}_{parameters.domain}_{parameters.institute}_{parameters.model}_{parameters.parameters.experiment}_{parameters.ensemble}_{parameters.rcm}_{parameters.downscaling}_{parameters.baseline}_{parameters.aggregation}_{parameters.period_eval}.nc"


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
    area_shp = gp.read_file(area_file)
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

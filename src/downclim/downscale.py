from __future__ import annotations

from pathlib import Path

import xarray as xr


def generate_dataset_names(
    proj_file: str,
    area: str,
    project: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    baseline: str,
    aggregation: str,
    period: str,
) -> str:
    return f"{Path(proj_file).parent()}/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"


def open_datasets(
    base_hist_file: str,
    proj_file: str,
    area: str,
    project: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    baseline: str,
    aggregation: str,
    period_future: str,
    period_hist: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open the datasets.

    Returns:
        tuple: Datasets.
    """

    # open
    proj_future_file = generate_dataset_names(
        proj_file,
        area,
        project,
        domain,
        institute,
        model,
        experiment,
        ensemble,
        rcm,
        downscaling,
        baseline,
        aggregation,
        period_future,
    )
    proj_hist_file = generate_dataset_names(
        proj_file,
        area,
        project,
        domain,
        institute,
        model,
        experiment,
        ensemble,
        rcm,
        downscaling,
        baseline,
        aggregation,
        period_hist,
    )

    proj_future = xr.open_mfdataset(proj_future_file, parallel=True)
    proj_hist = xr.open_mfdataset(proj_hist_file, parallel=True)
    base_hist = xr.open_mfdataset(base_hist_file, parallel=True)

    return base_hist, proj_hist, proj_future


def bias_correction(
    base_hist: xr.Dataset, proj_hist: xr.Dataset, proj_future: xr.Dataset
) -> xr.Dataset:
    """Bias correction of the projections.

    Args:
        base_hist (xr.Dataset): Baseline historical data.
        proj_hist (xr.Dataset): Historical projections.
        proj_future (xr.Dataset): Future projections.

    Returns:
        xr.Dataset: Bias corrected projections.
    """

    # anomalies
    anomalies = proj_future - proj_hist
    # add to the baseline
    proj_ds = base_hist + anomalies
    if "pr" in list(proj_future.keys()):
        anomalies_rel = (proj_future - proj_hist) / proj_hist
        anomalies["pr"] = anomalies_rel["pr"]
        proj_ds = base_hist * (1 + anomalies)

    return proj_ds


def downscale(
    method: str,
    proj_file: str,
    base_hist_file: str,
    area: str,
    project: str,
    domain: str,
    institute: str,
    model: str,
    experiment: str,
    ensemble: str,
    rcm: str,
    downscaling: str,
    baseline: str,
    aggregation: str,
    period_future: str,
    period_hist: str,
    proj_hist_file: str,
    proj_future_file: str,
) -> None:
    base_hist, proj_hist, proj_future = open_datasets(
        base_hist_file, proj_hist_file, proj_future_file
    )

    if method == "bias_correction":
        proj_ds = bias_correction(base_hist, proj_hist, proj_future)
    else:
        msg = "Method not implemented yet, only bias_correction is available."
        raise ValueError(msg)

    # prep and write
    proj_ds.to_netcdf(
        f"{Path(proj_file).parent()}/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period_future}_{period_hist}_bc.nc"
    )

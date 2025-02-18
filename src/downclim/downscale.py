from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import xarray as xr


class DownscaleMethod(Enum):
    """Class to define the downscaling methods available."""

    BIAS_CORRECTION = "bias_correction"
    QUANTILE_MAPPING = "quantile_mapping"
    DYNAMICAL = "dynamical"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented downscaling method '{value}'. Right now only 'bias_correction' is implemented."
        raise ValueError(msg)


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
    """Generate the dataset names for the projections.

    Args:
        proj_file (str): Future projections dataset.
        area (str): Area name.
        project (str): Project name.
        domain (str): Domain name.
        institute (str): Institute name.
        model (str): Model name.
        experiment (str): Experiment name.
        ensemble (str): Ensemble name.
        rcm (str): RCM name.
        downscaling (str): Downscaling method.
        baseline (str): Baseline product.
        aggregation (str): Aggregation method.
        period (str): Period.

    Returns:
        str: Dataset name as a netCDF file.
    """
    return f"{Path(proj_file).parent}/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"


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
    """Open all the datasets for downscaling : baseline historical, historical projections and future projections.

    Args:
        base_hist_file (str): Baseline historical dataset filename.
        proj_file (str): Future projections dataset filename.
        area (str): Area name.
        project (str): Project name.
        domain (str): Domain name.
        institute (str): Institute name.
        model (str): Model name.
        experiment (str): Experiment name.
        ensemble (str): Ensemble name.
        rcm (str): RCM name.
        downscaling (str): Downscaling method.
        baseline (str): Baseline product.
        aggregation (str): Aggregation method.
        period_future (str): Future period.
        period_hist (str): Historical period

    Returns:
        tuple (xr.Dataset, xr.Dataset, xr.Dataset): baseline historical, historical projections and future projections datasets.
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

    return (
        xr.open_mfdataset(base_hist_file, parallel=True),
        xr.open_mfdataset(proj_hist_file, parallel=True),
        xr.open_mfdataset(proj_future_file, parallel=True),
    )


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

    # Compute anomalies
    anomalies = proj_future - proj_hist
    # Add to the baseline
    proj_ds = base_hist + anomalies
    if "pr" in proj_future:
        anomalies_rel = (proj_future - proj_hist) / (proj_hist + 1)
        anomalies["pr"] = anomalies_rel["pr"]
        proj_ds = base_hist * (1 + anomalies)

    return proj_ds


def downscale(
    method: DownscaleMethod,
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
) -> None:
    base_hist, proj_hist, proj_future = open_datasets(
        base_hist_file,
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
        period_hist,
    )

    if method == DownscaleMethod.BIAS_CORRECTION:
        proj_ds = bias_correction(base_hist, proj_hist, proj_future)
    else:
        msg = "Method not implemented yet, only bias_correction is available."
        raise ValueError(msg)

    reference_file = f"{period_reference_output}/{aoi_n}_{period_reference_product.product_name}_{aggregation.value}_{period[0]}-{period[1]}.nc"
    reference = xr.open_dataset(reference_file[0])
    # regridder = xe.Regridder(ds, reference, "bilinear")
    # ds_r = regridder(ds, keep_attrs=True)

    # prep and write
    proj_ds.to_netcdf(
        f"{Path(proj_file).parent}/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period_future}_{period_hist}_bc.nc"
    )

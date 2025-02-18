from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import multiprocess as mp
import pandas as pd
import xarray as xr
import xesmf as xe
from pydantic import BaseModel, Field, field_validator

from .aoi import get_aoi_informations
from .connectors import connect_to_esgf
from .utils import (
    Aggregation,
    DataProduct,
    Frequency,
    get_monthly_climatology,
    prep_dataset,
    split_period,
)


class CORDEXContext(BaseModel):
    """Context about the query on the CORDEX dataset. Entries of the dictionary can be either `str` or `Iterables` (e.g. `list`) if multiple values are provided. These following keys are available, and correspond to the keys defined defined on ESGF nodes, cf. https://esgf-node.ipsl.upmc.fr/search/cordex-ipsl/ ."""

    project: list[str] = Field(default="CORDEX", description="Name of the project")
    product: list[str] | None = Field(
        default="output",
        description="Name of the product. You probably want to keep the default value.",
    )
    domain: list[str] | None = Field(
        default=None,
        examples=["EUR-11", "AFR-22"],
        description="CORDEX domain(s) to look for",
    )
    institute: list[str] | None = Field(
        default=None,
        example=["IPSL", "NCAR"],
        description="Institute name that produced the data",
    )
    driving_model: Iterable[str] | None = Field(
        default=None,
        description="Name of the global climate model used to drive the RCM",
    )
    experiment: Iterable[str] | None = Field(
        default=["historical", "rcp26"],
        description="Name of the experiment type of the simulation",
    )
    experiment_family: Iterable[str] | None = Field(
        default=None,
        example=["RCP", "Historical"],
        description="Type of experiment : weither 'RCP' or 'Historical' or 'All'",
    )
    ensemble: Iterable[str] | None = Field(
        default="r1i1p1", description="Ensemble member"
    )
    rcm_model: Iterable[str] | None = Field(
        default=None,
        example=["WRF381P", "RCA4"],
        description="Name of the regional climate model",
    )
    downscaling_realisation: Iterable[str] | None = Field(
        default=None,
        example=["v1", "v2"],
        description="Version of the downscaling realisation",
    )
    time_frequency: Frequency = Field(
        default=Frequency.MONTHLY,
        example="mon",
        description="Time frequency of the data",
    )
    variable: Iterable[str] | None = Field(
        default=("tas", "tasmin", "tasmax", "pr"), description="Variables requested"
    )
    variable_long_name: Iterable[str] | None = Field(
        default=None, example=[], description="Long name of the variables"
    )

    class Config:
        """Pydantic configuration for the DownClimContext class."""

        arbitrary_types_allowed = True

    @classmethod
    def to_list(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return [v]
        return v

    @field_validator("project")
    @classmethod
    def validate_project(
        cls, project: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not project:
            msg = "No project provided, defaulting to 'CORDEX'"
            warnings.warn(msg, stacklevel=2)
            return "CORDEX"
        return project

    @field_validator("product")
    @classmethod
    def validate_product(
        cls, product: str | Iterable[str] | None
    ) -> str | Iterable[str]:
        if not product:
            msg = "No product provided, defaulting to 'output'"
            warnings.warn(msg, stacklevel=2)
            return "output"
        return product

    @field_validator("experiment", mode="before")
    @classmethod
    def validate_experiment(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            msg = "No experiment provided, defaulting to ['historical', 'rcp26']"
            warnings.warn(msg, stacklevel=2)
            return ["historical", "rcp26"]
        if isinstance(v, str):
            return [v]
        if not any(exp == "historical" for exp in v):
            msg = """Historical experiment is mandatory to associate with projections.
                By default we add 'historical' to the list of experiments."""
            warnings.warn(msg, stacklevel=2)
            return [*v, "historical"]
        return v


def _get_wget_download_lines(wget_file: str) -> list[str]:
    with Path(wget_file).open(encoding="utf-8") as reader:
        return [
            num for num, line in enumerate(reader, 1) if re.match(r".*\.(nc)", line)
        ]


def _get_cordex_wget(
    script: str,
    i: int,
    periods: tuple[(int, int)],
    temp_fold: str,
) -> bool:
    script_name = f"{temp_fold}/wget_{i}.sh"

    # Write script to file
    # Select only required files corresponding to the required periods
    with Path("_" + script_name, "w").open() as writer:
        for line in script.split("\n"):
            if re.match(r".*\.(nc)", line):
                tmin, tmax = (
                    int(year[:4])
                    for year in line.split(".nc")[0].split("_")[-1].split("-")
                )
                if any(period[0] <= tmax and period[1] >= tmin for period in periods):
                    writer.write(line + "\n")
            else:
                writer.write(line + "\n")

    os(script_name).chmod(0o750)
    subprocess.check_output(["/bin/bash", script_name], cwd=temp_fold)
    return True


def get_cordex(
    aois: list[gpd.GeoDataFrame],
    variables: list[str],
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    projection_year: tuple[int, int] = (2071, 2100),
    # time_frequency: str = "mon",
    aggregation: str = "monthly-means",
    domain: str = "EUR-11",
    institute: str = "IPSL",
    model: str = "IPSL-IPSL-CM5A-MR",
    rcm: str = "WRF381P",
    experiment: str = "rcp45",
    ensemble: str = "r1i1p1",
    baseline: str = "chelsa2",
    downscaling: str = "v1",
    esgf_credential: str = "config/credentials_esgf.yml",
    nb_threads: int = 1,
    output_dir: str = "./results/cordex",
    tmp_dir: str = "./results/tmp/cordex",
    keep_tmp_directory: bool = False,
) -> None:
    """
    Get CORDEX data for a given region and period.

    Parameters
    ----------
    base_file: list
        List of baseline files.
    domain: str
        Domain of the region.
    areas: list
        List of areas.
    institute: str
        Institute.
    model: str
        Model.
    experiment: str
        Experiment.
    ensemble: str
        Ensemble.
    baseline: str
        Baseline.
    rcm: str
        RCM.
    downscaling: str
        Downscaling.
    variables: list
        List of variables.
    time_frequency: str
        Time frequency.
    esgf_credential: str
        ESGF credentials.
    threads: int
        Number of threads for downloading in parallel. Default is 1 (i.e. no parallel downloading).
    periods: list
        List of periods.
    aggregation: str
        Aggregation.
    tmp_dir: str
        Temporary directory.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path.is_dir(tmp_dir):
        context = {
            "domain": domain,
            "gcm": model,
            "rcm": rcm,
            "time_frequency": "mon",
            "experiment": experiment,
            "var": variables,
        }
        # connect
        connector = connect_to_esgf(esgf_credential, server=DataProduct.CORDEX.url)
        ctx = connector.new_context(facets="*", **context)
        all_scripts = [res.file_context().get_download_script() for res in ctx.search()]

        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        # download
        pool = mp.Pool(nb_threads)
        pool.starmap_async(
            _get_cordex_wget,
            [(script, i, tmp_dir) for i, script in enumerate(all_scripts)],
        ).get()
        pool.close()
    else:
        print(
            f"Folder '{tmp_dir}' already present, assuming already downloaded data. If this is not the case, "
        )

    # read & prepare
    files = [f"{tmp_dir}/{f}" for f in os.listdir(tmp_dir) if re.match(r".*\.(nc)", f)]
    files_hist = [f for f in files if re.search("_historical", f)]
    files_rcp = [f for f in files if f not in files_hist]
    ds_hist = xr.open_mfdataset(files_hist, parallel=True)
    ds_rcp = xr.open_mfdataset(files_rcp, parallel=True)
    ds_hist = prep_dataset(ds_hist)
    ds_rcp = prep_dataset(ds_rcp)

    check_file = "toto.nc"
    periods = [baseline_year, evaluation_year, projection_year]
    # regrid and write per country
    for aoi in aois:
        base_file = f"{Path(check_file).parent}/{aoi}_base.nc"
        base = xr.open_dataset(base_file)
        regridder_hist = xe.Regridder(ds_hist, base, "bilinear", ignore_degenerate=True)
        ds_hist_r = regridder_hist(ds_hist, keep_attrs=True)
        regridder_rcp = xe.Regridder(ds_rcp, base, "bilinear", ignore_degenerate=True)
        ds_rcp_r = regridder_rcp(ds_rcp, keep_attrs=True)
        for period in periods:
            dmin, dmax = split_period(period)
            if dmax <= "2005-01-01":
                ds_a = (
                    ds_hist_r.sel(time=slice(dmin, dmax))
                    .groupby("time.month")
                    .mean("time")
                )
            else:
                ds_a = (
                    ds_rcp_r.sel(time=slice(dmin, dmax))
                    .groupby("time.month")
                    .mean("time")
                )
            path = f"{Path(check_file).parent}/{aoi}_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"
            ds_a.to_netcdf(path)

    if not keep_tmp_directory:
        shutil.rmtree(tmp_directory)


def get_cordex_from_list(
    aoi: list[gpd.GeoDataFrame],
    cordex_simulations: pd.DataFrame,
    baseline_year: tuple[int, int] = (1980, 2005),
    evaluation_year: tuple[int, int] = (2006, 2019),
    projection_year: tuple[int, int] = (2071, 2100),
    aggregation: Aggregation = Aggregation.MONTHLY_MEAN,
    output_dir: str = "./results/cordex",
    tmp_dir: str = "./results/tmp/cordex",
    baseline_product: DataProduct = DataProduct.CHELSA,
    baseline_output_dir: str = "./results/chelsa2",
    evaluation_product: DataProduct = DataProduct.CHELSA,
    evaluation_output_dir: str = "./results/chelsa2",
    nb_threads: int = 1,
    keep_tmp_dir: bool = False,
) -> None:
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Get AOIs information
    aois_names, _ = get_aoi_informations(aoi)

    # connect
    connector = connect_to_esgf(esgf_credential, server=DataProduct.CORDEX.url)
    context = build_context_from_list(cordex_simulations)
    ctx = connector.new_context(facets="*", **context)
    all_scripts = [res.file_context().get_download_script() for res in ctx.search()]

    # download
    pool = mp.Pool(nb_threads)
    pool.starmap_async(
        _get_cordex_wget,
        [(script, i, tmp_dir) for i, script in enumerate(all_scripts)],
    ).get()
    pool.close()

    # read & prepare
    files = [f"{tmp_dir}/{f}" for f in os.listdir(tmp_dir) if re.match(r".*\.(nc)", f)]
    files_hist = [f for f in files if re.search("_historical", f)]
    files_rcp = [f for f in files if f not in files_hist]
    ds_hist = xr.open_mfdataset(files_hist, parallel=True)
    ds_rcp = xr.open_mfdataset(files_rcp, parallel=True)
    ds_hist = prep_dataset(ds_hist, DataProduct.CORDEX)
    ds_rcp = prep_dataset(ds_rcp, DataProduct.CORDEX)

    # Define time periods
    period_reference_products = [
        baseline_product,
        evaluation_product,
        evaluation_product,
    ]
    period_reference_outputs = [
        baseline_output_dir,
        evaluation_output_dir,
        evaluation_output_dir,
    ]
    periods = [baseline_year, evaluation_year, projection_year]

    # regrid and write per aoi and period
    for period_reference_product, period_reference_output, period in zip(
        period_reference_products, period_reference_outputs, periods, strict=False
    ):
        tmin, tmax = split_period(period)
        if aggregation == Aggregation.MONTHLY_MEAN:
            ds_clim = get_monthly_climatology(ds.sel(time=slice(tmin, tmax)))
        else:
            msg = "Currently only monthly-means aggregation available!"
            raise ValueError(msg)
        for aoi_n in aois_names:
            output_file = f"{output_dir}/{aoi_n}_CORDEX_{domain}_{institution}_{source}_{experiment}_{member}_none_none_{baseline}_{aggregation.value}_{period[0]}-{period[1]}.nc"
            reference_file = f"{period_reference_output}/{aoi_n}_{period_reference_product.product_name}_{aggregation.value}_{period[0]}-{period[1]}.nc"
            reference = xr.open_dataset(reference_file)
            regridder = xe.Regridder(
                ds_clim, reference, "bilinear", ignore_degenerate=True
            )
            ds_clim_r = regridder(ds_clim, keep_attrs=True)
            path = f"{Path(check_file).parent}/{aoi}_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"
            ds_a.to_netcdf(output_file)

    if not keep_tmp_dir:
        shutil.rmtree(tmp_dir)

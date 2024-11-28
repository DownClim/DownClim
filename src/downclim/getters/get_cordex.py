from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import multiprocess as mp
import xarray as xr
import xesmf as xe

from .connectors import connect_to_esgf, data_urls
from .utils import (
    prep_dataset,
    split_period,
)


def _get_wget_download_lines(wget_file: str) -> list[str]:
    with Path(wget_file).open() as reader:
        return [
            num for num, line in enumerate(reader, 1) if re.match(r".*\.(nc)", line)
        ]


def _get_cordex_wget(
    script: str,
    i: int,
    periods: tuple[(int, int)],
    temp_fold: str,
):
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
    periods: tuple[(int, int)] = ((1980, 2005), (2006, 2019), (2071, 2100)),
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
    threads: int = 1,
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

    tmp_directory = "./results/tmp/cordex"
    output_directory = "./results/cordex"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if not Path.is_dir(tmp_directory):
        context = {
            "domain": domain,
            "gcm": model,
            "rcm": rcm,
            "time_frequency": "mon",
            "experiment": experiment,
            "var": variables,
        }
        # connect
        connector = connect_to_esgf(esgf_credential, server=data_urls["esgf"])
        ctx = connector.new_context(facets="*", **context)
        all_scripts = [res.file_context().get_download_script() for res in ctx.search()]

        Path(tmp_directory).mkdir(parents=True, exist_ok=True)

        # download
        pool = mp.Pool(threads)
        pool.starmap_async(
            _get_cordex_wget,
            [(script, i, tmp_directory) for i, script in enumerate(all_scripts)],
        ).get()
        pool.close()
    else:
        print(
            f"Folder '{tmp_directory}' already present, assuming already downloaded data. If this is not the case, "
        )

    # read & prepare
    files = [
        f"{tmp_directory}/{f}"
        for f in os.listdir(tmp_directory)
        if re.match(r".*\.(nc)", f)
    ]
    files_hist = [f for f in files if re.search("_historical", f)]
    files_rcp = [f for f in files if f not in files_hist]
    ds_hist = xr.open_mfdataset(files_hist, parallel=True)
    ds_rcp = xr.open_mfdataset(files_rcp, parallel=True)
    ds_hist = prep_dataset(ds_hist)
    ds_rcp = prep_dataset(ds_rcp)

    check_file = "toto.nc"
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

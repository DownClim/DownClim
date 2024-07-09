from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import multiprocess as mp
import numpy as np
import xarray as xr
import xesmf as xe

from .utils import (
    add_offsets,
    connect_to_esgf,
    convert_cf_to_dt,
    scale_factors,
    split_period,
    variables_attributes,
)


def get_cordex_nc(
    scripts: Iterable[str],
    i: int,
    temp_fold: str = "tmp/cordex/",
):
    script_name = "wget_" + str(i) + ".sh"
    with Path.open(temp_fold + script_name, "w") as writer:
        writer.write(scripts[i])
    Path.chmod(temp_fold + script_name, 0o750)
    subprocess.check_output(["/bin/bash", script_name], cwd=temp_fold)
    return True


def prep_proj(ds):
    cf = type(ds["time"].to_numpy()[0]) is not np.datetime64
    if cf:  # only cftime if not dt but should include more cases
        ds["time"] = [*map(convert_cf_to_dt, ds.time.values)]
    for key in ds:
        ds[key] = ds[key] * scale_factors["cordex"][key] + add_offsets["cordex"][key]
        ds[key].attrs = variables_attributes[key]
    return ds


def get_cordex(
    aois: list[str],
    base_files: list[str],
    domain: str,
    institute: str,
    model: str,
    rcm: str,
    experiment: str = "rcp75",
    ensemble: str = "r1i1p1",
    baseline: str = "chelsa2",
    downscaling: str = "v1",
    variables: Iterable[str] = ["tas", "pr", "tasmin", "tasmax"],
    esgf_credential: str = "config/credentials_esgf.yml",
    threads: int = 1,
    periods: Iterable[str] = ["1980-2005", "2006-2019", "2071-2100"],
    aggregation: str = "monthly-means",
    tmp_dir: str = "tmp/cordex",
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

    if not Path.is_dir(tmp_dir):
        context = {
            "domain": domain,
            "gcm": model,
            "rcm": rcm,
            "experiment": experiment,
            "var": variables,
        }
        # connect
        ctx = connect_to_esgf(context, esgf_credential)
        all_scripts = [res.file_context().get_download_script() for res in ctx.search()]

        # download
        Path.mkdir(tmp_dir)
        pool = mp.Pool(threads)
        pool.starmap_async(
            get_cordex_nc, [(all_scripts, i) for i in list(range(len(all_scripts)))]
        ).get()
        pool.close()
    else:
        print("Folder already present, assuming already downloaded data.")

    # read & prepare
    files = [tmp_dir + f for f in os.listdir(tmp_dir) if re.match(r".*\.(nc)", f)]
    files_hist = [f for f in files if re.search("_historical", f)]
    files_rcp = [f for f in files if f not in files_hist]
    ds_hist = xr.open_mfdataset(files_hist, parallel=True)
    ds_rcp = xr.open_mfdataset(files_rcp, parallel=True)
    ds_hist = prep_proj(ds_hist)
    ds_rcp = prep_proj(ds_rcp)

    check_file = "toto.nc"
    # regrid and write per country
    for i in list(range(len(aois))):
        base = xr.open_dataset(base_files[i])
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
            path = f"{Path(check_file).parent}/{aois[i]}_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{baseline}_{aggregation}_{period}.nc"
            ds_a.to_netcdf(path)

    shutil.rmtree(tmp_dir)
    with Path.open(check_file, "w") as writer:
        writer.write("Done.")

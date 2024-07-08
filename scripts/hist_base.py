# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
in_file =  snakemake.input[0]
area_file =  snakemake.input[1]
out_file = snakemake.output[0]
area = snakemake.params.area
origin = snakemake.params.origin
domain = snakemake.params.domain
institute = snakemake.params.institute
model = snakemake.params.model
experiment = snakemake.params.experiment
ensemble = snakemake.params.ensemble
rcm = snakemake.params.rcm
downscaling = snakemake.params.downscaling
base = snakemake.params.base
aggregation = snakemake.params.aggregation
period_proj = snakemake.params.period_proj
period_proj = snakemake.params.period_proj
period_eval = snakemake.params.period_eval
ds_method = snakemake.params.ds_method
      
# libs
import pandas as pd
import xarray as xr
import geopandas as gp
import numpy as np

# code
area_shp = gp.read_file(area_file)
ds = xr.open_dataset(in_file).rio.clip(area_shp.geometry.values, area_shp.crs)
variables = list(ds.keys())
months = list(range(1,13))
a = []
for v in variables:
    for m in months:
        if v == "pr":
            low = 0
            high = 2000
            step = 10
        else :
            low = 0
            high = 1000
            step = 1
        bins= np.arange(low, high, step)
        labels= np.arange(low+step/2, high-step/2, step)
        out = pd.cut(ds.sel(month=m)[v].values.ravel(), bins=bins, labels=labels)
        res = out.value_counts().to_frame()
        res['bin'] = res.index
        res.insert(0, "month", m)
        res.insert(0, "variable", v)
        a.append(res)
tab = pd.concat(a)
tab.insert(0, "area", area)
tab.insert(0, "origin", origin)
tab.insert(0, "type", "raw")
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
tab[["area", "origin", "type", "domain", "institute", "model", "experiment", "ensemble", "rcm", "downscaling", "base", 
     "aggregation", "period_proj", "period_eval", "ds_method", "month", "variable", "bin", "count"]].to_csv(out_file, sep="\t", index=False)

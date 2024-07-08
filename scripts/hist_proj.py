# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
ds_file =  snakemake.input[0]
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
baseline = snakemake.params.base
aggregation = snakemake.params.aggregation
period_proj = snakemake.params.period_proj
period_proj = snakemake.params.period_proj
period_eval = snakemake.params.period_eval
ds_method = snakemake.params.ds_method
      
# test
# ds_file = "results/downscaled/New-Caledonia_CMIP6_world_MRI_MRI-ESM2-0_ssp126_r1i1p1f1_none_none_chelsa2_monthly-means_2006-2019_1980-2005_bc.nc"
# area_file = "results/areas/New-Caledonia.shp"
# area="New-Caledonia"
# origin="CMIP6"
# domain="world"
# institute="MRI"
# model="MRI-ESM2-0"
# experiment="ssp126" 
# ensemble="r1i1p1f1"
# rcm="none"
# downscaling="none"
# baseline="chelsa2"
# aggregation="monthly-means"
# period_proj="2006-2019"
# period_eval="1980-2005"
# ds_method="bc"
# base_eval="chelsa2"

# libs
import pandas as pd     
import numpy as np   
import xarray as xr
import geopandas as gp

# funs
def get_hist(pred_ds, type_in):
    variables = list(pred_ds.keys())
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
            out = pd.cut(pred_ds.sel(month=m)[v].values.ravel(), bins=bins, labels=labels)
            res = out.value_counts().to_frame()
            res['bin'] = res.index
            res.insert(0, "month", m)
            res.insert(0, "variable", v)
            a.append(res)
    tab = pd.concat(a)
    tab.insert(0, "area", area)
    tab.insert(0, "origin", origin)
    tab.insert(0, "type", type_in)
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
    tab[["area", "origin", "type", "domain", "institute", "model", "experiment", "ensemble", "rcm", "downscaling", "base", 
     "aggregation", "period_proj", "period_eval", "ds_method", "month", "variable", "bin", "count"]]
    return(tab)

# code
proj_file = "results/projections/" + area + "_" + origin + "_" + domain + "_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_" + rcm + "_" + downscaling + "_" + baseline + "_" + aggregation + "_" + period_eval + ".nc"
area_shp = gp.read_file(area_file)
ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
proj = xr.open_dataset(proj_file).rio.clip(area_shp.geometry.values, area_shp.crs)
pd.concat([get_hist(ds, "downscaled"), get_hist(proj, "raw")]).to_csv(out_file, sep="\t", index=False)

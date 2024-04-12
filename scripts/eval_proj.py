# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
ds_file =  snakemake.input[0]
base_file = snakemake.input[1]
area_file =  snakemake.input[2]
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
base_eval = snakemake.params.base_eval
variables = snakemake.params.variables

# test
# ds_file = "results/downscaled/New-Caledonia_CORDEX_AUS-22_GERICS_NCC-NorESM1-M_rcp85_r1i1p1_REMO2015_v1_chelsa2_monthly-means_2006-2019_1980-2005_bc.nc"
# base_file = "results/baselines/New-Caledonia_chelsa2_monthly-means_1980-2005.nc"
# area_file = "results/areas/New-Caledonia.shp"
# area="New-Caledonia"
# origin="CORDEX"
# domain="AUS-22"
# institute="GERICS"
# model="NCC-NorESM1-M"
# experiment="rcp85" 
# ensemble="r1i1p1"
# rcm="REMO2015"
# downscaling="v1"
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
def get_eval(pred_ds, base_ds, type_in):
    months = list(range(1,13))
    a = []
    for v in variables:
        for m in months:
            pred_0 = pred_ds.sel(month=m)[v].values.ravel()
            pred = pred_0[~np.isnan(pred_0)]
            obs = base_ds.sel(month=m)[v].values.ravel()
            obs = obs[~np.isnan(pred_0)]
            d = {
                'metric': ['CC', 'RMSE', "SDE", "bias"],
                'value': [np.corrcoef(pred, obs)[1,0], np.sqrt(np.mean(pow(pred - obs, 2))), np.std(pred - obs), np.mean(pred - obs)]
            }
            res = pd.DataFrame(data = d)
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
    tab.insert(0, "base", base)
    tab.insert(0, "aggregation", aggregation)
    tab.insert(0, "period_proj", period_proj)
    tab.insert(0, "period_eval", period_eval)
    tab.insert(0, "ds_method", ds_method)
    tab.insert(0, "base_eval", base_eval)
    tab[["area", "origin", "type", "domain", "institute", "model", "experiment", "ensemble", "rcm", "downscaling", "base", 
     "aggregation", "period_proj", "period_eval", "ds_method", "base_eval",
     "month", "variable", "metric", "value"]]
    return(tab)

# code
proj_file = "results/projections/" + area + "_" + origin + "_" + domain + "_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_" + rcm + "_" + downscaling + "_" + baseline + "_" + aggregation + "_" + period_eval + ".nc"
area_shp = gp.read_file(area_file)
ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
proj = xr.open_dataset(proj_file).rio.clip(area_shp.geometry.values, area_shp.crs)
base = xr.open_dataset(base_file).rio.clip(area_shp.geometry.values, area_shp.crs)
pd.concat([get_eval(ds, base, "downscaled"), get_eval(proj, base, "raw")]).to_csv(out_file, sep="\t", index=False)

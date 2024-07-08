# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
base_files = snakemake.input
areas = snakemake.params.area
institute = snakemake.params.institute
model = snakemake.params.model
experiment = snakemake.params.experiment
ensemble = snakemake.params.ensemble
baseline = snakemake.params.base
variables = snakemake.params.variables
time_frequency = snakemake.params.time_frequency
check_file = snakemake.output[0]
threads = snakemake.threads
periods = snakemake.params.periods
aggregation = snakemake.params.aggregation

# test
# base_files = ["results/baselines/New-Caledonia_chelsa2_monthly-means_1980-2005.nc", 
#               "results/baselines/Côte-d'Ivoire_chelsa2_monthly-means_1980-2005.nc", 
#               "results/baselines/French-Guiana_chelsa2_monthly-means_1980-2005.nc"]
# check_file = ["results/projections/_CMIP6_world_CAS_FGOALS-g3_ssp585_r1i1p1f1_none_none_chelsa2_done.txt"]
# areas = ["New-Caledonia", "Côte-d'Ivoire", "French-Guiana"]
# institute = "CAS"
# model = "FGOALS-g3"
# experiment = "ssp585"
# ensemble = "r1i1p1f1"
# baseline = "chelsa2"
# variables =  ["pr", "tas", "tasmin", "tasmax"]
# time_frequency = "mon"
# threads = 10
# periods= ["1980-2005", "2006-2019", "2071-2100"]
# aggregation="monthly-means"

# libs
import os
import pandas as pd
import gcsfs
import xarray as xr
import xesmf as xe
import numpy as np
from datetime import datetime as dt

# funs
def convert_cf_to_dt(x):
  return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')

# conversions
if time_frequency == "mon":
        table_id = "Amon"
else :
    raise Exception("Currently only monthly time frequency available!")

# code
gcs = gcsfs.GCSFileSystem(token='anon')
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
a = []
for exp in ["historical", experiment]:
  if exp=="historical":
    activity="CMIP"
  else:
    activity="ScenarioMIP"
  for var in variables:
    search_string = "activity_id == '" + activity + "' & table_id == '" + table_id + "' & variable_id == '" + var + "' & experiment_id == '" + exp + "' & institution_id == '" + institute + "' & source_id == '" + model + "' & member_id == '" + ensemble + "'"
    df_ta = df.query(search_string)
    zstore = df_ta.zstore.values[-1]
    mapper = gcs.get_mapper(zstore)
    a.append(xr.open_zarr(mapper, consolidated=True))
ds = xr.merge(a, compat='override')
ds['time'] = np.sort(ds['time'].values)
dmin = min(map(lambda x: x.split("-")[0], periods)) + "-01-01"
dmax = max(map(lambda x: x.split("-")[1], periods)) + "-01-01"
cf = type(ds["time"].values[0]) is not np.datetime64
if cf:
  ds['time'] = [*map(convert_cf_to_dt, ds.time.values)]   
ds = ds.sel(time=slice(dt.strptime(dmin, '%Y-%m-%d'), dt.strptime(dmax, '%Y-%m-%d')))
ds = ds.chunk(chunks = {'time':100, 'lat': 400, 'lon': 400})
if 'pr' in list(ds.keys()):
  ds['pr'] = ds.pr * 60*60*24*ds.time.dt.days_in_month  # s-1 to month-1
  ds.pr.attrs = {'standard_name': 'precipitation', 
                 'long_name': 'Monthly precipitation',
                 'units': 'mm month-1', 
                 'explanation' : 'Precipitation in the earth\'s atmosphere, monthly means precipitation of water in all phases.'}
if 'tas' in list(ds.keys()):
  ds['tas'] = ds.tas - 273.15  # K to °C
  ds.tas.attrs = {'standard_name': 'temperature at surface', 
                  'long_name': 'Monthly mean daily air temperature',
                  'units': '°C', 
                  'explanation' : 'Monthly mean air temperatures at 2 meters.'}
if 'tasmin' in list(ds.keys()):
  ds['tasmin'] = ds.tasmin - 273.15  # K to °C
  ds.tasmin.attrs = {'standard_name': 'minimum temperature at surface', 
               'long_name': 'Monthly minimum daily air temperature',
               'units': '°C', 
               'explanation' : 'Monthly minimum air temperatures at 2 meters.'}
if 'tasmax' in list(ds.keys()):
  ds['tasmax'] = ds.tasmax - 273.15  # K to °C
  ds.tasmax.attrs = {'standard_name': 'maximum temperature at surface', 
                     'long_name': 'Monthly maximum daily air temperature',
                     'units': '°C', 
                     'explanation' : 'Monthly maximum air temperatures at 2 meters.'}
for period in periods:
  dmin = period.split("-")[0] + "-01-01"
  dmax = period.split("-")[1] + "-01-01"
  ds_a = ds.sel(time=slice(dmin, dmax)).groupby("time.month").mean("time")
  for i in list(range(len(areas))):
    base = xr.open_dataset(base_files[i])
    regridder = xe.Regridder(ds_a, base, "bilinear")
    ds_r = regridder(ds_a, keep_attrs=True)
    path = os.path.dirname(check_file) + "/" + areas[i] + "_CMIP6_world_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_none_none_" + baseline + "_" + aggregation + "_" + period + ".nc"
    ds_r.to_netcdf(path)

f = open(check_file, "a")
f.write("Done.")
f.close()

# # test
# ## v1
# anom = ds.sel(time=slice("2071-01-01", "2100-01-01")).groupby("time.month").mean("time") - ds.sel(time=slice("1980-01-01", "2005-01-01")).groupby("time.month").mean("time")
# i = 0
# base = xr.open_dataset(base_files[i])
# regridder = xe.Regridder(anom, base, "bilinear")
# anom_r = regridder(anom, keep_attrs=True)
# anom_r.to_netcdf("anom_v1.nc")
# ## v2
# fut = ds.sel(time=slice("2071-01-01", "2100-01-01")).groupby("time.month").mean("time")
# base = xr.open_dataset(base_files[i])
# regridder = xe.Regridder(fut, base, "bilinear")
# fut_r = regridder(fut, keep_attrs=True)
# hist = ds.sel(time=slice("1980-01-01", "2005-01-01")).groupby("time.month").mean("time")
# regridder = xe.Regridder(hist, base, "bilinear")
# hist_r = regridder(hist, keep_attrs=True)
# anom_r2 = fut_r - hist_r
# anom_r2.to_netcdf("anom_v2.nc")
# ## comp
# diff = anom_r - anom_r2
# diff.to_netcdf("diff.nc")

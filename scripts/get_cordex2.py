# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
base_files = snakemake.input
domain = snakemake.params.domain
areas = snakemake.params.area
domain = snakemake.params.domain
institute = snakemake.params.institute
model = snakemake.params.model
experiment = snakemake.params.experiment
ensemble = snakemake.params.ensemble
baseline = snakemake.params.base
rcm = snakemake.params.rcm
downscaling = snakemake.params.downscaling
variables = snakemake.params.variables
time_frequency = snakemake.params.time_frequency
esgf_credential = snakemake.params.esgf_credential
check_file = snakemake.output[0]
cores = snakemake.threads
periods = snakemake.params.periods
aggregation = snakemake.params.aggregation
temp_fold = snakemake.params.tmp

# test
# base_files = ["results/baselines/New-Caledonia_chelsa2_monthly-means_1980-2005.nc",
#               "results/baselines/Vanuatu_chelsa2_monthly-means_1980-2005.nc"]
# check_file = "results/projections/CORDEX_AUS-22_CLMcom-HZG_MOHC-HadGEM2-ES_rcp26_r1i1p1_CCLM5-0-15_v1_chelsa2_done.txt"
# areas = ["New-Caledonia", "Vanuatu"]
# domain = "AUS-22"
# institute = "CLMcom-HZG"
# model = "MOHC-HadGEM2-ES"
# experiment = "rcp26"
# ensemble = "r1i1p1"
# rcm = "CCLM5-0-15"
# downscaling = "v1"
# baseline = "chelsa2"
# variables = ["tas"]
# time_frequency = "mon"
# esgf_credential = "config/credentials_esgf.yml"
# cores = 10
# periods= ["1980-2005", "2006-2019", "2071-2100"]
# aggregation="monthly-means"
# temp_fold="results/projections/CORDEX_AUS-22_CLMcom-HZG_MOHC-HadGEM2-ES_rcp26_r1i1p1_CCLM5-0-15_v1_chelsa2_tmp/"

# libs
import os
import re
import subprocess
import shutil
from pyesgf.search import SearchConnection
import yaml
from pyesgf.logon import LogonManager
import xarray as xr
import xesmf as xe
from datetime import datetime as dt
import numpy
import ray

# funs
@ray.remote
def get_nc(scripts, i):
  script_name = "wget_" + str(i) + ".sh"
  with open(temp_fold + script_name, "w") as writer:
    writer.write(scripts[i])
  os.chmod(temp_fold + script_name, 0o750)
  subprocess.check_output(['/bin/bash', script_name], cwd=temp_fold)
  return(True)

def convert_cf_to_dt(x):
  return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')

# connect
creds = yaml.safe_load(open(esgf_credential, 'r'))
lm = LogonManager()
lm.logon_with_openid(openid=creds['openid'], password=creds['pwd'], interactive=False, bootstrap=True)

# list
server = 'https://esgf-node.ipsl.upmc.fr/esg-search/'
conn = SearchConnection(server, distrib=True)
# keys: https://esgf-node.llnl.gov/esg-search/search?project=CORDEX&facets=*&limit=0
ctx = conn.new_context(
  facets='*',
  project = "CORDEX",
  domain = domain,
  institute = institute,
  driving_model = model,
  experiment = [experiment, "historical"],
  ensemble = ensemble,
  rcm_name= rcm,
  rcm_version = downscaling,
  time_frequency = time_frequency,
  variable = variables
)
all_scripts = list(map(lambda res: res.file_context().get_download_script(), ctx.search()))

# download
os.mkdir(temp_fold)
ray.init(num_cpus=cores, ignore_reinit_error=True)
job_objs = [get_nc.remote(all_scripts, i) for i in list(range(len(all_scripts)))]
ray.get(job_objs)
ray.shutdown()

# read & prepare
files = [temp_fold + f for f in os.listdir(temp_fold) if re.match(r'.*\.(nc)', f)]
ds = xr.open_mfdataset(files, parallel=True)
cf = type(ds["time"].values[0]) is not numpy.datetime64
if cf: # only cftime if not dt but should include more cases
  ds['time'] = [*map(convert_cf_to_dt, ds.time.values)]
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

# regrid and write per country
for period in periods:
  dmin = period.split("-")[0] + "-01-01"
  dmax = period.split("-")[1] + "-01-01"
  ds_a = ds.sel(time=slice(dmin, dmax)).groupby("time.month").mean("time")
  for i in list(range(len(areas))):
    base = xr.open_dataset(base_files[i])
    regridder = xe.Regridder(ds_a, base, "bilinear")
    ds_r = regridder(ds_a, keep_attrs=True)
    path = os.path.dirname(check_file) + "/" + areas[i] + "_CORDEX_" + domain + "_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_" + rcm + "_" + downscaling + "_" + baseline + "_" + aggregation + "_" + period + ".nc"
    ds_r.to_netcdf(path)

shutil.rmtree(temp_fold)
f = open(check_file, "a")
f.write("Done.")
f.close()

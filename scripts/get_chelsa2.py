# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
area_files = snakemake.input
areas = snakemake.params.area
variables = snakemake.params.variables
time_frequency = snakemake.params.time_frequency
base_years = snakemake.params.base_years
temp_fold = snakemake.params.tmp
nc_files = snakemake.output
cores = snakemake.threads
periods = snakemake.params.periods
aggregation = snakemake.params.aggregation

# test
# area_files = ["results/areas/New-Caledonia.shp", "results/areas/Vanuatu.shp"]
# areas=["New-Caledonia", "Vanuatu"]
# variables = ["tas", "tasmin", "tasmax", "pr"]
# time_frequency = "mon"
# temp_fold = "results/baselines/chelsa2_tmp"
# base_years = "2004-2007"
# nc_files = ["results/baselines/New-Caledonia_chelsa2_monthly-means_1980-2005.nc", 
#             "results/baselines/New-Caledonia_chelsa2_monthly-means_2006-2019.nc", 
#             "results/baselines/Vanuatu_chelsa2_monthly-means_1980-2005.nc", 
#             "results/baselines/Vanuatu_chelsa2_monthly-means_2006-2019.nc"]
# cores=3
# periods= ["1980-2005", "2006-2019"]
# aggregation="monthly-means"
        
# libs
import os
import shutil
import xarray as xr
import rioxarray as rio
import pandas as pd
import geopandas
import datetime
import multiprocessing as mp
        
# funs
def get_year(year, areas, var, time_freq, month_max, temp_fold):
        areas_names = list(map(lambda x: x.NAME_0.values[0], areas))
        print("Getting year " + str(year) + " for variables " + var + " and areas " + str(areas_names))
        areas_bounds = list(map(lambda x: x.bounds, areas))
        a = {key: list() for key in areas_names}
        for month in range(1, month_max+1):
                url = 'https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/' + time_freq + '/' + var + "/" \
                        + 'CHELSA_' + var + '_' + '%02d' % (month,) + '_' + str(year) + "_V.2.1.tif"
                ds = rio.open_rasterio(url, decode_coords="all").to_dataset('band').rename_vars({1 : var})
                for i, _ in enumerate(areas): 
                        a[areas_names[i]].append(ds.rio.clip_box(minx=areas_bounds[i].minx[0], 
                                                                 miny=areas_bounds[i].miny[0],
                                                                 maxx=areas_bounds[i].maxx[0], 
                                                                 maxy=areas_bounds[i].maxy[0]))
        paths = {}
        for area_name in areas_names: 
                ds_year = xr.concat(a[area_name], pd.Index(pd.date_range(datetime.datetime(year, 1, 1), periods=month_max, freq="M"), name="time"))
                del a[area_name]
                ds_year = ds_year[["time", "x", "y", var]]
                if var == "pr":
                        ds_year[var].values = ds_year[var].values * 0.01
                        ds_year.pr.attrs = {'standard_name': 'precipitation', 
                                           'long_name': 'Monthly precipitation',
                                           'units': 'mm month-1', 
                                           'explanation' : 'Precipitation in the earth\'s atmosphere, monthly means precipitation of water in all phases.'}
                elif var == "tas":
                        ds_year[var] = ds_year[var] * 0.1 - 273.15
                        ds_year.tas.attrs = {'standard_name': 'temperature at surface', 
                                            'long_name': 'Monthly mean daily air temperature',
                                            'units': '°C', 
                                            'explanation' : 'Monthly mean air temperatures at 2 meters.'}
                elif var == "tasmin":
                        ds_year[var] = ds_year[var] * 0.1 - 273.15
                        ds_year.tasmin.attrs = {'standard_name': 'minimum temperature at surface', 
                                               'long_name': 'Monthly minimum daily air temperature',
                                               'units': '°C', 
                                               'explanation' : 'Monthly minimum air temperatures at 2 meters.'}
                elif var == "tasmax":
                        ds_year[var] = ds_year[var] * 0.1 - 273.15
                        ds_year.tasmax.attrs = {'standard_name': 'maximum temperature at surface', 
                                               'long_name': 'Monthly maximum daily air temperature',
                                               'units': '°C', 
                                               'explanation' : 'Monthly maximum air temperatures at 2 meters.'}
                else:   
                        "Problem, variable "+ var + " not recognized, you should use one of the following: tas, tasmin, tasmax, pr."
                paths[area_name] = temp_fold + "/" + area_name + "_" + var + "_" + str(year) + ".nc"
                ds_year.chunk({'time':1, 'x':200, 'y':200}).to_netcdf(paths[area_name])
                del ds_year
        return(paths)
  
# code
os.mkdir(temp_fold)

areas = list(map(geopandas.read_file, area_files))
areas_names = list(map(lambda x: x.NAME_0.values[0], areas))

base_years = list(map(int, base_years.split("-")))
years = list(range(base_years[0], base_years[1]+1))
if "pr" in variables and base_years[1] >= 2013:
        years.remove(2013) # issue with the tif
if "pr" in variables and base_years[1] >= 2016:
        years.remove(2016) # issue with the tif                

if time_frequency == "mon":
        tf = "monthly"
        months = 12
if time_frequency != "mon":
        raise Exception("Currently only monthly time frequency available!")

pool = mp.Pool(cores)
paths=[]
for v in variables:
        paths.append(pool.starmap_async(get_year, [(y, areas, v, tf, months, temp_fold) for y in years]).get())
pool.close()
paths2 = {key: list() for key in areas_names}
for path in paths:
        for p in path:
                for area_name in areas_names:
                        paths2[area_name].append(p[area_name])
del paths
      
for i, area_name in enumerate(areas_names):
        print("Merging files for area " + area_name + "...")
        ds=xr.open_mfdataset(paths2[area_name], decode_coords="all", parallel=True)
        for j, period in enumerate(periods):
                dmin = period.split("-")[0] + "-01-01"
                dmax = period.split("-")[1] + "-01-01"
                ds_a = ds.sel(time=slice(dmin, dmax)).groupby("time.month").mean("time")
                path = os.path.dirname(nc_files[0]) + "/" + area_name + "_chelsa2_" + aggregation + "_" + period + ".nc"
                ds_a.to_netcdf(path)
        
shutil.rmtree(temp_fold)

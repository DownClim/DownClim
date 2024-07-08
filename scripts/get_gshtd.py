# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
area_files = snakemake.input
areas = snakemake.params.area
variables = snakemake.params.variables
time_frequency = snakemake.params.time_frequency
nc_files = snakemake.output
periods = snakemake.params.periods
aggregation = snakemake.params.aggregation

# test
# area_files = ["results/areas/New-Caledonia.shp"]
# areas=["New-Caledonia"]
# variables = ["tas", "tasmin", "tasmax"]
# time_frequency = "mon"
# nc_files = ["results/baselines/New-Caledonia_gshtd_monthly-means_1980-2005.nc", 
#             "results/baselines/New-Caledonia_gshtd_monthly-means_2006-2019.nc"]
# periods= ["1980-2005", "2006-2019"]
# aggregation="monthly-means"
        
# libs
import os
import geopandas
import ee
import xarray as xr

# funs
def get_var(var, leg, period):
        dmin = period.split("-")[0] + "-01-01"
        dmax = period.split("-")[1] + "-01-01"
        if(var == "tas"):
                col = "TMEAN"
        if(var == "tasmin"):
                col = "TMIN"
        if(var == "tasmax"):
                col = "TMAX"
        ic = ee.ImageCollection("projects/sat-io/open-datasets/GSHTD/" + col).filterDate(dmin, dmax)
        ds = xr.open_dataset(
                ic,
                engine='ee',
                projection=ic.first().select(0).projection(),
                geometry=leg
        )
        ds = ds.transpose('time', 'lat', 'lon')
        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        ds = ds.rio.write_crs("epsg:4362")
        ds = ds.resample(time="M").mean()
        ds = ds.groupby("time.month").mean("time")
        ds = ds.where(ds.b1 > 0)
        ds['b1'] = ds.b1*0.02 - 273.15  # K to °C
        if(var == "tas"):
                std_name = "temperature"
        if(var == "tasmin"):
                std_name = "minimum temperature"
        if(var == "tasmax"):
                std_name = "maximum temperature"
        if(var == "tas"):
                type = "mean"
        if(var == "tasmin"):
                type = "minimum"
        if(var == "tasmax"):
                type = "maximum"
        ds.b1.attrs = {'standard_name': std_name + ' at surface', 
                        'long_name': 'Monthly ' + type + ' daily air temperature',
                        'units': '°C', 
                        'explanation' : 'Monthly ' + type + ' air temperatures at 2 meters.'}
        ds = ds.rename({'b1': var})
        return(ds)

def get_gshtd(bounds, period):
        leg1 = ee.Geometry.Rectangle(bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0])
        ds_tas = get_var("tas", leg1, period)
        ds_tasmin = get_var("tasmin", leg1, period)
        ds_tasmax = get_var("tasmax", leg1, period)
        ds = xr.merge([ds_tas, ds_tasmax, ds_tasmin])
        return(ds)

# code
areas = list(map(geopandas.read_file, area_files))
areas_names = list(map(lambda x: x.NAME_0.values[0], areas))
areas_bounds = list(map(lambda x: x.bounds, areas))
if time_frequency != "mon":
        raise Exception("Currently only monthly time frequency available!")
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
for i, area_name in enumerate(areas_names):
        for j, period in enumerate(periods):
                ds = get_gshtd(areas_bounds[i], period)
                path = os.path.dirname(nc_files[0]) + "/" + area_name + "_gshtd_" + aggregation + "_" + period + ".nc"
                ds.to_netcdf(path)
        
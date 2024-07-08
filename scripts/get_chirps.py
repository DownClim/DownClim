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
# variables = ["pr"]
# time_frequency = "mon"
# nc_files = ["results/baselines/New-Caledonia_chirps_monthly-means_1980-2005.nc", 
#             "results/baselines/New-Caledonia_chirps_monthly-means_2006-2019.nc"]
# periods= ["1980-2005", "2006-2019"]
# aggregation="monthly-means"
        
# libs
import os
import geopandas
import ee
import xarray as xr

# funs
def get_chirps(bounds, period):
        dmin = period.split("-")[0] + "-01-01"
        dmax = period.split("-")[1] + "-01-01"
        ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(dmin, dmax)
        leg1 = ee.Geometry.Rectangle(bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0])
        ds = xr.open_mfdataset(
                [ic],
                engine='ee',
                projection=ic.first().select(0).projection(),
                geometry=leg1
        )
        ds = ds.transpose('time', 'lat', 'lon')
        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        ds = ds.rio.write_crs("epsg:4362")
        ds = ds.resample(time="M").sum()
        ds = ds.groupby("time.month").mean("time")
        ds = ds.rename({'precipitation' : 'pr'})
        ds.pr.attrs = {'standard_name': 'precipitation', 
                        'long_name': 'Monthly precipitation',
                        'units': 'mm month-1', 
                        'explanation' : 'Precipitation in the earth\'s atmosphere, monthly means precipitation of water in all phases.'}
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
                ds = get_chirps(areas_bounds[i], period)
                path = os.path.dirname(nc_files[0]) + "/" + area_name + "_chirps_" + aggregation + "_" + period + ".nc"
                ds.to_netcdf(path)

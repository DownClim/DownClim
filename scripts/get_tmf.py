# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
future_file =  snakemake.input[0]
present_file =  snakemake.input[1]
rsds_file = snakemake.input[2]
area_file =  snakemake.input[3]
out_file = snakemake.output[0]
      
# test
# rsds_file = "results/tmf/rsds/Côte-d'Ivoire.nc"
# future_file = "results/downscaled/Côte-d'Ivoire_CMIP6_world_CAS_FGOALS-g3_ssp126_r1i1p1f1_none_none_chelsa2_monthly-means_2071-2100_1980-2005_bc.nc"
# present_file = "results/baselines/Côte-d'Ivoire_chelsa2_monthly-means_2006-2019.nc"
# area_file = "results/areas/Côte-d'Ivoire.shp"

# libs
import xarray as xr
import geopandas as gp
import numpy as np

# funs
def prep_tmf(proj, 
             rsds,
             pet_pr_max = 1,
             tmin = 16,
             prmin = 1000 ):
    ds = xr.merge([proj, rsds])
    ds['pet'] = 0.0023 * ds.rsds * (ds.tas + 17.8) * pow((ds.tasmax - ds.tasmin), 0.5)
    ds.pet.attrs = {'standard_name': 'potential evapotranspiration', 
                'long_name': 'Monthly potential evapotranspiration',
                'units': 'mm month-1', 
                'explanation' : 'Potential evapotranspiration for each month; calculated with the Penman-Monteith equation.'}
    ds_year = ds.mean("month")
    ds_year['pr'] = ds[["pr"]].sum("month").pr
    ds_year['pet'] = ds[["pet"]].sum("month").pet
    ds_year['pet_pr'] = ds_year.pet / ds_year.pr
    ds_year['tmf'] = (ds_year.tas >= tmin).astype(int)*(ds_year.pr >= prmin).astype(int)*(ds_year.pet_pr <= pet_pr_max).astype(int)
    ds_year = ds_year.where(np.invert(np.isnan(ds_year.tas)), drop = True)
    return(ds_year[['tas', 'pr', 'pet', 'pet_pr', 'tmf']])

# code
area_shp = gp.read_file(area_file)
rsds = xr.open_dataset(rsds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
present = xr.open_dataset(present_file).rio.clip(area_shp.geometry.values, area_shp.crs)
future = xr.open_dataset(future_file).rio.clip(area_shp.geometry.values, area_shp.crs)
present = prep_tmf(present, rsds)
future = prep_tmf(future, rsds)
anomalies = future
anomalies['tmf'] = (future - present).tmf
anomalies.to_netcdf(out_file)

# view
# import matplotlib.pyplot as plt
# anomalies.tmf.plot()
# plt.show()

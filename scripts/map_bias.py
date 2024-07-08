# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
ds_file =  snakemake.input[0]
base_file = snakemake.input[1]
area_file =  snakemake.input[2]
out_file = snakemake.output[0]
baseline = snakemake.params.base
base_eval = snakemake.params.base_eval

# test
# ds_file = "results/downscaled/New-Caledonia_CMIP6_world_MIROC_MIROC6_ssp585_r1i1p1f1_none_none_chelsa2_monthly-means_2006-2019_1980-2005_bc.nc"
# base_file = "results/baselines/New-Caledonia_gshtd_monthly-means_1980-2005.nc"
# area_file = "results/areas/New-Caledonia.shp"

# libs
import geopandas as gp
import xarray as xr
import xesmf as xe

# code
area_shp = gp.read_file(area_file)
ds = xr.open_dataset(ds_file).rio.clip(area_shp.geometry.values, area_shp.crs)
if(base_eval == baseline):
    base = xr.open_dataset(base_file).rio.clip(area_shp.geometry.values, area_shp.crs)
if(base_eval != baseline):
    base = xr.open_dataset(base_file, decode_coords='all')
    regridder = xe.Regridder(base, ds, "bilinear")
    base = regridder(base, keep_attrs=True)
bias = ds - base
bias.to_netcdf(out_file)

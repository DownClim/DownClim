# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
proj_file =  snakemake.input.proj_future
base_hist_file =  snakemake.input.base_hist
ds = snakemake.output[0]
area = snakemake.params.area
project = snakemake.params.project
domain = snakemake.params.domain
institute = snakemake.params.institute
model = snakemake.params.model
experiment = snakemake.params.experiment
ensemble = snakemake.params.ensemble
rcm = snakemake.params.rcm
downscaling = snakemake.params.downscaling
baseline = snakemake.params.base
aggregation = snakemake.params.aggregation
period_future = snakemake.params.period_future
period_hist = snakemake.params.period_hist

# test
# proj_file="results/projections/_CMIP6_world_AWI_AWI-CM-1-1-MR_ssp126_r1i1p1f1_none_none_chelsa2_done.txt"
# base_hist_file="results/baselines/Côte-d'Ivoire_chelsa2_monthly-means_1980-2005.nc"
# area="Côte-d'Ivoire"
# project="CMIP6"
# domain="world"
# institute="AWI"
# model="AWI-CM-1-1-MR"
# experiment="ssp126"
# ensemble="r1i1p1f1"
# rcm="none"
# downscaling="none"
# baseline="chelsa2"
# aggregation="monthly-means"
# period_future="2071-2100"
# period_hist="1980-2005"
# ds="results/downscaled/Côte-d'Ivoire_CORDEX_AFR-22_CLMcom-KIT_NCC-NorESM1-M_rcp85_r1i1p1_CCLM5-0-15_v1_chelsa2_monthly-means_2071-2100_1980-2005_bc.nc"

# libs
import os
import xarray as xr

# open
proj_future_file = os.path.dirname(proj_file) + "/" + area + "_" + project + "_" + domain + "_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_" + rcm + "_" + downscaling + "_" + baseline + "_" + aggregation + "_" + period_future + ".nc"
proj_hist_file = os.path.dirname(proj_file) + "/" + area + "_" + project + "_" + domain + "_" + institute + "_" + model + "_" + experiment + "_" + ensemble + "_" + rcm + "_" + downscaling + "_" + baseline + "_" + aggregation + "_" + period_hist + ".nc"
proj_future = xr.open_mfdataset(proj_future_file, parallel=True)
proj_hist = xr.open_mfdataset(proj_hist_file, parallel=True)
base_hist = xr.open_mfdataset(base_hist_file, parallel=True)

# anomalies
anomalies = proj_future - proj_hist
if 'pr' in list(proj_future.keys()):
  anomalies_rel =  (proj_future - proj_hist)/(proj_hist+1)
  anomalies["pr"] = anomalies_rel["pr"]

# add to the baseline
proj_ds = base_hist + anomalies
if 'pr' in list(proj_future.keys()):
  proj_ds2 = base_hist * (1 + anomalies)
  proj_ds["pr"] = proj_ds2["pr"]

# prep and write
proj_ds = proj_ds.assign_attrs(proj_future.attrs | anomalies.attrs)
proj_ds.attrs['downscaling'] = "Downscaled with DowClim v0.1.0"
proj_ds.attrs['downscaling_method'] = "Bias correction"
proj_ds.attrs['downscaling_baseline'] = baseline
proj_ds.to_netcdf(ds)

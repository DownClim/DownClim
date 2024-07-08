# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
area_files = snakemake.input
areas = snakemake.params.area
nc_files = snakemake.output

# test
# area_files = ["results/areas/Côte-d'Ivoire.shp", 
#               "results/areas/French-Guiana.shp", 
#               "results/areas/New-Caledonia.shp"]
# areas=["Côte-d'Ivoire", "French-Guiana", "New-Caledonia"]
# temp_fold = "results/tmf/rsds/chelsa2_tmp"
# nc_files = ["results/tmf/rsds/Côte-d'Ivoire.nc",
#             "results/tmf/rsds/French-Guiana.nc",
#             "results/tmf/rsds/New-Caledonia.nc"]
        
# libs
import xarray as xr
import rioxarray as rio
import pandas as pd
import geopandas
import datetime as dt
          
# code
areas = list(map(geopandas.read_file, area_files))
areas_names = list(map(lambda x: x.NAME_0.values[0], areas))
areas_bounds = list(map(lambda x: x.bounds, areas))

a = {key: list() for key in areas_names}
for month in range(1, 13):
        # https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies/1981-2010/rsds/CHELSA_rsds_1981-2010_01_V.2.1.tif
        url = 'https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies/1981-2010/rsds/CHELSA_rsds_1981-2010_' + '%02d' % (month,) + "_V.2.1.tif"
        ds = rio.open_rasterio(url, decode_coords="all").to_dataset('band').rename_vars({1 : "rsds"})
        for i, _ in enumerate(areas): 
                a[areas_names[i]].append(ds.rio.clip_box(minx=areas_bounds[i].minx[0], 
                                                        miny=areas_bounds[i].miny[0],
                                                        maxx=areas_bounds[i].maxx[0], 
                                                        maxy=areas_bounds[i].maxy[0]))

for area_name in areas_names: 
        ds = xr.concat(a[area_name], pd.Index(pd.date_range(dt.datetime(2010, 1, 1), periods=12, freq="M"), name="time"))
        ds = ds[["time", "x", "y", "rsds"]]
        ds['rsds'] = ds.rsds * 0.001 * ds.time.dt.days_in_month 
        ds.rsds.attrs = {'standard_name': 'potential evapotranspiration', 
                        'long_name': 'Monthly potential evapotranspiration',
                        'units': 'MJ m-2 month-1', 
                        'explanation' : 'Potential evapotranspiration for each month; calculated with the Penman-Monteith equation.'}
        ds = ds.groupby("time.month").mean("time")
        path = "results/tmf/rsds/" + area_name + ".nc"
        ds.to_netcdf(path)

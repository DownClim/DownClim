# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
in_files =  snakemake.input
area = snakemake.params.area
origin = snakemake.params.origin
base_eval = snakemake.params.base_eval

# test
# import os
# import re
# in_files =  ["results/bias/" + f for f in os.listdir("results/bias/") if re.match(r'.*\.(nc)', f)]
# origin = ["CORDEX", "CMIP6"]
# base_eval = ["chelsa2", "gshtd", "chirps"]
# area = ["New-Caledonia", "Côte-d'Ivoire", "French-Guiana"]

# libs
import re
import xarray as xr

# code
for o in origin:
    files_o = [f for f in in_files if re.search('_' + o + '_', f)]
    for b in base_eval:
        files_b = [f for f in files_o if re.search('_' + b + '.nc', f)]
        for a in area:
            files_a = [f for f in files_b if re.search('/' + a + '_', f)]
            ds_all = list(map(lambda f: xr.open_dataset(f).expand_dims(file=[f]), files_a))
            ds = xr.combine_by_coords(ds_all).mean("file")
            path = "results/evaluation/bias/" + a + "_" + o + "_" + b + ".nc"
            ds.to_netcdf(path)

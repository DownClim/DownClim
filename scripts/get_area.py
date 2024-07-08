# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
area = snakemake.params.area
area_file = snakemake.output[0]

# libs
import pygadm
import re

# code
area2 = re.sub("-", " ", area)
code = pygadm.AdmNames(area2).GID_0[0]
gdf = pygadm.AdmItems(admin = code)
gdf.NAME_0 = area
gdf.to_file(area_file)

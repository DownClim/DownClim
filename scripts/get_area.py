# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
area = snakemake.params.area
log10_eval_pts = snakemake.params.log10_eval_pts
area_file = snakemake.output[0]
area_fig = snakemake.output[1]
pts_file = snakemake.output[2]
pts_fig = snakemake.output[3]

# test
# area = "New-Caledonia"
# log10_eval_pts = 4

# libs
import pygadm
import matplotlib.pyplot as plt
import re

# code
area = re.sub("-", " ", area)
code = pygadm.AdmNames(area).GID_0[0]
gdf = pygadm.AdmItems(admin = code)
pts = gdf.sample_points(pow(10, log10_eval_pts))

# country
gdf.plot()
plt.savefig(area_fig)
gdf.to_file(area_file)

# points
pts.plot()
plt.savefig(pts_fig)
pts.to_file(pts_file)

# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
proj =  snakemake.input.proj
base =  snakemake.input.base
out_file = snakemake.output[0]

# libs
import pandas as pd 

# code
proj_all = pd.concat(list(map(lambda t: pd.read_csv(t, sep="\t"), proj)))
base_all = pd.concat(list(map(lambda t: pd.read_csv(t, sep="\t"), base)))
all = pd.concat([proj_all, base_all])
all = all[all["count"] > 0]
all.to_csv(out_file, sep="\t", index=False)

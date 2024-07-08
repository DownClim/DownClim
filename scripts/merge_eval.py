# log
log_file = open(snakemake.log[0],"w")
sys.stderr = sys.stdout = log_file

# variables
in_files =  snakemake.input
out_file = snakemake.output[0]

# libs
import pandas as pd 

# code
pd.concat(list(map(lambda t: pd.read_csv(t, sep="\t"), in_files))).to_csv(out_file, sep="\t", index=False)

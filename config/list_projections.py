# libs
from pyesgf.search import SearchConnection
import gcsfs
import pandas as pd

# cordex
project = "CORDEX"
activity = "none"
domains = ["SAM-22", "AFR-22", "AUS-22"]
experiments = ["rcp26", "rcp85"]
variables = ["tas", "tasmin", "tasmax", "pr"]
time_frequency = "mon"
server = 'https://esgf-node.ipsl.upmc.fr/esg-search/'
conn = SearchConnection(server, distrib=True)
ctx = conn.new_context(
   facets='*',
   project = project,
   domain = domains,
   experiment = experiments,
   time_frequency = time_frequency,
   variable = variables
  )  
results = ctx.search()
datasets = [res.dataset_id for res in results]
df = pd.DataFrame(dict(dataset=datasets))
df[["dataset", "datanode"]]  = df['dataset'].str.split('|', expand=True)
df[['project', 'product', 'domain', 'institute', 'model', 'experiment',
    'ensemble', 'rcm', 'downscaling', 'time_frequency', 'variable', 'version']] = df['dataset'].str.split('.', expand=True)
df.project = df.project.str.upper()
cordex = df[['project', 'domain', 'institute', 'model', 'experiment', 'ensemble', 'rcm', 'downscaling']].drop_duplicates()

# cmip6
## list ScenarioMIP
df_ta = df.query("activity_id == 'ScenarioMIP' & table_id == 'Amon' & variable_id == @variables & experiment_id == @experiments & member_id == 'r1i1p1f1'")
## chek vars in ScenarioMIP
df_ta["sim"] = df_ta["institution_id"] + "_" + df_ta["source_id"]
out = df_ta.groupby('sim')['variable_id'].apply(lambda x: '-'.join(sorted(pd.Series(x).drop_duplicates().tolist())) == '-'.join(sorted(variables)))
out = out.index[out].tolist()
df_ta2 = df_ta.query("sim == @out")
## list historical
df_ta_hist = df.query("activity_id == 'CMIP' & table_id == 'Amon' & variable_id == @variables & experiment_id == 'historical' & member_id == 'r1i1p1f1'")
## chek vars in historical
df_ta_hist["sim"] = df_ta_hist["institution_id"] + "_" + df_ta_hist["source_id"]
out = df_ta_hist.groupby('sim')['variable_id'].apply(lambda x: '-'.join(sorted(pd.Series(x).drop_duplicates().tolist())) == '-'.join(sorted(variables)))
out = out.index[out].tolist()
df_ta_hist2 = df_ta_hist.query("sim == @out")
hist_projs = df_ta_hist2[['sim']].drop_duplicates().sim.to_list()
## chek ScenarioMIP with historical
df_ta3 = df_ta2.query("sim == @hist_projs")
# prepare table
df_ta3 = df_ta3.rename(columns = {"institution_id": "institute", "source_id": "model", "experiment_id": "experiment", "member_id": "ensemble"})
df_ta3.insert(0, "project", "CMIP6")
df_ta3.insert(0, "domain", "world")
df_ta3.insert(0, "rcm", "none")
df_ta3.insert(0, "downscaling", "none")
cmip6 = df_ta3[['project', 'domain', 'institute', 'model', 'experiment', 'ensemble', 'rcm', 'downscaling']].drop_duplicates().groupby(["institute", "model", 'experiment',]).head(1)

# save
pd.concat([cordex, cmip6]).to_csv("config/projections_all.tsv", sep="\t", index=False)

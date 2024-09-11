from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .getters.connectors import connect_to_esgf, esgf_url


# cordex
def get_cordex_projections(
    domains: Iterable[str] = ["SAM-22", "AFR-22", "AUS-22"],
    experiments: Iterable[str] = ["rcp26", "rcp85"],
    variables: Iterable[str] = ["tas", "tasmin", "tasmax", "pr"],
    time_frequency: str = "mon",
) -> pd.DataFrame:
    ctx = connect_to_esgf(
        {
            "project": "CORDEX",
            "domain": domains,
            "driving_model": "*",
            "experiment": experiments,
            "time_frequency": time_frequency,
            "variable": variables,
        },
        esgf_credential="config/credentials_esgf.yml",
        server=esgf_url,
    )
    datasets = [res.dataset_id for res in ctx.search()]
    df_dataset = pd.DataFrame({"dataset": datasets})
    df_dataset[["dataset", "datanode"]] = df_dataset["dataset"].str.split(
        "|", expand=True
    )
    df_dataset[
        [
            "project",
            "product",
            "domain",
            "institute",
            "model",
            "experiment",
            "ensemble",
            "rcm",
            "downscaling",
            "time_frequency",
            "variable",
            "version",
        ]
    ] = df_dataset["dataset"].str.split(".", expand=True)
    df_dataset.project = df_dataset.project.str.upper()
    return df_dataset.drop_duplicates()


def get_cmip6_projections(
    df_dataset: pd.DataFrame,
    vars: Iterable[str] = ["tas", "tasmin", "tasmax", "pr"],
):
    # cmip6
    ## list ScenarioMIP
    df_ta = df_dataset.query(
        ###f"""activity_id == '{activity}' &
        ###table_id == '{table}' &
        ###variable_id == @{vars} &
        ###experiment_id == @{experiments} &
        ###member_id == '{members}'"""
    )
    ## check vars in ScenarioMIP
    df_ta["sim"] = df_ta["institution_id"] + "_" + df_ta["source_id"]
    out = df_ta.groupby("sim")["variable_id"].apply(
        lambda x: "-".join(sorted(pd.Series(x).drop_duplicates().tolist()))
        == "-".join(sorted(vars))
    )
    out = out.index[out].tolist()
    df_ta2 = df_ta.query("sim == @out")
    ## list historical
    df_ta_hist = df_dataset.query(
        "activity_id == 'CMIP' & table_id == 'Amon' & variable_id == @variables & experiment_id == 'historical' & member_id == 'r1i1p1f1'"
    )
    ## check vars in historical
    df_ta_hist["sim"] = df_ta_hist["institution_id"] + "_" + df_ta_hist["source_id"]
    out = df_ta_hist.groupby("sim")["variable_id"].apply(
        lambda x: "-".join(sorted(pd.Series(x).drop_duplicates().tolist()))
        == "-".join(sorted(vars))
    )
    out = out.index[out].tolist()
    ## check ScenarioMIP with historical
    df_ta3 = df_ta2.query("sim == @hist_projs")
    # prepare table
    df_ta3 = df_ta3.rename(
        columns={
            "institution_id": "institute",
            "source_id": "model",
            "experiment_id": "experiment",
            "member_id": "ensemble",
        }
    )
    df_ta3.insert(0, "project", "CMIP6")
    df_ta3.insert(0, "domain", "world")
    df_ta3.insert(0, "rcm", "none")
    df_ta3.insert(0, "downscaling", "none")
    return (
        df_ta3[
            [
                "project",
                "domain",
                "institute",
                "model",
                "experiment",
                "ensemble",
                "rcm",
                "downscaling",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "institute",
                "model",
                "experiment",
            ]
        )
        .head(1)
    )


# save
def save_projections(
    cordex_projections: pd.DataFrame,
    cmip6_projections: pd.DataFrame,
    output_file: str = "resources/projections_all.tsv",
):
    pd.concat([cordex_projections, cmip6_projections]).to_csv(
        output_file, sep="\t", index=False
    )

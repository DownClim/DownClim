from __future__ import annotations

import pandas as pd


# CORDEX
# ------
# ---------------
# save available simulations
def save_projections(
    cordex_simulations: pd.DataFrame | None = None,
    cmip6_simulations: pd.DataFrame | None = None,
    output_file: str = "resources/projections_all.csv",
) -> None:
    """
    Save the lists of available CORDEX and CMIP6 simulations to a CSV file.

    Parameters
    ----------
    cordex_simulations: pd.DataFrame
        DataFrame containing information about the available CORDEX simulations.
    cmip6_simulations: pd.DataFrame
        DataFrame containing information about the available CMIP6 simulations.
    output_file: str (default: "resources/projections_all.csv")
        Path to the output file.
    """

    pd.concat([cordex_simulations, cmip6_simulations]).to_csv(
        output_file, sep=",", index=False
    )

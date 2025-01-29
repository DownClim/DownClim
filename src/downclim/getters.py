from __future__ import annotations

from .dataset.chelsa2 import get_chelsa2
from .dataset.chirps import get_chirps
from .dataset.cmip6 import get_cmip6
from .dataset.cordex import get_cordex
from .dataset.gshtd import get_gshtd
from .dataset.utils import DataProduct
from .downclim import DownClimContext


def get_baseline(context: DownClimContext) -> None:
    """Get the baseline data for the DownClim context.

    Args:
        context (DownClimContext): the DownClim context previously defined
        containing all the necessary information.

    Returns:
        None: the baseline data.
    """
    if context.baseline_product is DataProduct.CHELSA2:
        get_chelsa2(
            aoi=context.aoi,
            variable=context.variable,
            baseline_year=context.baseline_year,
            evaluation_year=context.evaluation_year,
            keep_tmp_dir=context.keep_tmp_dir,
        )
    elif context.baseline_product is DataProduct.CHIRPS:
        get_chirps(
            aoi=context.aoi,
            baseline_year=context.baseline_year,
            evaluation_year=context.evaluation_year,
            time_frequency=context.time_frequency,
            aggregation=context.downscaling_aggregation,
        )
    elif context.baseline_product is DataProduct.GSHTD:
        get_gshtd(
            aoi=context.aoi,
            variable=context.variable,
            baseline_year=context.baseline_year,
            evaluation_year=context.evaluation_year,
            time_frequency=context.time_frequency,
            aggregation=context.downscaling_aggregation,
        )
    else:
        msg = f"Unknown or not implemented data product '{context.baseline_product}'."
        raise ValueError(msg)


def get_simulations(context: DownClimContext) -> None:
    """Get the simulations data for the DownClim context.

    Depending on the product chosen, this function will download either CMIP6 or CORDEX simulation,
    or both. It will download according to the CMIP6Context or CORDEXContext defined in the DownClimContext.

    Args:
        context (DownClimContext): the DownClim context previously defined
        containing all the necessary information.

    Returns:
        None: the simulations data is downloaded.
    """
    if context.use_cmip6:
        cmip6_context = context.cmip6_context
        get_cmip6(
            aoi=context.aoi,
            variable=cmip6_context.variable_id,
            baseline_year=context.baseline_year,
            evaluation_year=context.evaluation_year,
            projection_year=context.projection_year,
            time_frequency=context.time_frequency,
            aggregation=context.downscaling_aggregation,
            activity=cmip6_context.activity_id,
            institution=cmip6_context.institution_id,
            source=cmip6_context.source_id,
            experiment=cmip6_context.experiment_id,
            member=cmip6_context.member_id,
            table=cmip6_context.table_id,
            grid_label=cmip6_context.grid_label,
            baseline=context.baseline_product,
        )
    elif context.use_cordex:
        get_cordex(
            aois=context.aoi,
            variables=context.variable,
            periods=context.projection_year,
            institute=context.institute,
            model=context.model,
            experiment=context.experiment,
            ensemble=context.ensemble,
            baseline=context.baseline_product,
        )
    else:
        msg = (
            f"Unknown or not implemented data product '{context.simulations_product}'."
        )
        raise ValueError(msg)

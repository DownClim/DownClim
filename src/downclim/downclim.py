from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import geopandas as gpd
from pydantic import BaseModel, Field, InstanceOf, field_validator, model_validator

from .downscale import DownscaleMethod
from .getters.get_aoi import get_aoi
from .getters.utils import Aggregation, DataProduct, Frequency
from .list_projections import CMIP6Context, CORDEXContext


def to_list(v: Any) -> list:
    if not isinstance(v, list):
        return [v]
    return v


class DownclimContext(BaseModel):
    """Class to define the general context for the Downclim package.
    This includes all the parameters needed to run the downscaling process.
    """

    aois: InstanceOf[
        str
        | tuple[float, float, float, float, str]
        | gpd.GeoDataFrame
        | Iterable[str]
        | Iterable[tuple[float, float, float, float, str]]
        | Iterable[gpd.GeoDataFrame]
    ] = Field(
        ..., example="Vanuatu", description="Areas of interest to downscale data."
    )
    variables: Iterable[str] = Field(
        default=["tas", "pr"],
        example=["pr", "tas", "tasmin", "tasmax"],
        description="Variables to downscale.",
    )
    time_frequency: str = Field(
        default="mon", example="mon", description="Time frequency of the data."
    )
    downscaling_aggregation: str = Field(
        default="monthly_mean",
        example="monthly_mean",
        description="Aggregation method to build the climatology.",
    )
    baseline_product: str = Field(
        default="chelsa",
        example="chelsa",
        description="Baseline product to use for downscaling.",
    )
    downscaling_method: str = Field(
        default="bias_correction",
        example="bias_correction",
        description="Downscaling method to use.",
    )
    use_cordex: bool = Field(
        default=True,
        example=True,
        description="Whether to use CORDEX data for scenarios to downscale.",
    )
    use_cmip6: bool = Field(
        default=True,
        example=True,
        description="Whether to use CMIP6 data for scenarios to downscale.",
    )
    cordex_context: CORDEXContext | None
    cmip6_context: CMIP6Context | None
    baseline_years: tuple[int, int] = Field(
        default=(1980, 2005),
        example=(1980, 2005),
        description="Interval of years to use for the baseline period.",
    )
    evaluation_years: tuple[int, int] = Field(
        default=(2006, 2019),
        example=(2006, 2019),
        description="Interval of years to use for the evaluation period.",
    )
    projection_years: tuple[int, int] = Field(
        default=(2071, 2100),
        example=(2071, 2100),
        description="Interval of years to use for the projection period.",
    )
    evaluation_products: Iterable[str] | None = Field(
        ...,
        example=["chelsa"],
        description="Products to use for the evaluation period.",
    )
    nb_threads: int = 4
    memory_mb: int = 8192
    chunks: dict[str, int] = {"time": 1, "lat": 1000, "lon": 1000}
    keep_tmp_directory: bool = False
    output_directory: str | None

    @field_validator("aois", mode="before")
    @classmethod
    def validate_aois(cls, v) -> list[gpd.GeoDataFrame]:
        return [get_aoi(aoi) for aoi in to_list(v)]

    @field_validator("projection_years")
    @classmethod
    def check_projection_years(cls, v):
        if v[0] <= 2015:
            msg = "Beginning of projection period must start in 2015 or after."
            raise ValueError(msg)
        if v[1] >= 2100:
            msg = "End of projection period must end in 2100 or earlier."
            raise ValueError(msg)
        return v

    @field_validator("time_frequency", mode="before")
    @classmethod
    def validate_time_frequency(cls, v):
        if v in ("mon", "month", "monthly", "Monthly", "MONTHLY"):
            return Frequency.MONTHLY
        msg = "Only monthly frequency is available so far. Defaulting to 'mon'."
        raise ValueError(msg)

    @field_validator("downscaling_aggregation", mode="before")
    @classmethod
    def validate_downscaling_aggregation(cls, v):
        if v in ("monthly_mean", "monthly-mean", "monthly_means", "monthly-means"):
            return Aggregation.MONTHLY_MEAN
        msg = "Only monthly means aggregation is available so far. Defaulting to 'monthly_mean'."
        raise ValueError(msg)

    @field_validator("baseline_product", mode="before")
    @classmethod
    def validate_baseline_product(cls, v):
        match v.lower():
            case "chelsa", "chelsa2":
                return DataProduct.CHELSA
            case "gshtd":
                return DataProduct.GSHTD
            case "chirps":
                return DataProduct.CHIRPS
            case _:
                msg = "Only CHELSA, CHIRPS or GSHTD can be used as a baseline product so far."
                raise ValueError(msg)

    @field_validator("downscaling_method", mode="before")
    @classmethod
    def validate_downscaling_method(cls, v):
        match v.lower():
            case "bias_correction", "bias-correction":
                return DownscaleMethod.BIAS_CORRECTION
            case "quantile_mapping", "quantile-mapping":
                return DownscaleMethod.QUANTILE_MAPPING
            case "dynamical":
                return DownscaleMethod.DYNAMICAL
            case _:
                msg = "Only 'bias_correction', 'quantile_mapping' or 'dynamical' methods are available so far."
                raise ValueError(msg)

    @field_validator("evaluation_products", mode="before")
    @classmethod
    def validate_evaluation_products(cls, v, values) -> list[DataProduct]:
        if v is None:
            msg = "No evaluation products provided. Defaulting to baseline product."
            warnings.warn(msg, stacklevel=1)
            return [values.baseline_product]
        return to_list(v)

    @model_validator(mode="before")
    @classmethod
    def check_input_coherency(cls, v):
        if v.use_cordex and v.cordex_context is None:
            msg = "Cordex context must be provided if use_cordex is True."
            raise ValueError(msg)
        if v.use_cmip6 and v.cmip6_context is None:
            msg = "CMIP6 context must be provided if use_cmip6 is True."
            raise ValueError(msg)
        return v

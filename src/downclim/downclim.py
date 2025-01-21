from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from shutil import copyfile
from typing import Any

import geopandas as gpd
import yaml
from importlib_resources import files
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .downscale import DownscaleMethod
from .getters.get_aoi import get_aoi
from .getters.utils import Aggregation, DataProduct, Frequency
from .list_projections import CMIP6Context, CORDEXContext


def to_list(v: Any) -> list[Any]:
    if not isinstance(v, list):
        return [v]
    return v


class DownclimContext(BaseModel):
    """Class to define the general context for the Downclim package.
    This includes all the parameters needed to run the downscaling process.
    """

    aoi: (
        str
        | tuple[float, float, float, float, str]
        | gpd.GeoDataFrame
        | Iterable[str]
        | Iterable[tuple[float, float, float, float, str]]
        | Iterable[gpd.GeoDataFrame]
    ) = Field(
        example=["Vanuatu"],
        description="Areas of interest to downscale data. Mandatory field. Can be a string (name of the AOI), a tuple of 5 floats (minx, miny, maxx, maxy, name), a GeoDataFrame or a list of any of these.",
    )
    variables: str | Iterable[str] = Field(
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
        default="chelsa2",
        example="chelsa2",
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
    cordex_context: CORDEXContext | dict[str, Any] | None = Field(
        default=None,
        example=CORDEXContext(domain="AUS-22", experiment="rcp85"),
        description="CORDEXContext object to use for defining the CORDEX data required.",
    )
    cmip6_context: CMIP6Context | dict[str, Any] | None = Field(
        default=None,
        example=CMIP6Context(),
        description="CMIP6Context object to use for defining the CMIP6 data required.",
    )
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
    evaluation_product: str | Iterable[str] | None = Field(
        default=None,
        example="chelsa2",
        description="Products to use for the evaluation period.",
    )
    nb_threads: int = Field(
        default=1,
        example=4,
        description="Number of threads to use for downloading and the computation (when available).",
    )
    memory_mb: int = Field(
        default=4096,
        example=8192,
        description="Memory to use for downloading and the computation.",
    )
    chunks: dict[str, int] = Field(
        default={"time": 1, "lat": 1000, "lon": 1000},
        example={"time": 1, "lat": 1000, "lon": 1000},
        description="Chunks to use for the computation.",
    )
    output_dir: str = Field(
        default="./results",
        example="/my/output_dir/path/",
        description="Output directory to save the results.",
    )
    tmp_dir: str = Field(
        default="./tmp",
        example="/my/tmp_dir/path/",
        description="Temporary directory when downloading and processing the data.",
    )
    keep_tmp_dir: bool = Field(
        default=False,
        description="Whether to keep the temporary directory after the process.",
    )
    esgf_credentials: dict[str, str] | str | None = Field(
        default=None,
        example={"username": "my_user", "password": "my_password"},
        description="""ESGF credentials to use for downloading the data.
            Can be either a dictionary with 'username' and 'password' keys,
            or a path to a yaml file containing these credentials.""",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_data_product(cls, v: str) -> DataProduct:
        match v:
            case "chelsa" | "chelsa2":
                return DataProduct.CHELSA2
            case "gshtd":
                return DataProduct.GSHTD
            case "chirps":
                return DataProduct.CHIRPS
            case _:
                msg = "Only CHELSA, CHIRPS or GSHTD can be used as a baseline product so far."
                raise ValueError(msg)

    @field_validator("aoi", mode="after")
    @classmethod
    def validate_aoi(
        cls,
        v: str
        | tuple[float, float, float, float, str]
        | gpd.GeoDataFrame
        | Iterable[str]
        | Iterable[tuple[float, float, float, float, str]]
        | Iterable[gpd.GeoDataFrame],
    ) -> list[gpd.GeoDataFrame]:
        return [get_aoi(aoi) for aoi in to_list(v)]

    @field_validator("projection_years", mode="after")
    @classmethod
    def check_projection_years(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 2015:
            msg = """Beginning of projection period must start in 2015 or after.
            This corresponds to the first year of the CMIP6 scenarios."""
            raise ValueError(msg)
        if v[1] > 2100:
            msg = """End of projection period must end in 2100 or earlier.
            This corresponds to the last year of the CMIP6 / CORDEX scenarios."""
            raise ValueError(msg)
        return v

    @field_validator("time_frequency", mode="after")
    @classmethod
    def validate_time_frequency(cls, v: str) -> Frequency:
        if v.lower() in ("mon", "month", "monthly"):
            return Frequency.MONTHLY
        msg = "Only monthly frequency is available so far. Defaulting to 'mon'."
        raise ValueError(msg)

    @field_validator("downscaling_aggregation", mode="after")
    @classmethod
    def validate_downscaling_aggregation(cls, v: str) -> Aggregation:
        if v.lower() in (
            "monthly_mean",
            "monthly-mean",
            "monthly_means",
            "monthly-means",
        ):
            return Aggregation.MONTHLY_MEAN
        msg = "Only monthly means aggregation is available so far. Defaulting to 'monthly_mean'."
        raise ValueError(msg)

    @field_validator("baseline_product", mode="after")
    @classmethod
    def validate_baseline_product(cls, v: str) -> DataProduct:
        return cls.get_data_product(v.lower())

    @field_validator("downscaling_method", mode="after")
    @classmethod
    def validate_downscaling_method(cls, v: str) -> DownscaleMethod:
        match v.lower():
            case "bias_correction" | "bias-correction":
                return DownscaleMethod.BIAS_CORRECTION
            case "quantile_mapping" | "quantile-mapping":
                return DownscaleMethod.QUANTILE_MAPPING
            case "dynamical":
                return DownscaleMethod.DYNAMICAL
            case _:
                msg = "Only 'bias_correction', 'quantile_mapping' or 'dynamical' methods are available so far."
                raise ValueError(msg)

    @field_validator("output_dir", "tmp_dir", mode="after")
    @classmethod
    def validate_dir(cls, v: str) -> str:
        if not Path(v).exists():
            Path(v).mkdir(parents=True)
            msg = f"Directory {v} did not exist and was created."
            warnings.warn(msg, stacklevel=1)
        return v

    @field_validator("esgf_credentials", mode="before")
    @classmethod
    def validate_esgf_credentials(
        cls, v: dict[str, str] | str | None, info: ValidationInfo
    ) -> dict[str, str] | None:
        if v is None and info.data["use_cordex"]:
            msg = "'use_cordex' is True but no ESGF credentials provided. Please provide them to access the data."
            raise ValueError(msg)
        if isinstance(v, str):
            with Path(v).open(encoding="utf-8") as f:
                v = yaml.safe_load(f)
            if not all(key in v for key in ("username", "password")):
                msg = """ESGF credentials must have 'username' and 'password' keys.
                    Please check your yaml file."""
                raise ValueError(msg)
        if isinstance(v, dict) and not all(
            key in v for key in ("username", "password")
        ):
            msg = "ESGF credentials must have 'username' and 'password' keys."
            raise ValueError(msg)
        return v

    @field_validator("evaluation_product", mode="after")
    @classmethod
    def validate_evaluation_product(
        cls, v: str | Iterable[str] | None, info: ValidationInfo
    ) -> DataProduct | list[DataProduct]:
        if v is None:
            msg = "No evaluation products provided. Defaulting to the same product as baseline product."
            warnings.warn(msg, stacklevel=1)
            return [info.data["baseline_product"]]
        if isinstance(v, str):
            return [cls.get_data_product(v.lower())]
        if isinstance(v, Iterable):
            return [cls.get_data_product(p.lower()) for p in v]
        msg = "Evaluation products must be a string or a list of strings match the DataProducts available."
        raise ValueError(msg)

    @field_validator("output_dir", "tmp_dir", mode="after")
    @classmethod
    def validate_dir_aoi(cls, v: str, info: ValidationInfo) -> str:
        for aoi in info.data["aoi"]:
            aoi_name = aoi["NAME_0"].to_numpy()[0]
            aoi_path = Path(v).joinpath(aoi_name)
            if aoi_path.exists():
                msg = f"""Directory {aoi_path} already exists.
                Please act carefully as you may overwrite existing files."""
                warnings.warn(msg, stacklevel=1)
        return v

    @field_validator("cordex_context", mode="after")
    @classmethod
    def validate_cordex_context(
        cls, v: CORDEXContext | dict[str, Any] | None, info: ValidationInfo
    ) -> CORDEXContext | None:
        if info.data["use_cordex"]:
            if v is None:
                msg = "Cordex context must be provided if use_cordex is True."
                raise ValueError(msg)
            if isinstance(v, dict):
                v = CORDEXContext.model_validate(v)
        return v

    @field_validator("cmip6_context", mode="after")
    @classmethod
    def validate_cmip6_context(
        cls, v: CMIP6Context | dict[str, Any] | None, info: ValidationInfo
    ) -> CMIP6Context | None:
        if info.data["use_cmip6"]:
            if v is None:
                msg = "'use_cmip6' is set to True, however no CMIP6Context is provided. Using default CMIP6 context."
                warnings.warn(msg, stacklevel=1)
                v = CMIP6Context()
            elif isinstance(v, dict):
                v = CMIP6Context.model_validate(v)
        return v

    @field_validator("use_cmip6", mode="after")
    @classmethod
    def check_cmip6_coherency(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data["cmip6_context"] is None:
            msg = "CMIP6 context must be provided if use_cmip6 is True."
            raise ValueError(msg)
        return v


def generate_DownClimContext_template_file(output_file: str) -> None:
    """Generates a template file for DownClimContext.
    This is a pre-filled .yaml file with all the parameters available.

    Args:
        output_file (str): File to save the template.
    """
    copyfile(
        files("downclim.data").joinpath("DownClimContext_template.yaml"), output_file
    )


def define_DownClimContext_from_file(file: str) -> DownclimContext:
    """Reads a DownClimContext from a file.

    Args:
        file (str): File to read the context from.

    Returns:
        DownclimContext: The context read from the file.

    Raises:
        ValueError: if YAML has no or wrong 'aoi' mandatory value
        FileNotFoundError: if the file does not exist.
    """
    # Read YAML file
    try:
        with Path(file).open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        msg = f"File {file} does not exist."
        raise FileNotFoundError(msg) from e

    if not config.get("aoi"):
        msg = "Mandatory field 'aoi' is not present in the yaml file."
        raise ValueError(msg)

    keys_years = ["baseline_years", "evaluation_years", "projection_years"]
    for key in keys_years:
        if key in config:
            config[key] = tuple(
                int(year) for year in config[key].strip("()").split(",")
            )

    return DownclimContext.model_validate(config)

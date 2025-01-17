from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from shutil import copyfile
from typing import Any

import geopandas as gpd
import yaml
from importlib_resources import files
from pydantic import BaseModel, Field, InstanceOf, field_validator, model_validator

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

    aoi: InstanceOf[
        str
        | tuple[float, float, float, float, str]
        | gpd.GeoDataFrame
        | Iterable[str]
        | Iterable[tuple[float, float, float, float, str]]
        | Iterable[gpd.GeoDataFrame]
    ] = Field(example=["Vanuatu"], description="Areas of interest to downscale data.")
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
    cordex_context: CORDEXContext | None = Field(
        default=None,
        example=CORDEXContext(domain="AUS-22", experiment="rcp85"),
        description="CORDEXContext object to use for defining the CORDEX data required.",
    )
    cmip6_context: CMIP6Context | None = Field(
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
    evaluation_product: str | Iterable[str] = Field(
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

    @field_validator("aoi", mode="before")
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

    @field_validator("projection_years")
    @classmethod
    def check_projection_years(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 2015:
            msg = """Beginning of projection period must start in 2015 or after.
            This corresponds to the first year of the CMIP6 scenarios."""
            raise ValueError(msg)
        if v[1] >= 2100:
            msg = """End of projection period must end in 2100 or earlier.
            This corresponds to the last year of the CMIP6 / CORDEX scenarios."""
            raise ValueError(msg)
        return v

    @field_validator("time_frequency", mode="before")
    @classmethod
    def validate_time_frequency(cls, v: str) -> Frequency:
        if v in ("mon", "month", "monthly", "Monthly", "MONTHLY"):
            return Frequency.MONTHLY
        msg = "Only monthly frequency is available so far. Defaulting to 'mon'."
        raise ValueError(msg)

    @field_validator("downscaling_aggregation", mode="before")
    @classmethod
    def validate_downscaling_aggregation(cls, v: str) -> Aggregation:
        if v in ("monthly_mean", "monthly-mean", "monthly_means", "monthly-means"):
            return Aggregation.MONTHLY_MEAN
        msg = "Only monthly means aggregation is available so far. Defaulting to 'monthly_mean'."
        raise ValueError(msg)

    @field_validator("baseline_product", mode="before")
    @classmethod
    def validate_baseline_product(cls, v: str) -> DataProduct:
        match v.lower():
            case "chelsa", "chelsa2":
                return DataProduct.CHELSA2
            case "gshtd":
                return DataProduct.GSHTD
            case "chirps":
                return DataProduct.CHIRPS
            case _:
                msg = "Only CHELSA, CHIRPS or GSHTD can be used as a baseline product so far."
                raise ValueError(msg)

    @field_validator("evaluation_product", mode="before")
    @classmethod
    def validate_evaluation_product(cls, self, v: str) -> list[DataProduct]:
        if v is None:
            msg = "No evaluation products provided. Defaulting to the same product as baseline product."
            warnings.warn(msg, stacklevel=1)
            return [self.baseline_product]
        return to_list(v)

    @field_validator("downscaling_method", mode="before")
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

    @field_validator("output_dir", "tmp_dir", mode="before")
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
        cls, v: dict[str, str] | str | None
    ) -> dict[str, str] | str | None:
        if v is None:
            return None
        if isinstance(v, str):
            with Path(v).open() as f:
                v = yaml.safe_load(f)
        if not all(key in v for key in ("username", "password")):
            msg = "ESGF credentials must have 'username' and 'password' keys."
            raise ValueError(msg)
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_dir_aoi(cls, v: dict[str, Any]) -> dict[str, Any]:
        for aoi in v["aoi"]:
            aoi_name = aoi.get("NAME_0").to_numpy([0])
            aoi_out_path = Path(v["output_dir"]).joinpath(aoi_name)
            aoi_tmp_path = Path(v["tmp_dir"]).joinpath(aoi_name)
            for aoi_path in (aoi_out_path, aoi_tmp_path):
                if aoi_path.exists():
                    msg = f"""Directory {aoi_path} already exists.
                    Please act carefully as you may overwrite existing files."""
                    warnings.warn(msg, stacklevel=1)
        return v

    @model_validator(mode="before")
    @classmethod
    def check_input_coherency(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v["use_cordex"] and v["cordex_context"] is None:
            msg = "Cordex context must be provided if use_cordex is True."
            raise ValueError(msg)
        if v["use_cmip6"] and v["cmip6_context"] is None:
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
        files("./downclim/data").joinpath("DownClimContext_template.yml"), output_file
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
    with Path(file).open() as f:
        config = yaml.safe_load(f)

    if not config.get("aoi"):
        msg = "Mandatory field 'aoi' is not present in the yaml file."
        raise ValueError(msg)

    if Path(file).exists():
        msg = f"File {file} does not exist."
        raise FileNotFoundError(msg)

    return DownclimContext(
        aoi=config["aoi"],
        variables=config.get("variables"),
        time_frequency=config.get("time_frequency"),
        downscaling_aggregation=config.get("downscaling_aggregation"),
        downscaling_method=config.get("downscaling_method"),
        baseline_product=config.get("baseline_product"),
        use_cordex=config.get("use_cordex"),
        use_cmip6=config.get("use_cmip6"),
        cordex_context=config.get("cordex_context"),
        cmip6_context=config.get("cmip6_context"),
        baseline_years=config.get("baseline_years"),
        evaluation_years=config.get("evaluation_years"),
        projection_years=config.get("projection_years"),
        evaluation_product=config.get("evaluation_product"),
        nb_threads=config.get("nb_threads"),
        memory_mb=config.get("memory_mb"),
        chunks=config.get("chunks"),
        output_dir=config.get("output_dir"),
        tmp_dir=config.get("tmp_dir"),
        keep_tmp_dir=config.get("keep_tmp_dir"),
        esgf_credentials=config.get("esgf_credentials"),
    )

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from shutil import copyfile
from typing import Any

import geopandas as gpd
import yaml
from importlib_resources import files
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from .aoi import get_aoi
from .dataset.chelsa2 import get_chelsa2
from .dataset.chirps import get_chirps
from .dataset.cmip6 import CMIP6Context, get_cmip6
from .dataset.cordex import CORDEXContext, get_cordex, inspect_cordex
from .dataset.gshtd import get_gshtd
from .dataset.utils import Aggregation, DataProduct, Frequency
from .downscale import DownscaleMethod, run_downscaling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class DownClimContext(BaseModel):
    """Class to define the general context for the Downclim package.
    This includes all the parameters needed to run the downscaling process.
    """

    aoi: list[gpd.GeoDataFrame] = Field(
        examples=["Vanuatu"],
        description="""Areas of interest to downscale data. Mandatory field.
            Can be :
            - a string (name of the AOI),
            - a tuple of 5 floats (minx, miny, maxx, maxy, name),
            - a GeoDataFrame
            - or a list of any of these.""",
    )
    variable: list[str] = Field(
        default=["tas", "pr"],
        examples=["pr", ("tas", "tasmin", "tasmax")],
        description="Variables to downscale.",
    )
    time_frequency: Frequency = Field(
        default=Frequency.MONTHLY,
        examples=[Frequency.MONTHLY, "monthly"], # type: ignore[assignment]
        description="Time frequency of the data."
    )
    downscaling_aggregation: Aggregation = Field(
        default=Aggregation.MONTHLY_MEAN,
        examples=[Aggregation.MONTHLY_MEAN,"monthly_mean"], # type: ignore[assignment]
        description="Aggregation method to build the climatology.",
    )
    baseline_product: DataProduct = Field(
        default=DataProduct.CHELSA,
        examples=[DataProduct.CHELSA, "chelsa2", "chirps", "gshtd"],
        description="Baseline product to use for downscaling.",
    )
    evaluation_product: list[DataProduct] = Field(
        default=[DataProduct.CHELSA],
        examples=["chelsa2", (DataProduct.CHIRPS, "gshtd")],
        description="Products to use for the evaluation period.",
    )
    downscaling_method: DownscaleMethod = Field(
        default=DownscaleMethod.BIAS_CORRECTION,
        examples=[DownscaleMethod.BIAS_CORRECTION, "bias_correction"],
        description="Downscaling method to use.",
    )
    use_cordex: bool = Field(
        default=True,
        examples=[True, False],
        description="Whether to use CORDEX data for scenarios to downscale.",
    )
    use_cmip6: bool = Field(
        default=True,
        examples=[True, False],
        description="Whether to use CMIP6 data for scenarios to downscale.",
    )
    cordex_context: CORDEXContext | None = Field(
        default=None,
        examples=[CORDEXContext(domain=["AUS-22"], experiment=["historical", "rcp85"]),
                  {"domain": ["AUS-22"], "experiment": ["historical", "rcp85"]}],
        description="CORDEXContext object to use for defining the CORDEX data required.",
    )
    cmip6_context: CMIP6Context | None = Field(
        default=None,
        examples=[CMIP6Context(),
                  {"experiment": ["historical", "ssp585"]}],
        description="CMIP6Context object to use for defining the CMIP6 data required.",
    )
    historical_period: tuple[int, int] = Field(
        default=(1980, 2005),
        examples=[(1980, 2005)],
        description="Interval of years to use for the baseline period.",
    )
    evaluation_period: tuple[int, int] = Field(
        default=(2006, 2019),
        examples=[(2006, 2019)],
        description="Interval of years to use for the evaluation period.",
    )
    projection_period: tuple[int, int] = Field(
        default=(2071, 2100),
        examples=[(2071, 2100)],
        description="Interval of years to use for the projection period.",
    )
    nb_threads: int = Field(
        default=1,
        examples=[2, 4, 8],
        description="Number of threads to use for downloading and the computation (when available).",
    )
    memory_mb: int = Field(
        default=4096,
        examples=[8192],
        description="Memory to use for downloading and the computation.",
    )
    chunks: dict[str, int] = Field(
        default={"time": 1, "lat": 1000, "lon": 1000},
        examples=[{"time": 1, "lat": 1000, "lon": 1000},
                  {"time": 10, "lat": 100, "lon": 100}],
        description="Chunks to use for the computation.",
    )
    output_dir: str = Field(
        default="./results",
        examples=["/my/output_dir/path/",
                  "/another/output_dir/path/"],
        description="Output directory to save the results.",
    )
    tmp_dir: str = Field(
        default="./tmp",
        examples=["/my/tmp_dir/path/",
                  "/another/tmp_dir/path/"],
        description="Temporary directory when downloading and processing the data.",
    )
    keep_tmp_dir: bool = Field(
        default=False,
        description="Whether to keep the temporary directory after the process.",
    )
    cmip6_simulations_to_downscale: list[str] | None = Field(
        default=None,
        description="List of paths to downloaded CMIP6 simulations available locally. This list is used to determine which simulation can be downscaled.",
    )
    cordex_simulations_to_downscale: list[str] | None = Field(
        default=None,
        description="List of paths to downloaded CORDEX simulations available locally. This list is used to determine which simulation can be downscaled.",
    )
    esgf_credentials: dict[str, str] | None = Field(
        default=None,
        examples=[
            {
                "openid": "https://esgf-node.ipsl.upmc.fr/esgf-idp/openid/my_user",
                "password": "my_password",
            }
        ],
        description="""ESGF credentials to use for downloading the data.
            Can be either a dictionary with 'openid' and 'password' keys,
            or a path to a yaml file containing these 2 fields.""",
    )
    ee_project: str = Field(
        default=None,
        description="Earth Engine project ID to use for downloading the data stored in Google Earth Engine.",
    )

    class Config:
        """Pydantic configuration for the DownClimContext class."""

        arbitrary_types_allowed = True

    @classmethod
    def to_list(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return [v]
        return v

    @classmethod
    def parse_period(cls, v: str) -> tuple[int, int]:
        """Parse a period string in the format 'YYYY-YYYY'."""
        msg = """If you declare your time period as a string, it must be in the format 'YYYY-YYYY'.
                Otherwise, declare it as a list or tuple of two integers."""
        if "-" not in v:
            raise ValueError(msg)
        try:
            start_year, end_year = map(int, v.split("-"))
        except ValueError as e:
            raise ValueError(msg) from e
        return start_year, end_year

    @classmethod
    def get_data_product(cls, v: str) -> DataProduct:
        match v:
            case "chelsa" | "chelsa2":
                return DataProduct.CHELSA
            case "gshtd":
                return DataProduct.GSHTD
            case "chirps":
                return DataProduct.CHIRPS
            case _:
                msg = "Only CHELSA, CHIRPS or GSHTD can be used as a baseline product so far."
                raise ValueError(msg)

    @field_validator("aoi", mode="before")
    @classmethod
    def validate_aoi(
        cls,
        v: str | tuple[float, float, float, float, str] | gpd.GeoDataFrame | list[Any],
    ) -> list[gpd.GeoDataFrame]:
        return [get_aoi(aoi) for aoi in cls.to_list(v)]

    @field_validator("variable", mode="after")
    @classmethod
    def validate_variable(cls, v: Any | list[Any]) -> list[str]:
        if isinstance(v, str):
            return cls.to_list(v)
        return [str(var) for var in v]

    @field_validator("time_frequency", mode="before")
    @classmethod
    def validate_time_frequency(cls, v: str | Frequency) -> Frequency:
        if isinstance(v, str) and v.lower() in ["mon", "month", "monthly"]:
            return Frequency.MONTHLY  # type: ignore[return-value]
        if isinstance(v, Frequency):
            return v
        msg = "Only monthly frequency is available so far."
        raise ValueError(msg)

    @field_validator("downscaling_aggregation", mode="after")
    @classmethod
    def validate_downscaling_aggregation(cls, v: str | Aggregation) -> Aggregation:
        if isinstance(v, str) and v.lower() in [
            "monthly mean",
            "monthly_mean",
            "monthly-mean",
            "monthly_means",
            "monthly-means",
        ]:
            return Aggregation.MONTHLY_MEAN  # type: ignore[return-value]
        if isinstance(v, Aggregation):
            return v
        msg = "Only monthly means aggregation is available so far. Defaulting to 'monthly_mean'."
        raise ValueError(msg)

    @field_validator("baseline_product", mode="before")
    @classmethod
    def validate_baseline_product(cls, v: str | DataProduct) -> DataProduct:
        if isinstance(v, str):
            return cls.get_data_product(v.lower())
        if isinstance(v, DataProduct):
            return v
        msg = "Baseline product must be a string or a DataProduct."
        raise ValueError(msg)

    @field_validator("evaluation_product", mode="before")
    @classmethod
    def validate_evaluation_product(
        cls, v: str | DataProduct | Iterable[str] | Iterable[DataProduct] | None, info: ValidationInfo
    ) -> list[DataProduct]:
        if v is None:
            msg = "No evaluation products provided. Defaulting to the same product as baseline product."
            warnings.warn(msg, stacklevel=1)
            return [info.data["baseline_product"]]
        if isinstance(v, str):
            return [cls.get_data_product(v.lower())]
        if isinstance(v, DataProduct):
            return [v]
        if isinstance(v, Iterable):
            return [cls.get_data_product(p.lower()) for p in v if isinstance(p, str)]
        msg = "Evaluation products must be a string or a list of strings that match the DataProducts available."
        raise ValueError(msg)

    @field_validator("downscaling_method", mode="after")
    @classmethod
    def validate_downscaling_method(cls, v: str | DownscaleMethod) -> DownscaleMethod:
        if isinstance(v, str):
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
        if isinstance(v, DownscaleMethod):
            return v
        msg = "Downscaling method must be a string or a DownscaleMethod."
        raise ValueError(msg)

    @model_validator(mode="after")
    def check_cordex_consistency(self) -> Self:
        if self.use_cordex and self.cordex_context is None:
            msg = "cordex_context must be provided if use_cordex is True."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_cmip6_consistency(self) -> Self:
        if self.use_cmip6 and self.cmip6_context is None:
            msg = "cmip6_context must be provided if use_cmip6 is True."
            raise ValueError(msg)
        return self

    @field_validator("cordex_context", mode="after")
    @classmethod
    def validate_cordex_context(
        cls, v: CORDEXContext | dict[str, Any] | None
    ) -> CORDEXContext:
        if v is None:
            msg = """'use_cordex' is set to True, however no cordex_context is provided.
            Please correct to use a coherent context."""
            raise ValueError(msg)
        if isinstance(v, dict):
            v = CORDEXContext.model_validate(v)
        return v

    @field_validator("cmip6_context", mode="after")
    @classmethod
    def validate_cmip6_context(
        cls, v: CMIP6Context | dict[str, Any] | None
    ) -> CMIP6Context:
        if v is None:
            msg = """'use_cmip6' is set to True, however no cmip6_context is provided.
            Please correct to use a coherent context."""
            raise ValueError(msg)
        if isinstance(v, dict):
            v = CMIP6Context.model_validate(v)
        return v

    @model_validator(mode="after")
    def check_periods_consistency(self) -> Self:
        product = self.baseline_product
        if self.historical_period[0] < product.period[0] or self.historical_period[1] > product.period[1]:
            msg = f"""Baseline period must be within the period of the baseline product.
            The period available for {product.product_name} is {self.baseline_product.period}."""
            raise ValueError(msg)
        for product in self.evaluation_product:
            if self.evaluation_period[0] < product.period[0] or self.evaluation_period[1] > product.period[1]:
                msg = f"""Evaluation period must be within the period of the evaluation product.
                The period available for {product.product_name} product is {product.period}."""
                raise ValueError(msg)
        return self

    @field_validator("historical_period", "evaluation_period", "projection_period", mode="after")
    @classmethod
    def check_periods(
        cls, v: str | Iterable[int, int]
    ) -> tuple[int, int]:
        if isinstance(v, str):
            v = cls.parse_period(v)
        if not isinstance(v, Iterable) or len(v) != 2:
            msg = """All periods must be defined as an iterable (tuple, list) of 2 integers (start year, end year),
            or as a string in the format 'YYYY-YYYY'."""
            raise ValueError(msg)
        return tuple(v)

    @field_validator("projection_period", mode="after")
    @classmethod
    def check_projection_period(cls, v: Iterable[int, int]) -> tuple[int, int]:
        if v[0] <= 2015:
            msg = """Beginning of projection period must start in 2015 or after.
            This corresponds to the first year of the CMIP6 / CORDEX scenarios."""
            raise ValueError(msg)
        if v[1] > 2100:
            msg = """End of projection period must end in 2100 or earlier.
            This corresponds to the last year of the CMIP6 / CORDEX scenarios."""
            raise ValueError(msg)
        return tuple(v)

    @field_validator("output_dir", "tmp_dir", mode="after")
    @classmethod
    def validate_dir_aoi(cls, v: str, info: ValidationInfo) -> str:
        if not Path(v).exists():
            Path(v).mkdir(parents=True)
            msg = f"Directory {v} did not exist and was created."
            warnings.warn(msg, stacklevel=1)
        for aoi in info.data["aoi"]:
            aoi_name = aoi["NAME_0"].to_numpy()[0]
            aoi_path = Path(v).joinpath(aoi_name)
            if aoi_path.exists():
                msg = f"""Directory {aoi_path} already exists.
                Please act carefully as you may overwrite existing files."""
                warnings.warn(msg, stacklevel=1)
        return v

    @field_validator("esgf_credentials", mode="before")
    @classmethod
    def validate_esgf_credentials(
        cls, v: dict[str, str] | str | None, info: ValidationInfo
    ) -> dict[str, str] | None:
        mandatory_keys = ("openid", "password")
        if v is None:
            if info.data["use_cordex"]:
                msg = "'use_cordex' is True but no ESGF credentials provided. Please provide them to access the data."
                raise ValueError(msg)
            return None
        if isinstance(v, str):
            with Path(v).open(encoding="utf-8") as f:
                v = yaml.safe_load(f)
            if not all(key in v for key in mandatory_keys):
                msg = f"""ESGF credentials must have {mandatory_keys} keys.
                    Please check your yaml file."""
                raise ValueError(msg)
            return v
        if isinstance(v, dict):
            if not all(key in v for key in mandatory_keys):
                msg = f"""ESGF credentials must have {mandatory_keys} keys.
                    Please check your dictionary input."""
                raise ValueError(msg)
            return v
        msg = "ESGF credentials must be a dictionary, a path to a yaml file or None."
        raise ValueError(msg)

    @field_validator("ee_project", mode="before")
    @classmethod
    def validate_ee_project(cls, v: str | None) -> str | None:
        if v is None:
            msg = """No Earth Engine project ID provided.
            You won't be able to access Earth Engine datasets."""
            warnings.warn(msg, stacklevel=1)
        elif not isinstance(v, str):
            msg = "ee_project: Earth Engine project ID must be a string."
            raise ValueError(msg)
        return v

    def _get_baseline_product(self) -> None:
        """Get the baseline data for the DownClim context.

        This means that it downloads the baseline product for the baseline period and builds a climatology.

        Returns:
            None: the baseline data files downloaded.
        """
        logger.info("Downloading baseline product...")
        if self.baseline_product is DataProduct.CHELSA:
            get_chelsa2(
                aoi=self.aoi,
                variable=self.variable,
                period=self.historical_period,
                frequency=self.time_frequency,
                aggregation=self.downscaling_aggregation,
                nb_threads=self.nb_threads,
                output_dir=f"{self.output_dir}/{self.baseline_product.product_name}",
                tmp_dir=f"{self.tmp_dir}/{self.baseline_product.product_name}",
                keep_tmp_dir=self.keep_tmp_dir,
            )
        elif self.baseline_product is DataProduct.CHIRPS:
            get_chirps(
                aoi=self.aoi,
                period=self.historical_period,
                time_frequency=self.time_frequency,
                aggregation=self.downscaling_aggregation,
                output_dir=f"{self.output_dir}/{self.baseline_product.product_name}",
            )
        elif self.baseline_product is DataProduct.GSHTD:
            get_gshtd(
                aoi=self.aoi,
                variable=self.variable,
                period=self.historical_period,
                time_frequency=self.time_frequency,
                aggregation=self.downscaling_aggregation,
                output_dir=f"{self.output_dir}/{self.baseline_product.product_name}",
                )
        else:
            msg = f"Unknown or not implemented data product '{self.baseline_product.product_name}'."
            raise ValueError(msg)

    def _get_evaluation_product(self) -> None:
        """Get the evaluation data for the DownClim context.

        This means that it downloads the evaluation product for the evaluation period and builds a climatology.

        Returns:
            None: the evaluation data.
        """
        logger.info("Downloading evaluation product...")
        for product in self.evaluation_product:
            if product is DataProduct.CHELSA:
                get_chelsa2(
                    aoi=self.aoi,
                    variable=self.variable,
                    period=self.evaluation_period,
                    frequency=self.time_frequency,
                    aggregation=self.downscaling_aggregation,
                    nb_threads=self.nb_threads,
                    output_dir=f"{self.output_dir}/{product.product_name}",
                    tmp_dir=f"{self.tmp_dir}/{product.product_name}",
                    keep_tmp_dir=self.keep_tmp_dir,
                )
            elif product is DataProduct.CHIRPS:
                get_chirps(
                    aoi=self.aoi,
                    period=self.evaluation_period,
                    time_frequency=self.time_frequency,
                    aggregation=self.downscaling_aggregation,
                    output_dir=f"{self.output_dir}/{product.product_name}",
                    ee_project=self.ee_project
                )
            elif product is DataProduct.GSHTD:
                get_gshtd(
                    aoi=self.aoi,
                    variable=self.variable,
                    period=self.evaluation_period,
                    time_frequency=self.time_frequency,
                    aggregation=self.downscaling_aggregation,
                    output_dir=f"{self.output_dir}/{product.product_name}",
                    ee_project=self.ee_project
                )
            else:
                msg = f"Unknown or not implemented data product '{product.product_name}'."
                raise ValueError(msg)


    def _get_simulations(self) -> None:
        """Get the simulations data for the DownClim context.

        Depending on the product chosen, this function will download either CMIP6 or CORDEX simulation,
        or both. It will download according to the CMIP6Context or CORDEXContext defined in the DownClimContext.

        Parameters
        ----------
        self: DownClimContext

        Returns
        -------
        None: the simulations data is downloaded.
        """
        logger.info("Downloading simulations data...")
        if self.use_cmip6:
            logger.info("     Downloading CMIP6 simulations...")
            cmip6_simulations = self.cmip6_context.list_available_simulations()
            get_cmip6(
                aoi=self.aoi,
                cmip6_simulations=cmip6_simulations,
                historical_period=self.historical_period,
                evaluation_period=self.evaluation_period,
                projection_period=self.projection_period,
                aggregation=self.downscaling_aggregation,
                output_dir=f"{self.output_dir}/cmip6",
                chunks=self.chunks,
            )
        if self.use_cordex:
            logger.info("     Downloading CORDEX simulations...")
            cordex_simulations = inspect_cordex(
                context=self.cordex_context,
                esgf_credential=self.esgf_credentials
            )
            get_cordex(
                aoi=self.aoi,
                cordex_simulations=cordex_simulations,
                historical_period=self.historical_period,
                evaluation_period=self.evaluation_period,
                projection_period=self.projection_period,
                aggregation=self.downscaling_aggregation,
                output_dir=f"{self.output_dir}/cordex",
                tmp_dir=self.tmp_dir,
                nb_threads=self.nb_threads,
                keep_tmp_dir=self.keep_tmp_dir,
                esgf_credentials=self.esgf_credentials,
            )

    def download_data(self) -> None:
        """Downloads the data required defined in the DownClim context.

        This includes the baseline product, evaluation product and simulations data.
        It will download the data according to the DownClimContext defined.

        Returns:
            None: the data is downloaded in the output directory.
        """
        logger.info("Starting data download...")
        logger.info("   Downloading baseline product...")
        self._get_baseline_product()
        logger.info("   Downloading evaluation product...")
        self._get_evaluation_product()
        logger.info("   Downloading simulations...")
        self._get_simulations()
        logger.info("Data download complete.")


    def downscale(
        self,
        cmip6_simulations_to_downscale: list[str] | None = None,
        cordex_simulations_to_downscale: list[str] | None = None,
        reference_grid_file: str | None = None,
    ) -> None:
        """Runs the downscaling process with the current context.

        Parameters
        ----------
        self: DownClimContext
        cmip6_simulations_to_downscale: list[str] | None
            List of paths to CMIP6 simulations to downscale. If None, uses all files from
            self.output_dir/cmip6. Defaults to None.
        cordex_simulations_to_downscale: list[str] | None
            List of paths to CORDEX simulations to downscale. If None, uses all files from
            self.output_dir/cordex. Defaults to None.
        reference_grid_file: str | None
            Path to the reference grid file used to regrid the data. If None, will use the grid file from the baseline product.

        Returns
        -------
        None: the downscaling process is run and results are saved in the output directory.

        """
        run_downscaling(
            aoi=self.aoi,
            historical_period=self.historical_period,
            projection_period=self.projection_period,
            baseline_product=self.baseline_product,
            cmip6_simulations_to_downscale=cmip6_simulations_to_downscale,
            cordex_simulations_to_downscale=cordex_simulations_to_downscale,
            reference_grid_file=reference_grid_file,
            aggregation=self.downscaling_aggregation,
            method=self.downscaling_method,
            input_dir=self.output_dir,
        )


def generate_DownClimContext_template_file(output_file: str) -> None: # pylint: disable=invalid-name
    """Generates a template file for DownClimContext.
    This is a pre-filled .yaml file with all the parameters available.

    Args:
        output_file (str): File to save the template.
    """
    copyfile(
        files("downclim").parent.parent.joinpath("data", "DownClimContext_template.yaml"), output_file
    )


def define_DownClimContext_from_file(file: str) -> DownClimContext: # pylint: disable=invalid-name
    """Reads a DownClimContext from a file.

    Args:
        file (str): File to read the context from.

    Returns:
        DownClimContext: The context read from the file.

    Raises:
        ValueError: if YAML has no or wrong 'aoi' mandatory value
        FileNotFoundError: if the file does not exist.
    """
    # Read YAML file
    try:
        with Path(file).open(encoding="utf-8") as f:
            context = yaml.safe_load(f)
    except FileNotFoundError as e:
        msg = f"File {file} does not exist."
        raise FileNotFoundError(msg) from e

    if not context.get("aoi"):
        msg = "Mandatory field 'aoi' is not present in the yaml file."
        raise ValueError(msg)

    return DownClimContext.model_validate(context)

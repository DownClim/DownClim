from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import xarray as xr
import xesmf as xe
from aenum import MultiValueEnum

from ..logging_config import get_logger

logger = get_logger(__name__)

# GDAL configuration to avoid EPSG:4326 warnings
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"

TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


class Frequency(MultiValueEnum):
    """Class to define the frequency of the data handled.
    Usually 'daily | monthly | annualy'.
        So far only 'monthly' is implemented.

    Args:
        MultiValueEnum ('str'): string representation of the time frequency.

    Raises:
        ValueError: if the frequency is not implemented.
    """

    MONTHLY = "monthly", "m", "mon", "month", "Monthly", "Month", "MONTHLY", "MONTH"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented frequency '{value}'. Right now only 'monthly' is implemented."
        raise ValueError(msg)


class Aggregation(MultiValueEnum):
    """Class to define the aggregation method of the data handled.
    Usually 'mean | sum | max | min'.
    So far only 'monthly-mean' is implemented.

    Args:
        MultiValueEnum ('str'): string representation of the aggregation method.

    Raises:
        ValueError: if the aggregation method is not implemented.
    """

    MONTHLY_MEAN = "monthly-mean", "monthly_mean", "monthly_means", "monthly-means"

    @classmethod
    def _missing_(cls, value: Any) -> None:
        msg = f"Unknown or not implemented aggregation method '{value}'. Right now only 'monthly-mean' is implemented."
        raise ValueError(msg)


@dataclass
class DataProductProperties:
    """Data class to define the properties of the data products handled.

    Args:
        product_name (str): name of the data product.
        period (tuple[int, int]): period of the data product in the format (begin_year, end_year).
        variables_names (dict[str, str]): name of the variables in the data product and their correspondence as CMOR names
        scale_factor (dict[str, float]): scale factor for each variable in the data product.
        add_offset (dict[str, float]): add offset for each variable in the data product.
        lon_lat_names (dict[str, str]): longitude and latitude variable names.
        url (str): url to download the data product.
            If the data is stored on earthengine, provides the image collection path instead.

    """

    product_name: str
    period: tuple[int, int]
    variables_names: dict[str, str]
    scale_factor: dict[str, float]
    add_offset: dict[str, float]
    lon_lat_names: dict[str, str]
    url: str


class DataProduct(DataProductProperties, Enum):
    """Enum Class to define the data products handled.

    Raises:
        ValueError: if no method exists for downloading the data product.

    Returns:
        DataProduct: the data product to retrieve.
    """

    ERA5 = (
        "era5",
        (1979, 2019),
        {
            "mean_2m_air_temperature": "tas",
            "minimum_2m_air_temperature": "tasmin",
            "maximum_2m_air_temperature": "tasmax",
            "dewpoint_2m_temperature": "tdps",
            "total_precipitation": "pr",
            "surface_pressure": "ps",
            "mean_sea_level_pressure": "psl",
        },
        {
            "mean_2m_air_temperature": 1,
            "minimum_2m_air_temperature": 1,
            "maximum_2m_air_temperature": 1,
            "dewpoint_2m_temperature": 1,
            "total_precipitation": 1000 / (60 * 60 * 24),
            "surface_pressure": 1,
            "mean_sea_level_pressure": 1,
        },
        {
            "mean_2m_air_temperature": -273.15,
            "minimum_2m_air_temperature": -273.15,
            "maximum_2m_air_temperature": -273.15,
            "dewpoint_2m_temperature": -273.15,
            "total_precipitation": 0,
            "surface_pressure": 0,
            "mean_sea_level_pressure": 0,
        },
        {"lon": "lon", "lat": "lat"},
        "ECMWF/ERA5/MONTHLY",
    )

    # cmi: Climate moisture index (kg.m-2.month-1)
    # hurs : near surface relative humidity (%)
    # rsds: surface solar radiation downwards (MJ.m-2.day-1)
    # pet: potential evapotranspiration (kg.mm.month-1)
    # vpd: vaport pressure deficit (Pa)
    CHELSA = (
        "chelsa",
        (1979, 2018),
        {},
        {
            "cmi": 0.1,
            "hurs": 0.01,
            "rsds": 0.001,
            "pet": 0.01,
            "pr": 0.1,
            "tas": 0.1,
            "tasmin": 0.1,
            "tasmax": 0.1,
            "vpd": 0.1,
        },
        {
            "cmi": 0,
            "hurs": 0,
            "rsds": 0,
            "pet": 0,
            "pr": 0,
            "tas": -273.15,
            "tasmin": -273.15,
            "tasmax": -273.15,
            "vpd": 0,
        },
        {"lon": "x", "lat": "y"},
        "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL",
    )
    CMIP6 = (
        "cmip6",
        (1850, 2100),
        {},
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    )
    CORDEX = (
        "cordex",
        (1850, 2100),
        {},
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "https://esgf-node.ipsl.upmc.fr/esg-search",
    )
    GSHTD = (
        "gshtd",
        (2001, 2020),
        {"tas": "tas", "tasmin": "tasmin", "tasmax": "tasmax"},
        {"tas": 0.02, "tasmin": 0.02, "tasmax": 0.02},
        {"tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "projects/sat-io/open-datasets/GSHTD/",
    )
    CHIRPS = (
        "chirps",
        (1980, 2024),
        {"precipitation": "pr"},
        {"pr": 1},
        {"pr": 0},
        {"lon": "lon", "lat": "lat"},
        "UCSB-CHG/CHIRPS/DAILY",
    )


@dataclass
class VariableAttributesDescription:
    """Data class to define the attributes of the variables in the dataset."""

    standard_name: str
    long_name: str
    units: str
    explanation: str


class VariableAttributes(VariableAttributesDescription, Enum):
    """Enum Class to define the attributes of the variables in the dataset.

    Implementation is done var the variables handled in Downclim so far.
    """

    pr = (
        "precipitation",
        "Monthly precipitation",
        "mm month-1",
        "Precipitation in the earth's atmosphere, monthly means precipitation of water in all phases.",
    )
    tas = (
        "temperature at surface",
        "Monthly mean daily air temperature",
        "°C",
        "Monthly mean air temperatures at 2 meters.",
    )
    tasmin = (
        "minimum temperature at surface",
        "Monthly minimum daily air temperature",
        "°C",
        "Monthly minimum air temperatures at 2 meters.",
    )
    tasmax = (
        "maximum temperature at surface",
        "Monthly maximum daily air temperature",
        "°C",
        "Monthly maximum air temperatures at 2 meters.",
    )
    pet = (
        "potential evapotranspiration",
        "Monthly potential evapotranspiration",
        "mm month-1",
        "Potential evapotranspiration for each month; calculated with the Penman-Monteith equation.",
    )


def split_period(period: tuple[int, int]) -> tuple[str, str]:
    """
    Split a period string into two dates to define dmin and dmax.

    Parameters
    ----------
    period: tuple[int, int]
        Period in the format (begin_year, end_year).

    Returns
    -------
    tuple
        Tuple with two strings representing the start and end of the period.
    """
    return (f"{period[0]}-01-01", f"{period[1]}-12-31")


def sel_period(ds: xr.Dataset, tmin: str, tmax: str) -> xr.Dataset:
    """
    Select a time period on a Dataset.

    Parameters
    ----------
    ds: xr.Dataset
        ds to subsample
    tmin: str
        Time-slice initiation, obtained from "split_period", on format "YYYY-MM-DD"
    tmax: str
        Time-slice end, obtained from "split_period", on format "YYYY-MM-DD"

    Returns
    -------
    xr.Dataset
        Original dataset sliced of the time period.
    """
    try:
        return ds.sel(time=slice(tmin, tmax))
    except ValueError as e:
        # Check if the problem is the last day not existing (e.g., 360-day calendar)
        if "invalid day number" in str(e):
            logger.warning(
                "Last day of the period not available, probably due to 360 days calendar. Fixing that using a 360 calendar"
            )
            tmax_fixed = tmax[:-2] + "30"
            return ds.sel(time=slice(tmin, tmax_fixed))
        raise  # re-raise if it's some other error


def convert_cf_to_dt(t: np.datetime64) -> dt:
    return dt.strptime(str(t), "%Y-%m-%d %H:%M:%S")


def prep_dataset(ds: xr.Dataset, data_product: DataProduct) -> xr.Dataset:
    """Prepare a dataset for downscaling.

    Some operations (scaling and offsets) are applied to the dataset depending on
    the variables in the dataset.

    Parameters
    ----------
    ds: xr.Dataset original dataset to prepare.
    data_product: DataProduct
        Type of dataset to prepare.

    Returns
    -------
    xr.Dataset
        Prepared dataset with scaling and offsets applied to the variables.
    """
    cf = not isinstance(ds["time"].to_numpy()[0], np.datetime64)
    if cf:  # only cftime if not dt but should include more cases
        ds = xr.decode_cf(ds, decode_times=TIME_CODER)
    for key in ds:
        try:
            ds[key] = (
                ds[key] * data_product.scale_factor[key] + data_product.add_offset[key]
            )
            ds[key].attrs = asdict(VariableAttributes[str(key)].value)
            if data_product.variables_names:
                ds.rename({key: data_product.variables_names[key]})
        except KeyError as error:
            msg = f"Can't find scale factor and/or offset for variable {key} in dataset {data_product.product_name}."
            raise KeyError(msg) from error
    ds = ds.rename(
        {
            data_product.lon_lat_names["lon"]: "lon",
            data_product.lon_lat_names["lat"]: "lat",
        }
    )
    if (ds.lon.max().values) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.roll(lon=int(len(ds["lon"]) / 2), roll_coords=True)

    return ds.rio.write_crs("epsg:4326", inplace=True).rio.set_spatial_dims(
        x_dim="lon", y_dim="lat", inplace=True
    )


def get_monthly_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the monthly mean of a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with daily values.

    Returns
    -------
    xr.Dataset
        Dataset with average monthly means values.
    """
    return ds.resample(time="1ME").mean()


def get_monthly_climatology(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute the monthly climatology of a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with monthly values over multiple years.

    Returns
    -------
    xr.Dataset
        Dataset with average monthly means values.
    """

    return ds.groupby("time.month").mean("time")


def climatology_filename(
    outputdir: str,
    aoi_name: str,
    dataproduct: DataProduct,
    aggregation: Aggregation,
    period: tuple[int, int],
) -> str:
    """
    Generate the filename for the climatology dataset.

    Parameters
    ----------
    outputdir: str
        Output directory for the results.
    aoi_name: str
        Name of the area of interest.
    dataproduct: DataProduct
        Data product type.
    aggregation: Aggregation
        Aggregation method.
    period: tuple[int, int]
        Period for the climatology dataset.

    Returns
    -------
    str
        Generated filename for the climatology dataset.
    """
    return f"{outputdir}/{aoi_name}_{dataproduct.product_name}_{aggregation.value}_{period[0]}-{period[1]}.nc"


def get_regridder(
    ds_source: xr.Dataset,
    ds_target: xr.Dataset,
    source_grid_file: str,
    target_grid_file: str,
    output_dir: str,
    method: str = "bilinear",
) -> xe.Regridder:
    logger.info(
        "Checking if regridder file from %s to downscaling grid %s already exists",
        source_grid_file,
        target_grid_file,
    )
    regridder_filename = f"{output_dir}/regridder/regridder-{Path(source_grid_file).stem}-to-{Path(target_grid_file).stem}.nc"
    if Path(regridder_filename).is_file():
        logger.info(
            "Found regridder file %s. Using it for regridding.", regridder_filename
        )
        regridder = xe.Regridder(
            ds_source, ds_target, method, weights=regridder_filename
        )
    else:
        Path(f"{output_dir}/regridder").mkdir(parents=True, exist_ok=True)
        logger.info(
            """Could not find regridder file in %s.
                    Creating a new one. This may take a while...""",
            regridder_filename,
        )
        regridder = xe.Regridder(ds_source, ds_target, method)
        regridder.to_netcdf(regridder_filename)
    return regridder


def check_input_dir(input_dir: str | None, default_dir: str) -> str:
    """
    Check the input directory.

    Details: During the data workflow, we need to access already created dataset. We need to ensure the directory exists.

    Parameters
    ----------
    input_dir: str | None
        The input directory where the data is located.
    default_dir: str
        The default input directory to use if input_dir is None.

    Returns
    -------
    str
        The input directory that was checked.
    """
    logger.info("Checking input directory...")
    if input_dir is None:
        input_dir = default_dir
        logger.warning(
            "Input directory not provided. Using default input directory %s", input_dir
        )
    if not Path(input_dir).is_dir():
        msg = f"Input directory {input_dir} not found."
        logger.error(msg)
        raise FileNotFoundError(msg)
    return input_dir


def _check_output_dir(
    output_dir: str | None, default_dir: str, subdir: list[str] | None = None
) -> str:
    """
    Check and create the output directory and subdirectories.

    Details: When data is downloaded and needs to be saved, we need to ensure that we can create the necessary directories.

    Parameters
    ----------
    output_dir: str | None
        The output directory where the results will be saved.
    default_dir: str
        The default output directory to use if output_dir is None.
    subdir: list[str] | None
        A list of subdirectories to create within the output_dir.

    Returns
    -------
    str
        The output directory that was checked or created.
    """

    logger.info("Checking output directory...")
    if output_dir is None:
        output_dir = default_dir
        logger.warning(
            "Output directory not provided. Using default output directory %s.",
            output_dir,
        )
    logger.info("Setting output directory: %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if subdir:
        for sub in subdir:
            logger.info("Creating output subdirectory: %s", f"{output_dir}/{sub}")
            Path(f"{output_dir}/{sub}").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_grid_file(
    output_dir: str, data_product: DataProduct, aoi_name: str, ds: xr.Dataset
) -> None:
    """
    Save the grid file of the given dataset (only if not already present)

    Parameters
    ----------
    output_dir: str
        The output directory where the grid file will be saved.
    data_product: DataProduct
        The data product for which to save the grid file.
    aoi_name: str
        The name of the area of interest.
    ds: xr.Dataset
        The dataset for which to save the grid file.
    """
    grid_file = f"{output_dir}/{data_product.product_name}_{aoi_name}_grid.nc"
    if not Path(grid_file).is_file():
        # Save the grid for the dataset
        logger.info("Saving %s grid for %s...", data_product.product_name, aoi_name)
        ds[["lon", "lat"]].to_netcdf(grid_file)
        logger.info("Grid saved to %s", grid_file)

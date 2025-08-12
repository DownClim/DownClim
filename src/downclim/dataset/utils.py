from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from enum import Enum
from typing import Any

import numpy as np
import xarray as xr
from aenum import MultiValueEnum

# GDAL configuration to avoid EPSG:4326 warnings
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
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
        scale_factor (dict[str, float]): scale factor for each variable in the data product.
        add_offset (dict[str, float]): add offset for each variable in the data product.
        lon_lat_names (dict[str, str]): longitude and latitude variable names.
        url (str): url to download the data product.
            If the data is stored on earthengine, provides the image collection path instead.

    """

    product_name: str
    period: tuple[int, int]
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

    CHELSA = (
        "chelsa",
        (1979, 2018),
        {"pr": 0.1, "tas": 0.1, "tasmin": 0.1, "tasmax": 0.1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "x", "lat": "y"},
        "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL",
    )
    CMIP6 = (
        "cmip6",
        (1850, 2100),
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
    )
    CORDEX = (
        "cordex",
        (1850, 2100),
        {"pr": 60 * 60 * 24, "tas": 1, "tasmin": 1, "tasmax": 1},
        {"pr": 0, "tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "https://esgf-node.ipsl.upmc.fr/esg-search",
    )
    GSHTD = (
        "gshtd",
        (2001, 2020),
        {"tas": 0.02, "tasmin": 0.02, "tasmax": 0.02},
        {"tas": -273.15, "tasmin": -273.15, "tasmax": -273.15},
        {"lon": "lon", "lat": "lat"},
        "projects/sat-io/open-datasets/GSHTD/",
    )
    CHIRPS = "chirps", (1980, 2024), {"pr": 1}, {"pr": 0}, {"lon": "lon", "lat": "lat"}, "UCSB-CHG/CHIRPS/DAILY"


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


def convert_cf_to_dt(t: np.datetime64) -> dt:
    return dt.strptime(str(t), "%Y-%m-%d %H:%M:%S")


def prep_dataset(ds: xr.Dataset, dataset_type: DataProduct) -> xr.Dataset:
    """Prepare a dataset for downscaling.

    Some operations (scaling and offsets) are applied to the dataset depending on
    the variables in the dataset.

    Parameters
    ----------
    ds: xr.Dataset original dataset to prepare.
    dataset_type: DataProduct
        Type of dataset to prepare.

    Returns
    -------
    xr.Dataset
        Prepared dataset with scaling and offsets applied to the variables.
    """
    cf = not isinstance(ds["time"].to_numpy()[0], np.datetime64)
    if cf:  # only cftime if not dt but should include more cases
        ds["time"] = [*map(convert_cf_to_dt, ds.time.values)]
    for key in ds.keys():
        try:
            ds[key] = ds[key] * dataset_type.scale_factor[key] + dataset_type.add_offset[key]
            ds[key].attrs = asdict(VariableAttributes[key].value)
        except KeyError as error:
            msg = f"Can't find scale factor and/or offset for variable {key} in dataset {dataset_type.product_name}."
            raise KeyError(msg) from error
    return ds


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

def climatology_filename(outputdir: str, aoi_name: str, dataproduct: DataProduct, aggregation: Aggregation, period: tuple[int, int]) -> str:
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

def get_grid(ds: xr.Dataset, data_product: DataProduct) -> xr.Dataset:
    """Save the reference grid for a given data product.

    Args:
        ds (xr.Dataset): The dataset containing the reference grid.
        data_product (DataProduct): The data product for which to save the reference grid.

    Returns
    -------
    xr.Dataset
        The reference grid dataset.
    """
    return ds[[data_product.lon_lat_names['lon'], data_product.lon_lat_names['lat']]]. \
            rename({data_product.lon_lat_names['lon']:'lon', data_product.lon_lat_names['lat']:'lat'})

from __future__ import annotations

import contextlib
import importlib.resources
from pathlib import Path

import yaml


def read_parameters(parameters_file: str | None = None) -> dict:
    """
    Read Downclim parameters from a file.

    Args:
        parameters_file (str | None, optional): Path to the parameters file. If None, the default parameters
            file is used. Defaults to None.

    Returns:
        dict: Dictionary with the parameters.
    """
    if not parameters_file:
        parameters_file = importlib.resources.path(__package__, "data/parameters.yml")

    # read parameters
    with contextlib.suppress(yaml.YAMLError), Path(parameters_file).open() as stream:
        parameters = yaml.safe_load(stream)
    return parameters

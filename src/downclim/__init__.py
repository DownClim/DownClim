"""
Copyright (c) 2024 Sylvain Schmitt. All rights reserved.

DownClim: Tool for dynamical downscaling for regional and national climate projections.
"""

from __future__ import annotations

from .logging_config import DownClimLoggerConfig, get_logger, setup_logging

__all__ = [
    "DownClimLoggerConfig",
    "get_logger",
    "setup_logging",
]

# Main exports
try:
    from .aoi import (  # noqa: F401
        extend_bounds,
        get_aoi,
        get_aoi_informations,
        sample_aoi,
    )
    from .downclim import (  # noqa: F401
        DownClimContext,
        define_DownClimContext_from_file,
        generate_DownClimContext_template_file,
    )
    from .downscale import (  # noqa: F401
        DownscaleMethod,
        bias_correction,
        run_downscaling,
    )
    from .evaluation import (  # noqa: F401
        compute_correlation,
        compute_evaluation,
        compute_mean,
        compute_rmse,
        compute_std,
        run_evaluation,
    )

    __all__.extend(
        [
            "DownClimContext",
            "DownscaleMethod",
            "bias_correction",
            "compute_correlation",
            "compute_evaluation",
            "compute_mean",
            "compute_rmse",
            "compute_std",
            "define_DownClimContext_from_file",
            "extend_bounds",
            "generate_DownClimContext_template_file",
            "get_aoi",
            "get_aoi_informations",
            "run_downscaling",
            "run_evaluation",
            "sample_aoi",
        ]
    )
except ImportError:
    pass

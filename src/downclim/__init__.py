"""
Copyright (c) 2024 Sylvain Schmitt. All rights reserved.

DownClim: Tool for dynamical downscaling for regional and national climate projections.
"""

from __future__ import annotations

from .logging_config import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
]

# Main exports
try:
    from .downclim import (
        DownClimContext,
        define_DownClimContext_from_file,
        generate_DownClimContext_template_file,
    )
    __all__.extend([
        "DownClimContext",
        "define_DownClimContext_from_file",
        "generate_DownClimContext_template_file",
    ])
except ImportError:
    pass

"""
Use examples of DownClim logging system
"""

from __future__ import annotations

import downclim

# Basic configuration
logger = downclim.setup_logging(
    level="INFO",  # ou logging.INFO
    log_file="my_downclim_analysis.log",
    console=True
)

# Advanced configuration
advanced_logger = downclim.setup_logging(
    level="DEBUG",
    log_file="detailed_logs.log",
    console=True,
    format_string="[%(asctime)s] %(name)s:%(levelname)s - %(message)s",
    max_file_size=50 * 1024 * 1024,  # 50 MB
    backup_count=3
)

# Configuration for debugging
debug_logger = downclim.setup_logging(
    level="DEBUG",
    log_file="debug.log",
    console=False  # No console output
)

# Configuration for production (error logs only)
prod_logger = downclim.setup_logging(
    level="ERROR",
    log_file="errors.log",
    console=False
)

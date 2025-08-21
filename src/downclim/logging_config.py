"""Configuration du logging pour le package DownClim."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any


class DownClimLoggerConfig:
    """Logging configuration class for DownClim."""

    _configured = False
    _logger_name = "downclim"

    @classmethod
    def setup_logging(
        cls,
        level: str | int = logging.INFO,
        log_file: str | Path | None = None,
        console: bool = True,
        format_string: str | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Configures logging for DownClim.

        Parameters
        ----------
        level : str | int
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file : str | Path, optional
            Path to the log file. If None, no log file is created.
        console : bool
            If True, log messages are displayed in the console
        format_string : str, optional
            Custom format for log messages
        max_file_size : int
            Maximum log file size in bytes
        backup_count : int
            Number of backup files to keep

        Returns
        -------
        logging.Logger
            The main DownClim logger
        """
        # Avoid multiple reconfigurations
        if cls._configured:
            return logging.getLogger(cls._logger_name)

        # Main logger
        logger = logging.getLogger(cls._logger_name)
        logger.setLevel(level)

        # Prevent propagation to root logger to avoid duplicates
        logger.propagate = False

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Default format
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)

        # Handler console
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Handler file
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._configured = True

        # Initial message
        logger.info("DownClim starting... Enjoy!")
        logger.info("DownClim logging system initialized")

        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Obtains a logger for a specific module.

        Parameters
        ----------
        name : str
            Module name (usually __name__)

        Returns
        -------
        logging.Logger
            Logger configured for the module
        """
        # If logging is not yet configured, use a minimal configuration
        if not cls._configured:
            cls.setup_logging(level=logging.INFO, log_file="DownClim.log")

        # Use the module name as is - it should already be in the form
        # "downclim.module" for DownClim modules
        module_logger = logging.getLogger(name)
        module_logger.propagate = True

        return module_logger

    @classmethod
    def set_level(cls, level: str | int) -> None:
        """
        Change the logging level for all DownClim loggers.

        Parameters
        ----------
        level : str | int
            New logging level
        """
        logger = logging.getLogger(cls._logger_name)
        logger.setLevel(level)

        # Update all handlers
        for handler in logger.handlers:
            handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Utility function to obtain a logger.

    Parameters
    ----------
    name : str
        Module name (use __name__)

    Returns
    -------
    logging.Logger
        Logger configured
    """
    return DownClimLoggerConfig.get_logger(name)

def setup_logging(**kwargs: Any) -> logging.Logger:
    """
    Utility function to configure logging.

    Parameters
    ----------
    **kwargs
        Arguments to pass to DownClimLoggerConfig.setup_logging()

    Returns
    -------
    logging.Logger
        Main DownClim logger
    """
    return DownClimLoggerConfig.setup_logging(**kwargs)

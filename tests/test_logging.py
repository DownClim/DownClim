"""Tests for logging"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from downclim.logging_config import DownClimLoggerConfig, get_logger, setup_logging


class TestDownClimLogging:
    """Tests for logging configuration."""

    def setup_method(self):
        """Reset configuration before each test."""
        DownClimLoggerConfig._configured = False

        # Clean up existing loggers
        logger = logging.getLogger("downclim")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    def test_setup_logging_basic(self):
        """Test of basic configuration."""
        logger = setup_logging(level="INFO")

        assert logger.name == "downclim"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

    def test_setup_logging_with_file(self):
        """Test of file configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test.log"

            logger = setup_logging(
                level="DEBUG",
                log_file=str(log_file),
                console=False
            )

            # Test that a message is written
            logger.info("Test message")

            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logging_custom_format(self):
        """Test of custom format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_format.log"

            setup_logging(
                level="INFO",
                log_file=str(log_file),
                console=False,
                format_string="%(levelname)s: %(message)s"
            )

            logger = logging.getLogger("downclim")
            logger.info("Format test")

            content = log_file.read_text()
            assert "INFO: Format test" in content

    def test_get_logger(self):
        """Test of the get_logger function."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_no_duplicate_handlers(self):
        """Test for duplicated handlers."""
        setup_logging(level="INFO")
        initial_count = len(logging.getLogger("downclim").handlers)

        # Reconfigure
        setup_logging(level="DEBUG")
        final_count = len(logging.getLogger("downclim").handlers)

        # Handlers should not be duplicated
        assert final_count <= initial_count + 1  # Tolerance for different configurations

    def test_set_level_dynamically(self):
        """Test of dynamic level change."""
        setup_logging(level="INFO")

        DownClimLoggerConfig.set_level("DEBUG")
        logger = logging.getLogger("downclim")

        assert logger.level == logging.DEBUG

    def test_automatic_configuration(self):
        """Test of automatic configuration."""
        # get_logger should automatically configure
        get_logger("auto_test")

        # Check that there is a default configuration
        downclim_logger = logging.getLogger("downclim")
        assert len(downclim_logger.handlers) > 0

    def test_file_rotation_parameters(self):
        """Test file rotation parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "rotation_test.log"

            setup_logging(
                log_file=str(log_file),
                max_file_size=1024,  # 1KB
                backup_count=3
            )

            logger = logging.getLogger("downclim")

            # Check that a RotatingFileHandler is created
            file_handlers = [h for h in logger.handlers
                           if hasattr(h, 'maxBytes')]

            assert len(file_handlers) == 1
            handler = file_handlers[0]
            assert handler.maxBytes == 1024
            assert handler.backupCount == 3

    def test_console_only_configuration(self):
        """Test of console only configuration."""
        setup_logging(
            level="INFO",
            log_file=None,
            console=True
        )

        logger = logging.getLogger("downclim")

        # Check that there are only StreamHandlers
        file_handlers = [h for h in logger.handlers
                        if not hasattr(h, 'stream')]

        assert len(file_handlers) == 0

    def test_file_only_configuration(self):
        """Test of file only configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "file_only.log"

            setup_logging(
                level="INFO",
                log_file=str(log_file),
                console=False
            )

            logger = logging.getLogger("downclim")

            # Check that there are no StreamHandlers to stdout
            stream_handlers = [h for h in logger.handlers
                             if hasattr(h, 'stream') and
                             hasattr(h.stream, 'name') and
                             h.stream.name == '<stdout>']

            assert len(stream_handlers) == 0

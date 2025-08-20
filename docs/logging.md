# DownClim logging

## Overview

The DownClim package uses a centralized and configurable logging system that
allows:

- Capturing all important events
- Customizing log levels
- Configuring output (console, file, or both)
- Managing log file rotation
- Avoiding conflicts with other libraries

## Automatic configuration

The system automatically configures itself with default parameters upon first
use:

- Level: INFO
- File: DownClim.log
- Console: enabled
- Format : `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

## Custom configuration

### Basic configuration

```python
import downclim

# Simple configuration
logger = downclim.setup_logging(level="INFO", log_file="mon_projet.log", console=True)
```

### Advanced configuration

```python
import downclim
import logging

logger = downclim.setup_logging(
    level=logging.DEBUG,
    log_file="logs/analysis.log",
    console=True,
    format_string="[%(asctime)s] %(name)s:%(levelname)s - %(message)s",
    max_file_size=50 * 1024 * 1024,  # 50 MB
    backup_count=5,  # Keep 5 backup files
)
```

### Configurations for multiple environments

```python
import downclim

# Configuration for development
downclim.setup_logging(level="DEBUG", log_file="dev.log", console=True)

# Configuration for production
downclim.setup_logging(
    level="ERROR",
    log_file="/var/log/downclim/errors.log",
    console=False,
    max_file_size=100 * 1024 * 1024,  # 100 MB
    backup_count=10,
)

# Configuration for tests
downclim.setup_logging(level="WARNING", log_file="tests.log", console=False)
```

#### Development

```python
# Detailed logs with console output
downclim.setup_logging(level="DEBUG", log_file="dev.log", console=True)
```

#### Production

```python
# Error logs only to file
downclim.setup_logging(
    level="ERROR",
    log_file="/var/log/downclim/errors.log",
    console=False,
    max_file_size=100 * 1024 * 1024,  # 100 MB
    backup_count=10,
)
```

#### Tests

```python
# Logs uniquement en fichier pour éviter la pollution de sortie
downclim.setup_logging(level="WARNING", log_file="tests.log", console=False)
```

## Multiple logging levels

| Level    | Usage                                            |
| -------- | ------------------------------------------------ |
| DEBUG    | Detailed information for debugging               |
| INFO     | General information about the execution          |
| WARNING  | Warnings (unexpected but non-blocking behaviors) |
| ERROR    | Errors that prevent a function from executing    |
| CRITICAL | Critical errors that may stop the program        |

## Dynamic control

### Change the level at runtime

```python
import downclim

# Initial configuration
downclim.setup_logging(level="INFO")

# Change for more details
downclim.DownClimLoggerConfig.set_level("DEBUG")

# Back to normal level
downclim.DownClimLoggerConfig.set_level("INFO")
```

### Get a logger for a custom module

```python
import downclim

# For your own code
logger = downclim.get_logger(__name__)
logger.info("My custom message")
```

## File rotation

```python
import downclim

# For your own code
logger = downclim.get_logger(__name__)
logger.info("My custom message")
```

Logging system includes automatic file rotation :

- when a log file reaches the maximum size, it is renamed with a suffix
- Example: `analysis.log` → `analysis.log.1`
- Old files are deleted according to `backup_count`

## Best practices

### 1. Configuration at the beginning of the script

```python
import downclim

# Configure before any usage
downclim.setup_logging(level="INFO", log_file="mon_analyse.log")
```

### 2. Environment management

```python
import os
import downclim

# Configuration according to the environment
if os.getenv("ENV") == "production":
    level = "ERROR"
    console = False
else:
    level = "DEBUG"
    console = True

downclim.setup_logging(level=level, console=console, log_file="downclim.log")
```

### 3. Structured logs for analysis

```python
# Use formats suitable for analysis
downclim.setup_logging(
    format_string="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    log_file="structured.log",
)
```

## Help

### Problem: Duplicate logs

```python
# Solution: reconfigure properly
downclim.DownClimLoggerConfig._configured = False
downclim.setup_logging(level="INFO", log_file="new.log")
```

### Problem: No visible logs

```python
# Check the configuration
import logging

logger = logging.getLogger("downclim")
print(f"Level: {logger.level}")
print(f"Handlers : {logger.handlers}")
```

### Problem: File permissions

```python
# Use an accessible directory
import tempfile
import os

log_dir = os.path.join(tempfile.gettempdir(), "downclim_logs")
os.makedirs(log_dir, exist_ok=True)

downclim.setup_logging(log_file=os.path.join(log_dir, "downclim.log"))
```

## Migration from the old system

If you were using the old system with `logging.basicConfig()`, replace :

```python
# Old
import logging

logging.basicConfig(level=logging.INFO)

# New
import downclim

downclim.setup_logging(level="INFO")
```

The new system is compatible and does not interfere with other logging
configurations.

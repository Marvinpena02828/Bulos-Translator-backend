# Logging Configuration Module

## Overview

The logging configuration module (`logging_config.py`) provides comprehensive logging capabilities for the Bulos Translator backend. It implements all requirements from Requirement 10 (Error Handling and Logging) of the backend architecture specification.

## Features

### 1. **Multiple Log Handlers**
- **Console Handler**: Real-time output to stdout for monitoring
- **File Handler**: Rotating file handler for all logs (INFO and above)
- **Error Handler**: Separate rotating file handler for ERROR logs only

### 2. **Rotating File Handler**
- **Max Size**: 10MB per log file (configurable)
- **Backups**: 5 backup files retained (configurable)
- **Automatic Rotation**: Old files automatically backed up when size limit reached
- Files are named: `app.log`, `app.log.1`, `app.log.2`, etc.

### 3. **Sensitive Data Filtering**
The `SensitiveDataFilter` class automatically redacts sensitive information from logs:
- Passwords
- Authentication tokens (including Bearer tokens)
- API keys
- Secrets

**Example:**
```python
# Input log message:
logger.info('User login with password="mysecret123"')

# Logged as:
User login with password="[REDACTED]"
```

### 4. **Structured Log Formatting**

**Console Format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
Example: 2024-06-24 18:30:45 - app.services - INFO - User logged in
```

**File Format (includes function and line number):**
```
%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
Example: 2024-06-24 18:30:45 - app.services - INFO - login_user:42 - User logged in
```

## Usage

### Basic Setup

```python
from utils.logging_config import configure_logging, get_logger

# Configure logging (typically done once at application startup)
configure_logging()

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

### Custom Configuration

```python
from utils.logging_config import configure_logging

# Configure with custom settings
configure_logging(
    log_level="DEBUG",                    # Set to DEBUG for development
    log_file_path="./logs/custom.log",    # Custom log file path
    log_max_bytes=5 * 1024 * 1024,       # 5MB max size
    log_backup_count=10                   # Keep 10 backups
)
```

### Using Settings Defaults

By default, `configure_logging()` uses values from `config.py` settings:
- `settings.log_level` (default: "INFO")
- `settings.log_file_path` (default: "./logs/app.log")
- `settings.log_max_bytes` (default: 10MB)
- `settings.log_backup_count` (default: 5)

## Requirements Mapping

| Requirement | Implementation |
|------------|----------------|
| 10.1: Log exceptions with stack trace | Use `logger.error("message", exc_info=True)` to include stack traces |
| 10.2: Log API requests with timestamps | Log format includes timestamps; middleware logs requests |
| 10.3: Log critical errors at ERROR level | Error handler captures ERROR and above |
| 10.4: Log informational events at INFO level | Console and file handlers set to INFO level |
| 10.5: Maintain log files with rotation | RotatingFileHandler with 10MB max, 5 backups |
| 10.6: Not log sensitive information | SensitiveDataFilter redacts passwords, tokens, keys |

## Log Files

After configuration, the following log files are created in the `logs/` directory:

- **`app.log`**: All logs (INFO, WARNING, ERROR, CRITICAL)
- **`app.log.1`**, **`app.log.2`**, etc.: Rotated backup files
- **`error.log`**: ERROR and CRITICAL logs only
- **`error.log.1`**, **`error.log.2`**, etc.: Rotated error backups

## Integration with FastAPI

The logging configuration should be called during application startup:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from utils.logging_config import configure_logging, get_logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Application starting...")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)
```

## Testing

### Unit Tests

Run the comprehensive test suite:
```bash
pytest tests/test_logging_config.py -v
```

Test coverage includes:
- Sensitive data filtering for passwords, tokens, API keys, secrets
- Log handler creation and configuration
- Rotating file handler settings (max bytes, backup count)
- Log level configuration
- Log format validation
- Integration tests for actual file writing
- Sensitive data redaction in written logs

### Manual Testing

Run the manual test script to see logging in action:
```bash
python test_logging_manual.py
```

This script demonstrates:
- Console output
- File logging
- Error logging
- Sensitive data filtering
- Exception logging with stack traces

## Best Practices

1. **Get loggers with module name:**
   ```python
   logger = get_logger(__name__)
   ```

2. **Use appropriate log levels:**
   - `DEBUG`: Detailed diagnostic information
   - `INFO`: General informational messages
   - `WARNING`: Warning messages (potential issues)
   - `ERROR`: Error messages (actual problems)
   - `CRITICAL`: Critical failures

3. **Include context in messages:**
   ```python
   logger.info(f"User {user_id} logged in from {ip_address}")
   logger.error(f"Failed to connect to database: {db_url}", exc_info=True)
   ```

4. **Use exc_info for exceptions:**
   ```python
   try:
       # code that may raise exception
   except Exception as e:
       logger.error("Operation failed", exc_info=True)
   ```

5. **Avoid logging sensitive data directly:**
   While the SensitiveDataFilter provides protection, it's best to avoid including sensitive data in log messages when possible.

## Performance Considerations

- **Async compatibility**: The logging module is thread-safe and works with FastAPI's async operations
- **File I/O**: Logging is buffered to minimize performance impact
- **Rotation**: File rotation happens automatically when size limits are reached
- **Filtering**: Sensitive data filtering uses compiled regex patterns for efficiency

## Troubleshooting

### Logs not appearing in files
- Check that the `logs/` directory exists (it's created automatically)
- Verify file permissions
- Ensure log level is set appropriately (INFO or DEBUG)

### Sensitive data not being redacted
- Verify the SensitiveDataFilter patterns match your data format
- Check that the filter is applied to all handlers
- Add custom patterns to `SensitiveDataFilter.SENSITIVE_PATTERNS` if needed

### Log files growing too large
- Adjust `log_max_bytes` to a smaller value
- Reduce `log_backup_count` to keep fewer backups
- Consider log level adjustment (INFO vs DEBUG)

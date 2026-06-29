"""Logging configuration module for Bulos Translator backend.

This module provides comprehensive logging configuration with:
- Console handler for real-time output
- Rotating file handlers for general and error logs
- Sensitive data filtering to prevent logging passwords and tokens
- Structured log formatting with timestamps and line numbers
"""
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import settings


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from log messages.
    
    This filter scans log messages for patterns indicating sensitive data
    (passwords, tokens, API keys) and redacts them before writing to logs.
    """
    
    # Patterns to detect sensitive data
    SENSITIVE_PATTERNS = [
        (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\s,}]+)', re.IGNORECASE), 'password'),
        (re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\s,}]+)', re.IGNORECASE), 'token'),
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\s,}]+)', re.IGNORECASE), 'api_key'),
        (re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\s,}]+)', re.IGNORECASE), 'secret'),
        (re.compile(r'authorization:\s*bearer\s+([^\s]+)', re.IGNORECASE), 'bearer_token'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact sensitive information from log record.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to allow the record to be logged (always returns True after redaction)
        """
        # Get the formatted message
        message = str(record.getMessage())
        
        # Check for sensitive patterns and redact
        for pattern, data_type in self.SENSITIVE_PATTERNS:
            if pattern.search(message):
                # Redact the sensitive value while preserving structure
                message = pattern.sub(f'{data_type}="[REDACTED]"', message)
        
        # Update the record message
        record.msg = message
        record.args = ()  # Clear args since we've already formatted the message
        
        return True


def configure_logging(
    log_level: Optional[str] = None,
    log_file_path: Optional[str] = None,
    log_max_bytes: Optional[int] = None,
    log_backup_count: Optional[int] = None
) -> None:
    """Configure application logging with console, file, and error handlers.
    
    Args:
        log_level: Logging level (defaults to settings.log_level)
        log_file_path: Path to main log file (defaults to settings.log_file_path)
        log_max_bytes: Max log file size before rotation (defaults to settings.log_max_bytes)
        log_backup_count: Number of backup files to keep (defaults to settings.log_backup_count)
    """
    # Use settings defaults if not provided
    log_level = log_level or settings.log_level
    log_file_path = log_file_path or settings.log_file_path
    log_max_bytes = log_max_bytes or settings.log_max_bytes
    log_backup_count = log_backup_count or settings.log_backup_count
    
    # Ensure log directory exists
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console Handler - for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler - rotating file for all logs
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=log_max_bytes,  # 10MB default
        backupCount=log_backup_count,  # 5 backups default
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Error Handler - separate file for errors only
    error_log_path = str(Path(log_file_path).parent / 'error.log')
    error_handler = RotatingFileHandler(
        filename=error_log_path,
        maxBytes=log_max_bytes,  # 10MB default
        backupCount=log_backup_count,  # 5 backups default
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    # Add sensitive data filter to all handlers
    sensitive_filter = SensitiveDataFilter()
    console_handler.addFilter(sensitive_filter)
    file_handler.addFilter(sensitive_filter)
    error_handler.addFilter(sensitive_filter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Log successful configuration
    root_logger.info("Logging configuration completed successfully")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log file: {log_file_path}")
    root_logger.info(f"Error log file: {error_log_path}")
    root_logger.info(f"Max log file size: {log_max_bytes / (1024*1024):.1f}MB")
    root_logger.info(f"Backup count: {log_backup_count}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__ from the calling module)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)

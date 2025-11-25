"""
Centralized logging configuration for VOKG
Supports both JSON and text formatting with structured context
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import json

from .config import settings


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra context if present
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add request_id if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add user_id if present
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # Add task_id if present (for Celery)
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter
    """

    def __init__(self):
        fmt = "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(
    service_name: str = "vokg",
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application

    Args:
        service_name: Name of the service for logger identification
        log_level: Override default log level
        log_format: Override default log format (json/text)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    level = log_level or settings.LOG_LEVEL
    format_type = log_format or settings.LOG_FORMAT
    file_path = log_file or settings.LOG_FILE

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Choose formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class ContextLogger:
    """
    Logger wrapper that adds contextual information to all log messages
    """

    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.context = context or {}

    def _log(self, level: int, message: str, **kwargs):
        extra = {"context": {**self.context, **kwargs}}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        extra = {"context": {**self.context, **kwargs}}
        self.logger.exception(message, extra=extra)

    def with_context(self, **kwargs) -> "ContextLogger":
        """
        Create a new logger with additional context
        """
        return ContextLogger(self.logger, {**self.context, **kwargs})


def get_logger(name: str, **context) -> ContextLogger:
    """
    Get a context-aware logger instance

    Args:
        name: Logger name
        **context: Additional context to include in all log messages

    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)


# Initialize default logger
default_logger = setup_logging()

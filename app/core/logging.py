"""Central logging configuration."""

from __future__ import annotations

import logging
import logging.config

from app.core.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": _LOG_FORMAT, "datefmt": "%Y-%m-%dT%H:%M:%S%z"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level.upper()},
            "loggers": {
                "uvicorn": {"handlers": ["console"], "level": settings.log_level.upper(), "propagate": False},
                "uvicorn.error": {"level": settings.log_level.upper()},
                "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
                "sqlalchemy.engine": {"level": "WARNING", "propagate": True},
                "httpx": {"level": "WARNING", "propagate": True},
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

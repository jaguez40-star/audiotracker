"""structlog en JSON, alineado con el stack de Robustez V2.0."""

from __future__ import annotations

import logging
import sys
import warnings

import structlog


def _silence_expected_warnings() -> None:
    """Calla avisos de dependencias sobre condiciones que ya resolvimos.

    pyannote avisa en cada import que torchcodec no carga y que su decodificacion
    interna fallara. Aqui es esperado — WDAC bloquea esas DLLs — y no aplica
    porque le pasamos el audio ya decodificado en memoria. Dejarlo visible seria
    un aviso permanente sobre algo que no esta roto.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*torchcodec is not installed correctly.*",
        category=UserWarning,
    )


def configure_logging(level: int = logging.INFO) -> None:
    _silence_expected_warnings()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

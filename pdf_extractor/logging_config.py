"""Shared logging setup for CLI entrypoints."""

from __future__ import annotations

import logging


# Libraries that spam DEBUG when root logger is DEBUG (esp. pdfplumber/pdfminer).
_NOISY_LOGGERS = (
    "pdfminer",
    "pdfminer.cmapdb",
    "pdfminer.psparser",
    "pdfminer.pdfinterp",
    "pdfminer.pdfdocument",
    "pdfminer.pdfpage",
    "pdfminer.pdfparser",
    "pdfminer.converter",
    "pdfplumber",
    "PIL",
    "urllib3",
    "filelock",
    "httpx",
    "httpcore",
    "openai",
    "transformers",
)


def configure_app_logging(verbose: bool = False) -> None:
    """Configure root logging and silence third-party DEBUG spam."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Even in -v mode, pdfminer token dumps are useless for diagnosing tables.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

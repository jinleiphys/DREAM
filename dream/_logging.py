"""Library logging configuration."""

import logging

logger = logging.getLogger("dream")
logger.addHandler(logging.NullHandler())

"""
KlipMachine - Logging
Simple logging configuration.
"""

import logging
import sys
from pathlib import Path

# Create logger
logger = logging.getLogger("KlipMachine")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler
logger.addHandler(console_handler)

# Prevent propagation
logger.propagate = False

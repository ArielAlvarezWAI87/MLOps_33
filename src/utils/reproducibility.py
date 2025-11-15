"""
reproducibility.py
───────────────────────────────────────────────
Utilities for ensuring reproducible results

Author: MLOps Team
Purpose: Set random seeds for all libraries to ensure reproducibility
"""

import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Global random seed
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - Environment variable for Python hash seed

    Args:
        seed (int): Random seed value. Default is 42.

    Example:
        >>> from src.utils.reproducibility import set_seeds
        >>> set_seeds(42)
    """
    logger.info(f"Setting random seed to {seed} for reproducibility")

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash seed (for dictionary ordering, etc.)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info("✅ Random seeds set successfully")


def get_seed() -> int:
    """
    Get the current global random seed value.

    Returns:
        int: The current random seed value
    """
    return RANDOM_SEED


def set_global_seed(seed: int):
    """
    Set the global random seed and update all libraries.

    Args:
        seed (int): New random seed value
    """
    global RANDOM_SEED
    RANDOM_SEED = seed
    set_seeds(seed)

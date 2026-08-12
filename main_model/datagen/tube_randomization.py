"""
Module: tube_randomization.py
Provides functionality for tube_randomization.
"""

import random
import math

# Tube constants
TUBE_HEIGHT_MIN = 0.7
TUBE_HEIGHT_MAX = 0.9

# Hounsfield Units (HU) range for realistic Endotracheal Tube appearance
TUBE_INTENSITY_MIN = 200.0
TUBE_INTENSITY_MAX = 500.0


def get_random_tube_diameter():
    """
    Returns a random tube diameter and its corresponding thickness.
    This should be called ONCE per prior-current pair so that
    both the prior and current tubes have the exact same diameter.
    """
    diameter = math.floor(min(max(random.gauss(8, 4), 8), 11))
    thickness = math.ceil(diameter / 2)
    return diameter, thickness


def get_random_tube_params(tube_diameter):
    """
    Returns a dictionary of randomized tube parameters.
    This should be called EACH TIME a tube is added (so prior and current
    can have different placement, intensity, path, etc.).
    """
    return {
        "placement": random.choice(["LEFT", "RIGHT"]),
        "height_fraction": min(max(random.gauss(0.8, 0.3), TUBE_HEIGHT_MIN), TUBE_HEIGHT_MAX),
        "intensity": random.uniform(TUBE_INTENSITY_MIN, TUBE_INTENSITY_MAX),
        "path_smoothing_sigma": random.uniform(12, 15),
        "walk_step_std": max(3, math.ceil(tube_diameter / 2))
    }

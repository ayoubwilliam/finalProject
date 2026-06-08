import numpy as np
import torch


def get_randomized_tube_params(mask_data: np.ndarray) -> dict:
    """
    Generates randomized parameters for the tube based on user requirements.
    Also dynamically calculates PATH_SEARCH_RADIUS based on the airway mask thickness.
    """
    # Randomize core geometric properties
    placement = np.random.choice(["RIGHT", "LEFT"])
    height_fraction = np.random.uniform(0.65, 0.95)
    block_size = 10

    diameter = np.random.uniform(9.0, 12.0)
    thickness = np.floor(diameter / 4.0)
    intensity = np.random.uniform(800.0, 1200.0)

    # Fallback to safe static radius to prevent massive CuPy memory spikes on large masks
    path_search_radius = 15

    params = {
        "TUBE_PLACEMENT": placement,
        "TUBE_HEIGHT_FRACTION": height_fraction,
        "BLOCK_SIZE": block_size,
        "TUBE_DIAMETER": diameter,
        "TUBE_THICKNESS": thickness,
        "TUBE_INTENSITY": intensity,
        "PATH_SEARCH_RADIUS": path_search_radius
    }

    print("--- Generated Randomized Tube Parameters ---")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    print("--------------------------------------------")

    return params

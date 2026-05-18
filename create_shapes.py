import sys
import numpy as np

from file_handler import load_nifti, save_nifti
from noise import create_noise


def create_ellipsoid(data: np.ndarray, center: list[int], radius: list[int]) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radius

    # Coordinate grids
    x = np.arange(data.shape[0])[:, None, None]
    y = np.arange(data.shape[1])[None, :, None]
    z = np.arange(data.shape[2])[None, None, :]

    # Ellipsoid equation (normalized distance)
    dist = np.sqrt(((x - cx) / rx) ** 2 +
                   ((y - cy) / ry) ** 2 +
                   ((z - cz) / rz) ** 2)

    # Fill ellipsoid
    mask = dist <= 1
    return mask


def create_cylinder(data: np.ndarray, center: list[int], radius: int, height: int) -> np.ndarray:
    cx, cy, cz = center

    # Create coordinate grids
    x = np.arange(0, data.shape[0])[:, None, None]
    y = np.arange(0, data.shape[1])[None, :, None]
    z = np.arange(0, data.shape[2])[None, None, :]

    # 1. Radial condition (xy plane)
    # Note: Comparing squared values is slightly faster than using np.sqrt
    radial_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (radius ** 2)

    # 2. Height condition (z axis)
    # This centers the height of the cylinder exactly at cz
    z_mask = np.abs(z - cz) <= (height / 2)

    # 3. Combine masks
    # The bitwise AND (&) will broadcast the final mask to the full (X, Y, Z) shape
    mask = radial_mask & z_mask

    return mask


def create_torus_segment(data: np.ndarray, center: list[int], tube_radius: int, height: int,
                         curve_radius: float, inner_radius: float = 0.0) -> np.ndarray:
    """
    Creates a curved tube (ET Tube) by extracting a segment from a Torus.
    tube_radius: The outer thickness of the tube (Minor Radius).
    curve_radius: How tight the bend is (Major Radius).
    inner_radius: The hollow center radius (default 0 for solid).
    """
    cx, cy, cz = center

    x = np.arange(0, data.shape[0])[:, None, None]
    y = np.arange(0, data.shape[1])[None, :, None]
    z = np.arange(0, data.shape[2])[None, None, :]

    hole_x = cx + curve_radius
    # hole_x = cx - curve_radius  # Shifts the hole to the left

    hole_z = cz

    dist_from_hole = np.sqrt((x - hole_x) ** 2 + (z - hole_z) ** 2)

    # Calculate squared distance from the core ring
    torus_dist_sq = (dist_from_hole - curve_radius) ** 2 + (y - cy) ** 2

    # ET Tube condition: Outer boundary AND Inner hollow boundary
    torus_mask = (torus_dist_sq <= tube_radius ** 2) & (torus_dist_sq >= inner_radius ** 2)

    z_mask = (z <= cz) & (z >= cz - height)

    x_mask = x <= hole_x
    # x_mask = x >= hole_x  # Keeps the opposite side of the torus

    mask = torus_mask & z_mask & x_mask

    return mask


def apply_mask(data: np.ndarray, mask: np.ndarray, intensity: int, is_noised: bool) -> np.ndarray:
    if is_noised:
        noise = create_noise(data.shape)
        data[mask] = intensity + noise[mask]
    else:
        data[mask] = intensity

    return data


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    intensity = int(sys.argv[3])

    data, affine, header = load_nifti(input_path)

    mask = create_cylinder(data, [250, 300, 400], 10, 100)

    data = apply_mask(data, mask, intensity, True)
    save_nifti(output_path, data, affine, header)

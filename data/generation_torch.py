import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from utils import exponential_kernel

def get_region_points(s, region_assignments, region_id):
    """
    Extract the points from the spatial locations (s) that belong to the region with id `region_id`.
    """
    region_mask = (region_assignments == region_id)
    region_points = s[region_mask]
    return region_points


def generate_ordered_uniform_random_locations(n_i, x_min, x_max, y_min, y_max):
    """
    Generate n_i ordered uniform random locations within the bounding box defined by
    (x_min, x_max) and (y_min, y_max).
    """
    x_coords = torch.rand(n_i) * (x_max - x_min) + x_min
    y_coords = torch.rand(n_i) * (y_max - y_min) + y_min

    ordered_indices = torch.argsort(x_coords)
    ordered_x = x_coords[ordered_indices]
    ordered_y = y_coords[ordered_indices]

    ordered_locations = torch.stack([ordered_x, ordered_y], dim=1)
    return ordered_locations


def generate_location_and_partitions(B, n_i, d_min=1):
    """
    Partition a square of size (0, d_min * sqrt(B)) x (0, d_min * sqrt(B)) into B regions.
    Then generate n_i points in each region.
    """
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(B, dtype=torch.float32))).item())
    region_length = d_min

    x_divisions = torch.linspace(0, grid_size * region_length, grid_size + 1)
    y_divisions = torch.linspace(0, grid_size * region_length, grid_size + 1)

    s = []
    region_assignments = []

    region_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if region_id >= B:
                break

            x_min, x_max = x_divisions[i], x_divisions[i + 1]
            y_min, y_max = y_divisions[j], y_divisions[j + 1]

            ordered_region_points = generate_ordered_uniform_random_locations(n_i, x_min, x_max, y_min, y_max)
            s.append(ordered_region_points)

            region_assignments.extend([region_id] * n_i)
            region_id += 1

    s = torch.cat(s, dim=0)
    region_assignments = torch.tensor(region_assignments, dtype=torch.long)
    return s, region_assignments

def generate_data(s, x, sigmasq, phi, beta_true, tausq_true, seed=42, spatial=True):
    """
    Generates spatial data for a full region first, then assigns regions.
    """
    N = s.shape[0]
    torch.manual_seed(seed)
    if spatial:
        K = sigmasq * exponential_kernel(s, s, phi)
        K = (K + (K.T))/2
        w = torch.tensor(torch.distributions.MultivariateNormal(torch.zeros(N), K).sample())
    else:
        w = torch.zeros(N)

    e = torch.normal(0, torch.sqrt(torch.tensor(tausq_true)), size=(N,))
    y = beta_true * x.flatten() + w + e

    return y, w, e

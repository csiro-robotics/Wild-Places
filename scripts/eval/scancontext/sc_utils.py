# Code based on ScanContext implementation: https://github.com/irapkaist/scancontext/blob/master/python/make_sc_example.py
# Partially vectorized implementation by Jacek Komorowski: https://github.com/jac99/Egonn/blob/main/third_party/scan_context/scan_context.py

import numpy as np
import numpy_indexed as npi

def sc2rk(sc):
    # Scan context to ring key
    return np.mean(sc, axis=1)

def pt2rs(points, gap_ring, gap_sector):
    # np.arctan2 produces values in -pi..pi range
    theta = np.arctan2(points[:, 1], points[:, 0]) + np.pi
    eps = 1e-6

    theta = np.clip(theta, a_min=0., a_max=2*np.pi-eps)
    faraway = np.linalg.norm(points[:, 0:2], axis=1)

    idx_ring = (faraway // gap_ring).astype(int)
    idx_sector = (theta // gap_sector).astype(int)

    return idx_ring, idx_sector

def get_sc(x):
    lidar_height = 1.6
    num_sector = 60
    num_ring = 20
    max_length = 80
    gap_ring = max_length / num_ring
    gap_sector = 2. * np.pi / num_sector
    idx_ring, idx_sector = pt2rs(x, gap_ring, gap_sector)
    height = x[:, 2] + lidar_height

    # Filter out points that are self.max_length or further away
    mask = idx_ring < num_ring
    idx_ring = idx_ring[mask]
    idx_sector = idx_sector[mask]
    height = height[mask]

    assert idx_ring.shape == idx_sector.shape
    assert idx_ring.shape == height.shape

    # Convert idx_ring and idx_sector to a linear index
    idx_linear = idx_ring * num_sector + idx_sector
    idx, max_height = npi.group_by(idx_linear).max(height)

    sc = np.zeros([num_ring, num_sector])
    # As per original ScanContext implementation, the minimum height is always non-negative
    # (they take max from 0 and height values)
    sc[idx // num_sector, idx % num_sector] = np.clip(max_height, a_min=0., a_max=None)

    return sc


def distance_sc(sc1, sc2):
    # Distance between 2 scan context descriptors
    num_sectors = sc1.shape[1]

    # Repeate to move 1 columns
    _one_step = 1  # const
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = np.roll(sc1, _one_step, axis=1)  # columne shift

        # Compare
        sc1_norm = np.linalg.norm(sc1, axis=0)
        sc2_norm = np.linalg.norm(sc2, axis=0)
        mask = ~np.logical_or(np.isclose(sc1_norm, 0.), np.isclose(sc2_norm, 0.))

        # Compute cosine similarity between columns of sc1 and sc2
        cossim = np.sum(np.multiply(sc1[:, mask], sc2[:, mask]), axis=0) / (sc1_norm[mask] * sc2_norm[mask])

        sim_for_each_cols[i] = np.sum(cossim) / np.sum(mask)

    yaw_diff = (np.argmax(sim_for_each_cols) + 1) % sc1.shape[1]  # because python starts with 0
    sim = np.max(sim_for_each_cols)
    dist = 1. - sim

    return dist, yaw_diff
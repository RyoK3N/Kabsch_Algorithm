# kabsch.py

import numpy as np
import streamlit as st

def perform_kabsch(ground_points, elevated_points):
    """
    Performs the Kabsch algorithm to find the optimal rotation and translation
    that aligns ground_points to elevated_points.

    Parameters:
        ground_points (list of tuples): List of (x, y) points from the ground image.
        elevated_points (list of tuples): List of (x, y) points from the elevated image.

    Returns:
        tuple: Rotation matrix (2x2), translation vector (2,), transformed_ground_points (Nx2),
               elevated_points_np (Nx2)
               Returns None if input points are insufficient or mismatched.
    """
    min_points = min(len(ground_points), len(elevated_points))
    if min_points < 3:
        st.warning("Please ensure at least 3 corresponding points are drawn on both images.")
        return None

    ground_points_np = np.array(ground_points[:min_points])
    elevated_points_np = np.array(elevated_points[:min_points])

    centroid_ground = np.mean(ground_points_np, axis=0)
    centroid_elevated = np.mean(elevated_points_np, axis=0)

    ground_centered = ground_points_np - centroid_ground
    elevated_centered = elevated_points_np - centroid_elevated


    H = ground_centered.T @ elevated_centered

    #Find H insverse 

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_elevated - R @ centroid_ground

    # Apply the transformation to ground points
    transformed_ground = (R @ ground_points_np.T).T + t

    return R, t, transformed_ground, elevated_points_np, H

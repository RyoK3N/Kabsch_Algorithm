# utils.py

import numpy as np
import streamlit as st

def extract_line_points(path):
    """
    Extracts numeric coordinates from the path list and groups them into (x, y) tuples.

    Parameters:
        path (list): The path list containing commands and coordinates.

    Returns:
        list: A list of (x, y) tuples representing points.
    """
    numeric_values = [val for val in path if isinstance(val, (int, float))]
    points = []

    # Group numeric values into (x, y) pairs
    for i in range(0, len(numeric_values), 2):
        if i + 1 < len(numeric_values):
            x = int(numeric_values[i])
            y = int(numeric_values[i + 1])
            points.append((x, y))

    return points

def calculate_angle(A, B, C):
    """
    Calculates the angle ABC (in degrees) given three points A, B, and C.

    Parameters:
        A (tuple): Coordinates of point A (x, y).
        B (tuple): Coordinates of point B (x, y).
        C (tuple): Coordinates of point C (x, y).

    Returns:
        float: The angle ABC in degrees.
    """
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    # Clamp the cosine to the valid range to avoid numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
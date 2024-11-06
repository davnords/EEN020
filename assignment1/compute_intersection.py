import numpy as np
from pflat import pflat
def compute_intersection(points1, points2):
    """
    Compute the interesction of lines between two points (3x2) given in homogeneous coordinates.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError('Input must be a 3x2 matrix')
    line1 = np.cross(points1[:, 0], points1[:, 1])
    line2 = np.cross(points2[:, 0], points2[:, 1])
    intersection = np.cross(line1, line2)
    return pflat(intersection)

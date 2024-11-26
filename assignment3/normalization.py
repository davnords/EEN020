import numpy as np

def compute_normalization_matrix(coords):
    """
    Computes the normalization matrix for a set of 2D homogeneous coordinates.

    Parameters:
        coords: np.ndarray of shape (3, n) - Homogeneous coordinates of points.
        
    Returns:
        N: np.ndarray of shape (3, 3) - Normalization matrix.
    """
    assert coords.shape[0] == 3, "Input should be in homogeneous coordinates (3 rows)."
    
    # Convert homogeneous coordinates to Cartesian
    cartesian_coords = coords[:2] / coords[2]
    
    # Compute mean and standard deviation
    mean = np.mean(cartesian_coords, axis=1)  # Shape (2,)
    std_dev = np.std(cartesian_coords, axis=1)  # Shape (2,)
    
    # Construct normalization matrix
    scale = np.diag([1/std_dev[0], 1/std_dev[1], 1])
    translation = np.array([
        [1, 0, -mean[0]],
        [0, 1, -mean[1]],
        [0, 0, 1]
    ])
    
    # Combine scaling and translation
    N = scale @ translation
    
    return N

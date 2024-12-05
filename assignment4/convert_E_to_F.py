import numpy as np

def convert_E_to_F(E, K1, K2):
    """
    Converts an essential matrix to a fundamental matrix using the calibration matrices.

    Parameters:
    - E (numpy.ndarray): The 3x3 essential matrix.
    - K1 (numpy.ndarray): The 3x3 calibration matrix of the first camera.
    - K2 (numpy.ndarray): The 3x3 calibration matrix of the second camera.

    Returns:
    - F (numpy.ndarray): The 3x3 fundamental matrix.
    """
    # Ensure the inputs are numpy arrays
    E = np.asarray(E)
    K1 = np.asarray(K1)
    K2 = np.asarray(K2)
    
    # Compute the fundamental matrix
    F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
    
    return F

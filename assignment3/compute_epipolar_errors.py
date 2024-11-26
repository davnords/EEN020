from matplotlib import pyplot as plt
from point_line_distances_2D import point_line_distance_2D
import numpy as np

def compute_epipolar_errors(F, x1s, x2s):
    if x1s.shape[0] != 3 or x2s.shape[0] != 3:
        raise ValueError("x1s and x2s should have shape (3, N)")
    assert np.allclose(x1s[-1], 1), 'Points are not normalized'
    assert np.allclose(x2s[-1], 1), 'Points are not normalized'
    _, n = x2s.shape
    lines = F@x1s
    errors = []
    for i in range(n):
        d = point_line_distance_2D(x2s[:2, i], lines[:, i])
        errors.append(d)

    plt.hist(errors, bins=100)
    
    return errors

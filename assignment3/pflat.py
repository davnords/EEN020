import numpy as np
import matplotlib.pyplot as plt

def pflat(mat: np.ndarray)->np.ndarray:
    assert mat[-1].all() != 0, 'Last row of matrix cannot be zero'
    return mat/mat[-1]


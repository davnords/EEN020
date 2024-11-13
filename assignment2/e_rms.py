import numpy as np

def e_rms(x, x_proj):
    """Calculate the root mean square error between the projected points and the original points.
    Arguments:
    x -- 2D points in the image (2xN)
    x_proj -- Projected 2D points in the image (2xN)
    """
    if x.shape[0] != 2 or x_proj.shape[0] != 2:
        raise ValueError('Input must be a 2xN matrix')
    if x.shape[1] != x_proj.shape[1]:
        raise ValueError('The number of points in x and x_proj must be the same')
    N = x.shape[1]
    e = np.linalg.norm(x - x_proj, ord='fro')
    e_rms = np.sqrt(np.sum(e**2) / N)
    return e_rms
import numpy as np
from pflat import pflat
import matplotlib.pyplot as plt

def psphere(x):
    """Normalizes each column of a 3xN matrix to lie on the unit sphere in 2D space."""
    norms = np.linalg.norm(x[:2, :], axis=0)
    return x / norms

def compute_line(p1, p2):
    """Computes the line between two points in homogeneous coordinates."""
    return np.cross(p1, p2)

def rital(linjer, st='-'):
    """Plots a 2D line between two points given in homogeneous coordinates."""
    if linjer.shape[1] != 2:
      raise ValueError('Input must be a 3x2 matrix')

    _, n = linjer.shape
    line = compute_line(linjer[:, 0], linjer[:, 1])
    
    rikt = psphere(np.array([[line[1]], [-line[0]], [0]]))
    punkter = pflat(np.cross(rikt.T, line).reshape(-1, 1))

    x_coords = [punkter[0, 0] - 2000 * rikt[0, 0], punkter[0, 0] + 2000 * rikt[0, 0]]
    y_coords = [punkter[1, 0] - 2000 * rikt[1, 0], punkter[1, 0] + 2000 * rikt[1, 0]]
    plt.plot(x_coords, y_coords, st)

  
  
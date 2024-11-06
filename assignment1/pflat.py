import numpy as np
import matplotlib.pyplot as plt

def pflat(mat: np.ndarray)->np.ndarray:
    assert mat[-1].all() != 0, 'Last row of matrix cannot be zero'
    return mat/mat[-1]

def plot_points_2d(mat: np.ndarray, path='plot/img.png')-> None:
    plt.plot(mat[0], mat[1], 'ro')
    plt.savefig(path)
    plt.show()

def plot_points_3d(mat: np.ndarray, path='plot/img.png')-> None:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(mat[0], mat[1], mat[2], 'ro')

    plt.savefig(path)
    plt.show()


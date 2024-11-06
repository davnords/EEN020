from scipy.io import loadmat
from pflat import plot_points_2d, plot_points_3d, pflat

if __name__ == "__main__":
    arr = loadmat('A1data/compEx1.mat')['x2D']
    normalized_arr = pflat(arr)
    plot_points_2d(normalized_arr, path='plots/ce_1_2dplot.png')
    plot_points_3d(arr, path='plots/ce_1_3dplot.png')

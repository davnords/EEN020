from scipy.io import loadmat
from pflat import plot_points_2d, plot_points_3d, pflat
import os

if __name__ == "__main__":
    arr_2d = loadmat('A1data/compEx1.mat')['x2D']
    arr_3d = loadmat('A1data/compEx1.mat')['x3D']
    normalized_arr_2d = pflat(arr_2d)
    normalized_arr_3d = pflat(arr_3d)

    if not os.path.exists('plots'):
        os.makedirs('plots')    

    plot_points_2d(normalized_arr_2d, path='plots/compEx1_plot1.png')
    plot_points_3d(normalized_arr_3d, path='plots/compEx1_plot2.png')

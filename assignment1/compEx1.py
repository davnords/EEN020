from scipy.io import loadmat
from pflat import plot_points_2d, plot_points_3d, pflat
import os

if __name__ == "__main__":
    arr = loadmat('A1data/compEx1.mat')['x2D']
    normalized_arr = pflat(arr)

    if not os.path.exists('plots'):
        os.makedirs('plots')    

    plot_points_2d(normalized_arr, path='plots/compEx1_plot1.png')
    plot_points_3d(arr, path='plots/compEx1_plot2.png')

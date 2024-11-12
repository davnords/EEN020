from PIL import Image
from scipy.io import loadmat
import os
from plotcams import plotcams
import numpy as np
import matplotlib.pyplot as plt
from pflat import pflat
from project_and_plot import project_and_plot

def find_non_nan_points(x, X):
    non_nan_indices = np.where(np.all(~np.isnan(x), axis=0))[0]
    print('Share of non-NaN points in the chosen camera:', len(non_nan_indices)/n)
    return X[:, non_nan_indices]

if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')    
    
    mat = loadmat('A2data/data/compEx1data.mat')
    X = pflat(mat['X'])
    x = mat['x'][0]
    P = mat['P'][0]
    imfiles = mat['imfiles'][0]

    m = len(P)
    n = X.shape[1]

    # Plot 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(P, ax=ax)
    plt.axis('equal')
    plt.title('3D points and cameras')
    plt.savefig('plots/compEx1_plot1.png')
    # plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------
    camera_choice = 0
    x0 = x[camera_choice]
    P0 = P[camera_choice]
    im0 = Image.open('A2data/data/'+str(imfiles[camera_choice][0]))
    X0 = find_non_nan_points(x0, X)
    x0_proj = pflat(P0 @ X0)

    # Plot 2
    project_and_plot(P0, X0, im0, 'plots/compEx1_plot2.png', title='3D projection into camera 0', show=False)    

    # ---------------------------------------------------------------------------------------------

    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1/16, 1/16, 0, 1]
    ])

    T2 = np.array([
        [1, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 1, 0],
        [1/9, 1/9, 0, 1]
    ])

    X1 = pflat(T1 @ X)
    X2 = pflat(T2 @ X)

    # Plot 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X1[0], X1[1], X1[2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(P, ax=ax)
    plt.axis('equal')
    plt.title('3D points and cameras with transformation T1')
    plt.savefig('plots/compEx1_plot3.png')
    # plt.show()
    plt.close()

    # Plot 4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X2[0], X2[1], X2[2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(P, ax=ax)
    plt.axis('equal')
    plt.title('3D points and cameras with transformation T2')
    plt.savefig('plots/compEx1_plot4.png')
    # plt.show()
    plt.close()

    # ---------------------------------------------------------------------------------------------
    camera_choice = 5
    P1 = P[camera_choice]
    x1 = x[camera_choice]

    X1 = find_non_nan_points(x1, X1)
    print(X1)
    X2 = find_non_nan_points(x1, X2)
    print(X2)
    im1 = Image.open('A2data/data/'+str(imfiles[camera_choice][0]))

    # Plot 5
    project_and_plot(P1, X1, im1, 'plots/compEx1_plot5.png', title='T1 transformed 3D projection into camera 1', show=True)    
    
    # Plot 6
    project_and_plot(P1, X2, im1, 'plots/compEx1_plot6.png', title='T2 transformed 3D projection into camera 1', show=True)    
    
    
from PIL import Image
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from camera_center_and_axis import  camera_center_and_axis
from math import *

if __name__ == "__main__":
    img = Image.open('A1data/compEx4.jpg')
    mat = loadmat('A1data/compEx4.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)
    v = mat['v']

    T = np.column_stack((np.eye(3), np.zeros((3, 1))))
    P = np.dot(K,T)
    C, a = camera_center_and_axis(P)
    corners = mat['corners']

    # Plot 1
    plt.axis('equal')
    plt.plot(corners[0], corners[1], 'ro')
    plt.imshow(img.convert("RGB"))
    plt.savefig('plots/compEx4_plot1.png')
    # plt.show()
    plt.close()
    
    # Plot 2
    corners = np.dot(K_inv, corners)
    plt.axis('equal')
    plt.plot(corners[0], corners[1], 'ro')
    plt.savefig('plots/compEx4_plot2.png')
    # plt.show()
    plt.close()

    # Plot 3
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(corners[0], corners[1], corners[2], 'ro')
    C = np.array([0, 0, 0])
    a = np.array([0,0, 1])
    ax.quiver(C[1], C[1], C[2], a[0], a[1], a[2], length=0.5, color='b')

    t = np.linspace(0, 5, 100)
    for corner in corners.T:
        line_points = C[:, None] + t * corner[:, None]
        ax.plot3D(line_points[0], line_points[1], line_points[2], 'g-', label="Line")

        scale= -v[3]/np.dot(v[:3].T, corner)
        x, y, z = scale*corner
        print('The point in the plane are: ', x, y, z)
        ax.plot3D(x, y, z, 'go')

    v0, v1, v2, v3 = v.flatten()  # Flatten to get 1D array
    x_vals = np.linspace(-5, 5, 50)  # Adjust range and number of points as needed
    y_vals = np.linspace(-5, 5, 50)  # Adjust range and number of points as needed
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Plane equation: v0 * x + v1 * y + v2 * z + v3 = 0 -> z = -(v0 * x + v1 * y + v3) / v2
    if v2 != 0:  # Avoid division by zero
        z_grid = -(v0 * x_grid + v1 * y_grid + v3) / v2
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=100, cstride=100, color='y', label="Plane")
    plt.savefig('plots/compEx4_plot3.png')
    # plt.show()
    plt.close()

    R2 = np.array([
        [cos(pi/5), 0, -sin(pi/5)],
        [0, 1, 0],
        [sin(pi/5), 0, cos(pi/5)]
    ])
    
    t2 = np.array([
        [-2.5],
        [0],
        [0]
    ])
    T2 = np.hstack((R2, t2) )
    P2 = np.dot(K, T2)

    C2 = t2
    # Compute principal axis a2 (third column of R2)
    a2 = R2[:, 2]



    # Plot 4

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.axis('equal')
    ax.plot3D(corners[0], corners[1], corners[2], 'ro')
    ax.quiver(C[0], C[1], C[2], a[0], a[1], a[2], length=1, color='b')
    ax.quiver(C2[0], C2[1], C2[2], a2[0], a2[1], a2[2], length=1, color='b')

    plt.savefig('plots/compEx4_plot4.png')
    # plt.show()
    plt.close()
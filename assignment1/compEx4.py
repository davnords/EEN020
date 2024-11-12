from PIL import Image
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from camera_center_and_axis import  camera_center_and_axis
from math import *
import cv2
from imwarp import imwarp
from pflat import pflat

if __name__ == "__main__":
    img = Image.open('A1data/compEx4.jpg')
    mat = loadmat('A1data/compEx4.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)
    v = pflat(mat['v'])

    T = np.column_stack((np.eye(3), np.zeros((3, 1))))
    P = np.dot(K,T)
    C, a = camera_center_and_axis(P)
    corners = mat['corners']

    # Plot 1
    plt.axis('equal')
    plt.plot(corners[0], corners[1], 'ro')
    plt.plot([corners[0][0], corners[0][1]],
            [corners[1][0], corners[1][1]], 'b-')  # Line from corner 1 to corner 2
    plt.plot([corners[0][1], corners[0][2]],
            [corners[1][1], corners[1][2]], 'b-')  # Line from corner 2 to corner 3
    plt.plot([corners[0][2], corners[0][3]],
            [corners[1][2], corners[1][3]], 'b-')  # Line from corner 3 to corner 4
    plt.plot([corners[0][3], corners[0][0]],
            [corners[1][3], corners[1][0]], 'b-')  # Line from corner 4 to corner 1
    plt.imshow(img.convert("RGB"))
    plt.title('Arnold with corners')
    plt.savefig('plots/compEx4_plot1.png')
    plt.show()
    plt.close()
    
    # Plot 2
    corners = np.dot(K_inv, corners)
    plt.axis('equal')
    plt.plot(corners[0], corners[1], 'ro')
    plt.title('Corners in normalized coordinates')
    plt.savefig('plots/compEx4_plot2.png')
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------

    # Plot 3
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(corners[0], corners[1], corners[2], 'ro')
    C = np.array([0, 0, 0])
    a = np.array([0,0, 1])
    ax.quiver(C[1], C[1], C[2], a[0], a[1], a[2], length=0.5, color='b')

    # Calculating the points in the plane based on Exercise 8
    t = np.linspace(0, 5, 100)
    corners_3d = []
    for corner in corners.T:
        line_points = C[:, None] + t * corner[:, None]
        ax.plot3D(line_points[0], line_points[1], line_points[2], 'g-', label="Line")

        scale= -v[3]/np.dot(v[:3].T, corner)
        x, y, z = scale*corner
        corners_3d.append([x, y, z])
        ax.plot3D(x, y, z, 'go')

    corners_3d = np.array(corners_3d).T
    print('The points in the plane are: \n', corners_3d)

    # Plot the plane
    v0, v1, v2, v3 = v.flatten()  # Flatten to get 1D array
    x_vals = np.linspace(-5, 5, 50)  # Adjust range and number of points as needed
    y_vals = np.linspace(-5, 5, 50)  # Adjust range and number of points as needed
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Plane equation: v0 * x + v1 * y + v2 * z + v3 = 0 -> z = -(v0 * x + v1 * y + v3) / v2
    if v2 != 0:  # Avoid division by zero
        z_grid = -(v0 * x_grid + v1 * y_grid + v3) / v2
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=100, cstride=100, color='y', label="Plane")
    plt.title('The points on the plane mapping to the corners')
    plt.savefig('plots/compEx4_plot3.png')
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------

    # Setting up the second camera
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
    
    # Defining the camera centres and principle axes
    C1, a1 = np.array([0, 0, 0]), np.array([0,0, 1])
    C2, a2 = t2, R2.T @ np.array([0,0, 1])

    # Plot 4
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.axis('equal')
    
    for corner in corners_3d.T:
        ax.plot3D(corner[0], corner[1], corner[2], 'ro')

    ax.quiver(C1[0], C1[1], C1[2], a[0], a[1], a[2], length=1, color='b')
    ax.quiver(C2[0], C2[1], C2[2], a2[0], a2[1], a2[2], length=1, color='b')
    if v2 != 0:  # Avoid division by zero
        z_grid = -(v0 * x_grid + v1 * y_grid + v3) / v2
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=100, cstride=100, color='y', label="Plane")

    plt.title('Camera centres and principle axes')
    plt.savefig('plots/compEx4_plot4.png')
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------

    # Computing the homography
    
    H = (R2 - np.dot(t2, v[:3].T))
    shifted_corners = pflat(np.dot(H, corners))
    assert shifted_corners[:, 2].all() == 1, 'The third coordinate of the shifted corners is not 1'

    P2 = np.dot(K, np.column_stack((R2, t2)))

    corners_3d_h = np.vstack((corners_3d, np.ones((1, 4))))
    shifted_corners_camera = pflat(np.dot(P2, corners_3d_h))  # Project 3D corners to 2D image coordinates
    shifted_corners_camera = np.dot(K_inv, shifted_corners_camera)

    assert np.allclose(shifted_corners_camera, shifted_corners), 'The projected corners are not the same'
    print('Check: The projected corners are the same')

    # Plot 5
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    for i, sc in enumerate([shifted_corners, shifted_corners_camera]):
        if i == 0:
            ax[i].set_title('Shifted corners by homography')
        else:
            ax[i].set_title('Shifted corners through camera projection')
        ax[i].axis('equal')
        ax[i].plot(sc[0], sc[1], 'ro')
        ax[i].plot([sc[0][0], sc[0][1]],
                [sc[1][0], sc[1][1]], 'b-')  # Line from corner 1 to corner 2
        ax[i].plot([sc[0][1], sc[0][2]],
                [sc[1][1], sc[1][2]], 'b-')  # Line from corner 2 to corner 3
        ax[i].plot([sc[0][2], sc[0][3]],
                [sc[1][2], sc[1][3]], 'b-')  # Line from corner 3 to corner 4
        ax[i].plot([sc[0][3], sc[0][0]],
                [sc[1][3], sc[1][0]], 'b-')  # Line from corner 4 to corner 1

    plt.tight_layout()
    plt.title('Comparison of using homography and camera projection')
    plt.savefig('plots/compEx4_plot5.png')
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------

    Htot = np.dot(np.dot(K, H), K_inv)
    original_corners = mat['corners']

    im = cv2.imread('A1data/compEx4.jpg')
    transformed_image, transformed_corners = imwarp(
        im, original_corners, Htot
    )

    # Plot 6
    plt.axis('equal')
    plt.imshow(transformed_image)
    plt.plot(transformed_corners[0], transformed_corners[1], 'ro')
    plt.title('Warped image and points with translated coordinate system')
    plt.savefig('plots/compEx4_plot6.png')
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------------------

    # Plot 7
    plt.axis('equal')
    transformed_corners = pflat(np.dot(Htot, original_corners))
    plt.plot(transformed_corners[0], transformed_corners[1], 'ro')
    plt.title('Actual result of Htot @ corners')
    plt.savefig('plots/compEx4_plot7.png')
    plt.show()
    plt.close()


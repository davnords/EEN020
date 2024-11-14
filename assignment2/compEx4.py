from pflat import pflat
import numpy as np
from triangulate_3D_point_DLT import triangulate_3D_point_DLT
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat

if __name__ == "__main__":
    mat = loadmat('A2data/data/compEx2data.mat')
    Xmodel = mat['Xmodel']
    x1 = np.load('variables/x1_compEx3.npy')
    x2 = np.load('variables/x2_compEx3.npy')
    x = np.array([x1, x2])
    (P1, P2) = np.load('variables/compEx2_P_matrices.npy')
    (K1, K2) = np.load('variables/compEx2_K_matrices.npy')
    (img1, img2) = [Image.open('A2data/data/cube1.JPG'), Image.open('A2data/data/cube2.JPG')]
    images = [img1, img2]
    
    assert x1.shape[0] == x2.shape[0], "Number of points in x1 and x2 do not match"
    assert x1.shape[-1] == 2, "x1 does not have 2 coordinates"

    assert P1.shape == (3, 4), "P1 is not a 3x4 matrix"
    assert P2.shape == (3, 4), "P2 is not a 3x4 matrix"

    n = x1.shape[0]
    X = []
    sv = []
    for i in range(n):
        xp1 = x1[i, :]
        xp2 = x2[i, :]
        Xi, svi = triangulate_3D_point_DLT(xp1, xp2, P1, P2)
        X.append(Xi)
        sv.append(svi)

    print('Mean singular value for optimization: ', np.mean(sv))
    X = np.array(X)
    X = np.vstack((X.T, np.ones(X.shape[0])))

    X_unnormalized = X.copy()

    x1_proj = pflat( P1 @ X )
    x2_proj = pflat( P2 @ X )

    height1, width1 = img1.size[1], img1.size[0]  # Dimensions of the first image
    height2, width2 = img2.size[1], img2.size[0]  # Dimensions of the second image

    # Removing points outside the image bounds
    x1_proj = pflat(P1 @ X)
    x2_proj = pflat(P2 @ X)

    # Get indices of points within bounds for both projections
    in_bounds_x1 = (0 <= x1_proj[0]) & (x1_proj[0] < width1) & (0 <= x1_proj[1]) & (x1_proj[1] < height1)
    in_bounds_x2 = (0 <= x2_proj[0]) & (x2_proj[0] < width2) & (0 <= x2_proj[1]) & (x2_proj[1] < height2)

    # Find points that are within bounds in both projections
    valid_indices = in_bounds_x1 & in_bounds_x2

    # Filter points based on the valid indices
    x1_proj = x1_proj[:, valid_indices]
    x2_proj = x2_proj[:, valid_indices]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    for i in range(2):
        if i == 0:
            xi, xi_proj = x1, x1_proj
        else:
            xi, xi_proj = x2, x2_proj
        ax[i].set_title('Camera '+str(i+1))
        ax[i].imshow(images[i].convert("RGB"))
        ax[i].plot(xi_proj[0], xi_proj[1], 'ro', alpha=0.3, markersize=0.5, label='Projected triangulated 3D points')
        ax[i].plot(xi[:, 0], xi[:, 1], 'bo', alpha=0.3, markersize=0.5, label='SIFT 2D keypoints')
        ax[i].legend(loc='upper right', fontsize=10, markerscale=10)

    plt.suptitle('Comparing SIFT 2D keypoints with triangulated 3D points')
    plt.savefig('plots/compEx4_plot1.png')
    # plt.show()
    plt.close()

    x1_normalized = pflat(np.linalg.inv(K1) @ np.vstack((x1.T, np.ones(x1.shape[0]))))
    x2_normalized = pflat(np.linalg.inv(K2) @ np.vstack((x2.T, np.ones(x2.shape[0]))))
    x_normalized = np.array([x1_normalized, x2_normalized])
    
    P1_normalized = np.linalg.inv(K1) @ P1
    P2_normalized = np.linalg.inv(K2) @ P2

    X = []
    sv = []
    for i in range(n):
        Xi, svi = triangulate_3D_point_DLT(x1_normalized[:2, i].T, x2_normalized[:2, i].T, P1_normalized, P2_normalized)
        X.append(Xi)
        sv.append(svi)
    
    print('Mean singular value for normalized optimization: ', np.mean(sv))
    X = np.vstack((np.array(X).T, np.ones(n)))

    P_matrices = np.array([P1, P2])
    x1_normalized_proj = pflat( P1 @ X )
    x2_normalized_proj = pflat( P2 @ X )
    x_normalized_proj = np.array([x1_normalized_proj, x2_normalized_proj])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    for i in range(2):
        xi_proj = x_normalized_proj[i]
        ax[i].set_title('Camera '+str(i+1))
        ax[i].imshow(img1.convert("RGB"))
        ax[i].plot(xi_proj[0], xi_proj[1], 'ro', alpha=0.3, markersize=0.5, label='Projected triangulated 3D points')
        ax[i].plot(x[i][:, 0], x[i][:, 1], 'bo', alpha=0.3, markersize=0.5, label='SIFT 2D keypoints')
        ax[i].legend(loc='upper right', fontsize=10, markerscale=10)

    plt.suptitle('Comparing SIFT 2D keypoints with NORMALIZED triangulated 3D points')
    plt.savefig('plots/compEx4_plot2.png')
    # plt.show()
    plt.close()

    # Computing the average pixel error 
    errors1 = np.linalg.norm(x1 - x1_normalized_proj[:2].T, axis=1)
    errors2 = np.linalg.norm(x2 - x2_normalized_proj[:2].T, axis=1)

    print('Average pixel error for camera 1: ', np.mean(errors1))
    print('Average pixel error for camera 2: ', np.mean(errors2))

    error_threshold = 3
    indices = np.where((errors1 < error_threshold) & (errors2 < error_threshold))[0]

    print(f"Share of points with pixel error less than {error_threshold} pixels: {(len(indices)/n)*100:.2f}%")

    X = X[:, indices]

    from plotcams import plotcams
    # Plot 3
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X[0], X[1], X[2], 'bo', alpha=0.5, markersize=0.4, label='Triangulated 3D points')
    ax.plot3D(Xmodel[0], Xmodel[1], Xmodel[2], 'ro', alpha=0.9, markersize=0.4, label='Real 3D points')
    ax.legend(loc='upper right', fontsize=10, markerscale=10)
    plotcams(P_matrices, ax=ax, scale=10)
    plt.title('3D points')
    plt.savefig('plots/compEx4_plot3.png')
    plt.show()
    plt.close()
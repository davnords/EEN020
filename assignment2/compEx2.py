from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np 
from estimate_camera_DLT import estimate_camera_DLT
from PIL import Image
from pflat import pflat
from plotcams import plotcams

def normalize_coordinates(coords, normalize=True):
    """
    Normalize the pixel values (x, y) to have mean 0 and std 1.

    Args:
    - coords (numpy array): A 3xN array of homogeneous coordinates.
    - normalize (bool): Whether to normalize the coordinates. If False, returns identity matrix.

    Returns:
    - normalized_coords (numpy array): Normalized coordinates.
    - N (numpy array): The normalization matrix.
    """
    # Extract x and y coordinates (ignore the homogeneous coordinate, assume it's the last row)
    x_coords = coords[0, :]
    y_coords = coords[1, :]
    
    # Compute mean and standard deviation
    x_mean, y_mean = np.mean(x_coords), np.mean(y_coords)
    x_std, y_std = np.std(x_coords), np.std(y_coords)
    
    # Construct the normalization matrix N
    if normalize:
        # Scaling and translation to normalize to mean 0 and std 1
        N = np.array([
            [1/x_std, 0, -x_mean/x_std],
            [0, 1/y_std, -y_mean/y_std],
            [0, 0, 1]
        ])
        # Normalize the coordinates
        normalized_coords = N @ coords
    else:
        # Identity matrix for no normalization
        N = np.eye(3)
        normalized_coords = coords
    
    return normalized_coords, N

if __name__ == "__main__":
    mat = loadmat('A2data/data/compEx2data.mat')
    x = mat['x'][0]
    x = np.array([x[0], x[1]])
    Xmodel = mat['Xmodel']
    Xmodel_homogeneous = np.vstack((Xmodel, np.ones(Xmodel.shape[1]))) # Make homogeneous
    startind = mat['startind']
    endind = mat['endind']

    include_normalization = True

    images = [Image.open('A2data/data/cube1.JPG'), Image.open('A2data/data/cube2.JPG')]
    
    x_normalized = []
    T = []
    for i in range(2):
        coords = pflat(np.vstack((x[i][0, :], x[i][1, :], x[i][2, :])))
        coords_normalized, Ti = normalize_coordinates(coords, normalize=include_normalization)
        x_normalized.append(coords_normalized)
        T.append(Ti)

    x_normalized = np.array(x_normalized)
    T = np.array(T)

    # Only first two dimension because the third dimension is always 1
    if include_normalization:
        assert np.allclose(np.mean(x_normalized, axis=-1)[:, :2], 0), "Mean of normalized x is not zero"
        assert np.allclose(np.std(x_normalized, axis=-1)[:, :2], 1), "Standard deviation of normalized x is not one"
        print('Assertion tests passed, mean and standard deviation of normalized x are correct')

    # Plot 1
    plt.scatter(x_normalized[0, 0, :], x_normalized[0, 1, :], label='Image 1', color='blue')
    plt.scatter(x_normalized[1, 0, :], x_normalized[1, 1, :], label='Image 2', color='red')
    plt.title('Normalized Points')
    plt.xlabel('Normalized x')
    plt.ylabel('Normalized y')
    plt.axis('equal')  # To keep the aspect ratio the same
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/compEx2_plot1.png')
    # plt.show()
    plt.close()

    image_points = []
    P_matrices = []

    # Plot 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    for i in range(2):
        ax[i].set_title('Points projected into camera ' + str(i+1))

        P_normalized = estimate_camera_DLT(x_normalized[i], Xmodel_homogeneous, printSV=True)
        P_denormalized = np.linalg.inv(T[i]) @ P_normalized
        P_matrices.append(P_denormalized)

        Xi = np.dot(P_denormalized, Xmodel_homogeneous)  # Project the 3D points into the camera
        Xi = Xi / Xi[-1]  # Normalize the points
        image_points.append(Xi)

        # Plot 2
        ax[i].imshow(images[i].convert("RGB"))
        ax[i].scatter(Xi[0], Xi[1], label='Projected 3D points', color='blue')
    plt.savefig('plots/compEx2_plot2.png')
    # plt.show()
    plt.close()

    # Plot 3
    image_points = np.array(image_points)
    P_matrices = np.array(P_matrices)
    plt.scatter(image_points[0, 0, :], image_points[0, 1, :], label='Image 1', color='blue')
    plt.scatter(image_points[1, 0, :], image_points[1, 1, :], label='Image 2', color='red')
    plt.title('Plotting points in the same figure')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/compEx2_plot3.png')
    # plt.show()
    plt.close()

    # Plot 4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xmodel[0], Xmodel[1], Xmodel[2], 'bo', alpha=0.5, markersize=0.7)
    plotcams(P_matrices, ax=ax, scale=10)
    plt.axis('equal')
    plt.title('3D points and cameras')
    plt.savefig('plots/compEx2_plot4.png')
    # plt.show()
    plt.close()

    from rq import rq
    K_matrices = []
    for i in range(2):
        R, Q = rq(P_matrices[0])
        R /= R[2, 2]
        assert np.allclose(R[2, 2], 1), "R[2, 2] is not 1"
        K_matrices.append(R)

    K_matrices = np.array(K_matrices)

    # Question for TAs: Why is the element in the middle of the top row of the calibration matrix not 0?
    print('Calibration matrix for the first camera: \n', R) 

    import os
    if not os.path.exists('variables'):
        os.makedirs('variables')   

    # Save the P matrices
    np.save('variables/compEx2_P_matrices.npy', P_matrices)
    np.save('variables/compEx2_K_matrices.npy', K_matrices)

    # ---------------------------------------------------------------------------------------------
    # Optional part
    from e_rms import e_rms

    selected_points = np.array([1, 4, 13, 16, 25, 28, 31])
    
    points_options = [False, True]
    normalization_options = [False, True]

    for points_option in points_options:
        for normalization_option in normalization_options:
            coords = pflat(np.vstack((x[0][0, :], x[0][1, :], x[0][2, :])))
            X = Xmodel_homogeneous
            camera_coords = coords
            if points_option:
                camera_coords = coords[:, selected_points]
                X = Xmodel_homogeneous[:, selected_points]

            camera_coords, T = normalize_coordinates(camera_coords, normalize=normalization_option)
            if normalization_option:
                assert np.allclose(np.mean(camera_coords, axis=-1)[:2], 0), "Mean of normalized x is not zero"
                assert np.allclose(np.std(camera_coords, axis=-1)[:2], 1), "Standard deviation of normalized x is not one"
            P_normalized = estimate_camera_DLT(camera_coords, X)
            P_denormalized = np.linalg.inv(T) @ P_normalized

            x_proj = np.dot(P_denormalized, Xmodel_homogeneous)  # Project the 3D points into the camera
            x_proj = x_proj / x_proj[-1]  # Normalize the points

            e_rms_value = e_rms(coords[:2], x_proj[:2])
            print(f"The root mean square error {'with' if normalization_option else 'without'} normalization and {'with' if points_option else 'without'} only selected points is: {e_rms_value:.4f}")




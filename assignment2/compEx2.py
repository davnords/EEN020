from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np 
from estimate_camera_DLT import estimate_camera_DLT
from PIL import Image
from plotcams import plotcams

def normalize_coordinates(coords, normalize=True):
    if not normalize:
        return coords, np.eye(3)
    mean = np.mean(coords, axis=1, keepdims=True)
    std_dev = np.std(coords, axis=1, keepdims=True)
    print('Mean x:', mean[0, 0])
    print('Mean y:', mean[1, 0])
    print('Standard deviation x:', std_dev[0, 0])
    print('Standard deviation y:', std_dev[1, 0])
    normalized_coords = (coords - mean) / std_dev
    T = np.array([[1 / std_dev[0, 0], 0, -mean[0, 0] / std_dev[0, 0]],
                  [0, 1 / std_dev[1, 0], -mean[1, 0] / std_dev[1, 0]],
                  [0, 0, 1]])
    return normalized_coords, T

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
        x_coords = x[i][0, :]
        y_coords = x[i][1, :]
        coords = np.vstack((x_coords, y_coords))
        coords_normalized, Ti = normalize_coordinates(coords, normalize=include_normalization)
        coords_normalized = np.vstack((coords_normalized, np.ones(coords_normalized.shape[1])))  # Make homogeneous
        x_normalized.append(coords_normalized)
        T.append(Ti)

    x_normalized = np.array(x_normalized)
    T = np.array(T)

    # Only first two dimension because the third dimension is always 1
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

        P_normalized = estimate_camera_DLT(x_normalized[i], Xmodel_homogeneous)
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
    plt.show()
    plt.close()

    from rq import rq
    R, Q = rq(P_matrices[0])
    R /= R[2, 2]
    assert np.allclose(R[2, 2], 1), "R[2, 2] is not 1"

    # Question for TAs: Why is the element in the middle of the top row of the calibration matrix not 0?
    print('Calibration matrix for the first camera: \n', R) 

    import os
    if not os.path.exists('variables'):
        os.makedirs('variables')   

    # Save the P matrices
    np.save('variables/compEx2_P_matrices.npy', P_matrices)



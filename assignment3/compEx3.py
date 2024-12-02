import numpy as np
from enforce_essential import enforce_essential
from extract_P_from_E import extract_P_from_E
from triangulate_3D_point_DLT import triangulate_3D_point_DLT
from scipy.io import loadmat
from pflat import pflat
from PIL import Image
from matplotlib import pyplot as plt
from plotcams import plotcams

if __name__ == "__main__":
    E = np.load('variables/E.npy')
    enforce_essential(E)

    mat = loadmat('./A3data/compEx1data.mat')
    mat2 = loadmat('./A3data/compEx2data.mat')
    im1, im2 = Image.open('./A3data/kronan1.JPG'), Image.open('./A3data/kronan2.JPG')

    K = mat2['K']
    K_inv = np.linalg.inv(K)

    x = mat['x']
    x = np.array([pflat(x[0][0]), pflat(x[1][0])])
    _, _, N = x.shape
    x_unnormalized = x.copy()
    x = np.array([K_inv@x[0], K_inv@x[1]])
    
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    P2_best = None
    X_best = None
    for P2 in P2s:
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = x[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1

        # Taking only P2 that has all points in front of both cameras
        if positive_depth_count == N:
            X_best = Xj
            P2_best = P2

    if P2_best is None or X_best is None:
        raise ValueError('No valid P2 found')

    X = np.array(X_best).T
    P2 = P2_best

    # Deciding best matrix / points
    X = np.vstack((X, np.ones(X.shape[1]))) # Make homogeneous

    # Removing normalization
    P1 = K@P1
    P2 = K@P2

    # Ensuring last coordinate is 1
    x1_proj = pflat(P1@X)

    plt.figure()
    plt.title('[...]')
    plt.plot(x_unnormalized[0, 0, :], x_unnormalized[0, 1, :], 'ro', alpha=0.3, markersize=1, label='Original points')
    plt.plot(x1_proj[0, :], x1_proj[1, :], 'bo', alpha=0.3, markersize=1, label='Reprojected 3D points')
    plt.legend(loc='upper right', fontsize=10, markerscale=10)
    plt.imshow(im1.convert("RGB"))
    plt.savefig('./plots/compEx3_plot1.png')
    plt.show()

    # Plot 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2], 'bo', alpha=0.2, markersize=0.5)
    plotcams([P1, P2], ax=ax, scale=3)
    plt.axis('equal')
    plt.title('[...]')
    plt.savefig('plots/compEx3_plot2.png')
    plt.show()
    plt.close()




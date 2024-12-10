from scipy.io import loadmat
import numpy as np
from pflat import pflat
from matplotlib import pyplot as plt
from levenberg_marquardt import linearize_reproj_err, compute_update, compute_total_reprojection_error, compute_reprojection_error, perform_bundle_adjustment
from tqdm import tqdm

if __name__ == "__main__":
    mat = loadmat('./A4data/compEx3data.mat')
    P = mat['P']
    P = np.array([P[0][0], P[0][1]])
    x = mat['x']
    x = np.array([pflat(x[0][0]), pflat(x[0][1])])
    X = pflat(mat['X'])
    N = X.shape[-1]

    X_lm = perform_bundle_adjustment(X, x, P, epochs=5)

    err = compute_total_reprojection_error(P[0], P[1], X, x)
    print(f"Total reprojection error before BA: {np.sum(err)}")
    print(f"Median reprojection error before BA: {np.median(err)}")

    err = compute_total_reprojection_error(P[0], P[1], X_lm, x)
    print(f"Total reprojection error after BA: {np.sum(err)}")
    print(f"Median reprojection error after BA: {np.median(err)}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0, :], X[1, :], X[2, :], 'bo', alpha=0.2, markersize=0.5, label='Original 3D points')
    ax.plot(X_lm[0, :], X_lm[1, :], X_lm[2, :], 'ro', alpha=0.2, markersize=0.5, label='3D points after optimization')
    ax.legend(loc='upper right', fontsize=10, markerscale=10)
    plt.axis('equal')
    plt.title('Comparison before and after bundle adjustment')
    plt.savefig(f"plots/compEx3_plot1.png")
    plt.show()
    plt.close()

    

    

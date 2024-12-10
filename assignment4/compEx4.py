import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pflat import pflat
from levenberg_marquardt import perform_bundle_adjustment, compute_total_reprojection_error
from tabulate import tabulate

if __name__ == "__main__":
    # Load data
    mat = loadmat('./A4data/compEx3data.mat')
    P = np.array([mat['P'][0][0], mat['P'][0][1]])
    x = np.array([pflat(mat['x'][0][0]), pflat(mat['x'][0][1])])
    X = pflat(mat['X'])

    # Add noise
    sigma_X = np.random.normal(0, 0.1, X.shape)
    sigma_x = np.random.normal(0, 3, x.shape)
    X_combinations = [X, X + sigma_X]
    x_combinations = [x, x + sigma_x]

    # Initialize results table
    table_data = []
    headers = [
        "X Type", "x Type", "Total Error Before BA", "Median Error Before BA",
        "Total Error After BA", "Median Error After BA"
    ]

    # Perform bundle adjustment and calculate errors
    for X_idx, X_variant in enumerate(X_combinations):
        for x_idx, x_variant in enumerate(x_combinations):
            X_lm = perform_bundle_adjustment(X_variant, x_variant, P, epochs=5)
            err_before = compute_total_reprojection_error(P[0], P[1], X_variant, x_variant)
            err_after = compute_total_reprojection_error(P[0], P[1], X_lm, x_variant)

            table_data.append([
                f"X{' + noise' if X_idx == 1 else ''}",
                f"x{' + noise' if x_idx == 1 else ''}",
                np.sum(err_before),
                np.median(err_before),
                np.sum(err_after),
                np.median(err_after)
            ])

            # Plot original and optimized 3D points for each case
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(X_variant[0, :], X_variant[1, :], X_variant[2, :], c='b', alpha=0.6, s=10, label='3D points before BA (noisy)')
            ax.scatter(X_lm[0, :], X_lm[1, :], X_lm[2, :], c='r', alpha=0.6, s=10, label='3D points after BA')

            ax.set_title(f'3D Points: {f"X + noise" if X_idx == 1 else "X"}, {f"x + noise" if x_idx == 1 else "x"}', fontsize=14)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.legend(loc='upper right', fontsize=10)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/compEx3_3D_comparison_X{X_idx}_x{x_idx}.png")
            plt.show()

    # Print results table
    print(tabulate(table_data, headers=headers, floatfmt=".4f"))
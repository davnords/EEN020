import numpy as np
from tqdm import tqdm

def perform_bundle_adjustment(X, xs, Ps, epochs=5, mu=0.1):
    """
    Perform bundle adjustment for multiple cameras and 3D points.

    Args:
        Xs (list of np.ndarray): List of 3D points in homogeneous coordinates, each of shape (4, n_points).
        xs (list of np.ndarray): List of observed 2D points, each of shape (2, n_points).
        Ps (list of np.ndarray): List of camera projection matrices, each of shape (3, 4).
        epochs (int): Number of optimization iterations.
        mu (float): Damping parameter for Levenberg-Marquardt.

    Returns:
        list of np.ndarray: Optimized projection matrices.
    """
    for epoch in tqdm(range(epochs), desc='Performing Bundle Adjustment...'):
        errors = []
        for i, (P, x) in enumerate(zip(Ps, xs)):
            R, t = P[:3, :3], P[:, 3].reshape(-1, 1)
            proj = R@X + t
            proj = proj/proj[-1, :]
            r = (proj-x)[:2, :].reshape(-1, 1)

            N = X.shape[-1]
            c = (R[2, :]@X).flatten()+t[-1]

            J1 = np.hstack((
                1 / c,
                np.zeros((N))
            ))
            J2 = np.hstack((
                np.zeros((N)),
                1 / c,
            ))
            J3 = np.hstack((
                -(R[0, :]@X+t[0])/c,
                -(R[1, :]@X+t[1])/c,
            ))
            J = np.vstack((J1, J2, J3)).T

            delta_t = -np.linalg.inv(J.T @J+mu*np.eye(3))@J.T @r
            errors.append((r**2).sum().item())
            print(errors)

            Ps[i][:, 3] += delta_t.flatten()

        print(f"Full reprojection error in epoch {epoch + 1}/{epochs} is: {np.sum(errors):.4f}")

    return Ps
import numpy as np
from tqdm import tqdm

def perform_translation_bundle_adjustment(Xs, xs, Ps, epochs=5, mu=0.1):
    """
    Perform bundle adjustment for multiple cameras and 3D points.

    Args:
        Xs (list of np.ndarray): List of 3D points, each of shape (3, n_points).
        xs (list of np.ndarray): List of observed 2D homogenous points, each of shape (3, n_points).
        Ps (list of np.ndarray): List of camera projection matrices, each of shape (3, 4).
        epochs (int): Number of optimization iterations.
        mu (float): Damping parameter for Levenberg-Marquardt.

    Returns:
        list of np.ndarray: Optimized projection matrices.
    """
    for epoch in tqdm(range(epochs), desc='Performing Bundle Adjustment...'):
        errors = []
        for i, (P, x, X) in enumerate(zip(Ps, xs, Xs)):
            R, t = P[:3, :3], P[:, 3].reshape(-1, 1)
            proj = R@X + t
            proj = proj/proj[-1, :]
            r = (proj-x)[:2, :].reshape(-1, 1)

            N = X.shape[-1]
            c = (R[2, :]@X).flatten()+t[-1]

            J1, J2, J3 = create_translation_jacobians(N, R, X, c, t)
            J = np.vstack((J1, J2, J3)).T

            delta_t = -np.linalg.inv(J.T @J+mu*np.eye(3))@J.T @r
            errors.append((r**2).sum().item())

            Ps[i][:, 3] += delta_t.flatten()

        print(f"Full reprojection error in epoch {epoch + 1}/{epochs} is: {np.sum(errors):.4f}")

    return Ps

def perform_extrinsic_bundle_adjustment(Xs, xs, Ps, epochs=20, mu=0.1):
    """
    Perform bundle adjustment for multiple cameras and 3D points.

    Args:
        Xs (list of np.ndarray): List of 3D points, each of shape (3, n_points).
        xs (list of np.ndarray): List of observed 2D homogenous points, each of shape (3, n_points).
        Ps (list of np.ndarray): List of camera projection matrices, each of shape (3, 4).
        epochs (int): Number of optimization iterations.
        mu (float): Damping parameter for Levenberg-Marquardt.

    Returns:
        list of np.ndarray: Optimized projection matrices.
    """
    S1, S2, S3 = create_S_matrices()
    mus = [mu for _ in Ps]
    for epoch in tqdm(range(epochs), desc='Performing Extrinsic Bundle Adjustment...'):
        errors = []
        for i, (P, x, X, mu) in enumerate(zip(Ps, xs, Xs, mus)):
            R, t = P[:3, :3], P[:, 3].reshape(-1, 1)

            r = compute_residual(R, t, X, x)
            current_error = (r**2).sum().item()

            N = X.shape[-1]
            c = (R[2, :]@X).flatten()+t[-1]

            J1, J2, J3 = create_translation_jacobians(N, R, X, c, t)
            J4 = create_rotation_parameter_jacobian(S1, R, t, X, c)
            J5 = create_rotation_parameter_jacobian(S2, R, t, X, c)
            J6 = create_rotation_parameter_jacobian(S3, R, t, X, c)
            J = np.vstack((J1, J2, J3, J4, J5, J6)).T

            delta = -np.linalg.inv(J.T @J+mu*np.eye(6))@J.T@r
            delta_t = delta[:3]
            a = delta[3:6]

            R = (np.eye(3)+(a[0]*S1+a[1]*S2+a[2]*S3))@R
            t += delta_t

            r = compute_residual(R, t, X, x)
            new_error = (r**2).sum().item()

            if new_error < current_error:
                # Accept the update and reduce the damping factor
                Ps[i] = np.hstack((R, t))
                mus[i] = max(mu / 10, 1e-7)  # Ensure mu doesn't become too small
            else:
                # Reject the update and increase the damping factor
                mus[i] = min(mu * 10, 1e7)  # Ensure mu doesn't become too large
            errors.append((r**2).sum().item())

        print(f"Full reprojection error in epoch {epoch + 1}/{epochs} is: {np.sum(errors):.4f}")

    return Ps

def compute_residual(R, t, X, x):
    proj = project_points(R, t, X)
    r = (proj-x)[:2, :].reshape(-1, 1)
    return r

def project_points(R, t, X):
    proj = R@X + t
    return proj/proj[-1, :]

def create_rotation_parameter_jacobian(S, R, t, X, c):
    J = np.hstack((
        (S[0, :]@R@X)/c-((R[0,:]@X+t[0])/(c**2))*(S[2, :]@R@X),
        (S[1, :]@R@X)/c-((R[1,:]@X+t[1])/(c**2))*(S[2, :]@R@X),
    ))
    return J

def create_translation_jacobians(N, R, X, c, t):
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
    return J1, J2, J3

def create_S_matrices():
    S1 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    S2 = np.array([
        [0, 0, -1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    S3 = np.array([
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    return S1, S2, S3

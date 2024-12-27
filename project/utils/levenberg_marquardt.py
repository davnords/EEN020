import numpy as np
from tqdm import tqdm

def compute_JP(P, X):
    """
    Compute the Jacobian with respect to the camera parameters for all 3D points.

    Args:
        P (np.ndarray): Projection matrix, shape (3, 4).
        X (np.ndarray): 3D points in homogeneous coordinates, shape (4, n).

    Returns:
        np.ndarray: Jacobian matrix, shape (2n, 12).
    """
    n = X.shape[1]  # Number of points

    u = P[0, :] @ X  # u-coordinate numerator, shape (n,)
    v = P[1, :] @ X  # v-coordinate numerator, shape (n,)
    w = P[2, :] @ X  # Denominator, shape (n,)

    # Compute Jacobian rows for all points
    J1 = np.hstack([
        -X.T / w[:, np.newaxis],                 # First row, first 4 elements
        np.zeros((n, 4)),                       # First row, next 4 elements
        (u / w**2)[:, np.newaxis] * X.T         # First row, last 4 elements
    ])

    J2 = np.hstack([
        np.zeros((n, 4)),                       # Second row, first 4 elements
        -X.T / w[:, np.newaxis],                # Second row, next 4 elements
        (v / w**2)[:, np.newaxis] * X.T         # Second row, last 4 elements
    ])

    # Stack the Jacobian rows for all points
    J = np.vstack([J1, J2])
    return J


def compute_R(P, X, x):
    """
    Compute the residuals for all 3D points.

    Args:
        P (np.ndarray): Projection matrix, shape (3, 4).
        X (np.ndarray): 3D points in homogeneous coordinates, shape (4, n).
        x (np.ndarray): Observed 2D points, shape (2, n).

    Returns:
        np.ndarray: Residuals, shape (2n,).
    """
    u = P[0, :] @ X  # u-coordinate numerator, shape (n,)
    v = P[1, :] @ X  # v-coordinate numerator, shape (n,)
    w = P[2, :] @ X  # Denominator, shape (n,)

    # Compute residuals for all points
    r = np.hstack([
        x[0, :] - u / w,  # Residual for u-coordinates
        x[1, :] - v / w   # Residual for v-coordinates
    ])
    return r

def linearize_reproj_err_P(Ps, xs, Xs):
    """
    Linearize reprojection error for all cameras.

    Args:
        Ps (list of np.ndarray): List of projection matrices, each of shape (3, 4).
        xs (list of np.ndarray): List of observed 2D points, each of shape (2, n_points).
        Xs (list of np.ndarray): List of 3D points in homogeneous coordinates, each of shape (4, n_points).

    Returns:
        tuple: 
            - List of residuals, one per camera, each of shape (2 * n_points,).
            - List of Jacobians, one per camera, each of shape (2 * n_points, 12).
    """
    residuals = [compute_R(P, X, x) for P, x, X in zip(Ps, xs, Xs)]
    jacobians = [compute_JP(P, X) for P, X in zip(Ps, Xs)]
    return residuals, jacobians

def compute_update_Ps(residuals, jacobians, mu):
    """
    Compute updates for all cameras.

    Args:
        residuals (list of np.ndarray): List of residuals, one per camera.
        jacobians (list of np.ndarray): List of Jacobians, one per camera.
        mu (float): Damping parameter for Levenberg-Marquardt.

    Returns:
        list of np.ndarray: Updates for all projection matrices, each of shape (3, 4).
    """
    delta_Ps = []
    for r, J in zip(residuals, jacobians):
        # Compute the Hessian approximation for this camera
        C = J.T @ J + mu * np.eye(J.shape[1])
        
        # Compute the gradient for this camera
        c = J.T @ r
        
        # Solve for the update vector
        delta = -np.linalg.solve(C, c)
        
        # Reshape delta into a (3, 4) matrix
        delta_P = delta.reshape(3, 4)
        delta_Ps.append(delta_P)

    return delta_Ps

def perform_bundle_adjustment(Xs, xs, Ps, epochs=5, mu=0.1):
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
        # Linearize reprojection error
        residuals, jacobians = linearize_reproj_err_P(Ps, xs, Xs)
        
        # Compute updates for all cameras
        delta_Ps = compute_update_Ps(residuals, jacobians, mu)
        
        # Update each camera's projection matrix
        for i in range(len(Ps)):
            Ps[i] += delta_Ps[i]
        print(f"Mean reprojection error in epoch {epoch + 1}/{epochs} is: {np.mean([np.linalg.norm(r)**2/len(r) for r in residuals])}")

    return Ps
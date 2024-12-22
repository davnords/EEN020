from tqdm import tqdm
import numpy as np

def estimate_T_robust(xs, Xs, R, eps, iterations=10):
    # Here need to implement DLT for 2 point correspondence to estimate T (since R is known)
    best_inliers = []
    best_T = None

    for i in tqdm(range(iterations), desc="RANSAC iterations"):
        # Randomly select 2 correspondences
        randind = np.random.choice(xs.shape[1], size=2, replace=False)
        x, X = xs[:, randind], Xs[:, randind]
        
        # Construct the M matrix based on the 2D-3D correspondences
        M = []
        for j in range(2):  # Two correspondences
            Xj = X[:, j]
            xj = x[:, j]
            # Form the system based on the equation: x = R * X + T
            row1 = np.hstack([R[0], -xj[0] * R[0], -Xj[0], -Xj[1], -Xj[2]])
            row2 = np.hstack([R[1], -xj[1] * R[1], -Xj[0], -Xj[1], -Xj[2]])
            M.append(row1)
            M.append(row2)
        
        M = np.array(M)
        
        # Solve the system using SVD to find the null space
        _, _, Vt = np.linalg.svd(M)
        T_est = Vt[-1, :3]  # The translation is the last column of Vt
        
        # Reproject 3D points using the estimated translation
        x_projected = R @ Xs[:3, :] + T_est[:, np.newaxis]
        x_projected /= x_projected[2, :]  # Normalize by homogeneous coordinate
        
        # Calculate the reprojection error (Euclidean distance)
        errors = np.linalg.norm(x_projected[:2, :] - xs[:2, :], axis=0)
        
        # Find inliers based on the error threshold
        inliers = np.where(errors < eps)[0]
        
        # Update best translation if we have more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_T = T_est

    return best_T, best_inliers

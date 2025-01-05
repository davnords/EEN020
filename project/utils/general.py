from tqdm import tqdm
import numpy as np
from .parallell_RANSAC import parallell_RANSAC
from .triangulate_3D_point_DLT import triangulate_3D_point_DLT
from .extract_P_from_E import extract_P_from_E
from .estimate_E_robust import estimate_E_robust
from .pflat import pflat
from PIL import Image
import cv2

# Load SIFT and naive matcher (correspondences come from RoMa matching)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

def make_homogenous(x):
    return np.vstack((x.T, np.ones(x.shape[0])))

def find_matches(imA_path, imB_path, model, K_inv, device='cuda:0'):
    imA = Image.open(imA_path) 
    imB = Image.open(imB_path) 

    imA_dims = (imA.height, imA.width)
    imB_dims = (imB.height, imB.width)

    H_A, W_A = imA_dims
    H_B, W_B = imB_dims
    
    warp, certainty = model.match(imA_path, imB_path, device=device)
    matches, certainty = model.sample(warp, certainty)
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()

    x1u, x2u = make_homogenous(kptsA), make_homogenous(kptsB)
    x1n, x2n = pflat(K_inv @ x1u), pflat(K_inv @ x2u)
    return x1n, x2n


def find_relative_rotations(image_paths, K_inv, model, eps, device='cuda:0'):
    out = []
    for i in range(len(image_paths)-1):
        imA_path = image_paths[i]
        imB_path = image_paths[i+1]
        x1n, x2n = find_matches(imA_path, imB_path, model, K_inv, device=device)
        R = parallell_RANSAC(x1n, x2n, eps, iterations=100)[:3, :3]
        out.append(R)
    return out

def upgrade_to_absolute_rotations(relative_rotations):
    """
    Upgrade relative rotations to absolute rotations.
    
    Parameters:
    - relative_rotations: List of 3x3 relative rotation matrices.
    
    Returns:
    - absolute_rotations: List of 3x3 absolute rotation matrices.
    """
    absolute_rotations = [np.eye(3)]  # First camera is the global reference frame
    for R_rel in relative_rotations:
        # Compute the absolute rotation by chaining the previous absolute rotation with the relative rotation
        absolute_rotations.append(absolute_rotations[-1] @ R_rel)
    
    return absolute_rotations

def triangulate_initial_points(x1n, x2n, eps, R_rel):
    """
    Triangulate initial 3D points using the DLT method.
    """
    # R_rel could be useful later as then we can triangulate the points in the right reference frame with right orientation

    E, _ = estimate_E_robust(x1n, x2n, eps, iterations=100)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    xn = np.array([x1n, x2n])

    N = x1n.shape[-1]
    
    Xjs = []
    depth_counts = []
    for P2 in tqdm(P2s, desc='Triangulating points...'):
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = xn[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1
        Xj = np.array(Xj)
        Xjs.append(Xj)
        depth_counts.append(positive_depth_count)
    
    depth_counts = np.array(depth_counts)
    highest_depth_count = np.argmax(depth_counts)
    return Xjs[highest_depth_count]

def perform_initial_scene_reconstruction(reference_imA_path, reference_imB_path, K_inv, epipolar_treshold, init_pair, absolute_rotations):
    """
    Perform initial scene reconstruction using the reference images.
    Matching is done using SIFT and the triangulation is done using the DLT method.
    """
    kp1, des1 = sift.detectAndCompute(cv2.imread(reference_imA_path,cv2.IMREAD_GRAYSCALE),None)
    kp2, des2 = sift.detectAndCompute(cv2.imread(reference_imB_path,cv2.IMREAD_GRAYSCALE),None)

    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    x1u = make_homogenous(np.float32([kp1[m.queryIdx].pt for m in good]))  # Points from first image
    x2u = make_homogenous(np.float32([kp2[m.trainIdx].pt for m in good]))  # Points from second image
    x1n, x2n = pflat(K_inv@x1u), pflat(K_inv@x2u)

    # Get the descriptors corresponding ONLY to the matched points
    good_indices = [m.queryIdx for m in good]
    descX = des1[good_indices]  # These descriptors now correspond to the points we triangulated

    # Find the relative rotation between the two images (using the already computed rotations)
    R_rel = absolute_rotations[init_pair[1]-1] @ absolute_rotations[init_pair[0]-1].T 

    X0 = triangulate_initial_points(x1n, x2n, epipolar_treshold, R_rel)
    X0 = X0.T
    R_ref_to_world = absolute_rotations[init_pair[0]-1][:3, :3]
    X0 = R_ref_to_world@X0

    X, indices = remove_3D_outliers(X0.T)
    X = X.T
    descX = descX[indices]

    return X, descX

def remove_3D_outliers(X, percentile=90):
    # Compute distances from the origin
    distances = np.linalg.norm(X, axis=1)

    # Find the 90th percentile distance
    threshold = np.percentile(distances, percentile)

    indices= distances <= threshold

    # Filter out points beyond the threshold
    X = X[indices]
    return X, indices

def reduced_camera_resectioning(X, descX, absolute_rotations, image_paths, K_inv, translation_threshold):
    Ps = []
    xs = []
    Xs = []
    from utils.estimate_T_robust import estimate_T_robust
    for i in range(len(image_paths)):
        R = absolute_rotations[i]
        objectPoints, imagePoints = match_3d_to_image(X, descX, image_paths[i])
        x2n = pflat(K_inv@make_homogenous(imagePoints))
        Ps.append(estimate_T_robust(x2n, make_homogenous(objectPoints.T), R, translation_threshold, iterations=500))
        xs.append(x2n)
        Xs.append(objectPoints)
    return Ps, xs, Xs

def triangulate_scene(image_paths, Ps, roma_model, K_inv, device='cuda:0'):
    X = []
    for i in tqdm(range(len(image_paths)-1), desc='Performing triangulation...'):
        imA_path = image_paths[i]
        imB_path = image_paths[i+1]

        P1 = Ps[i]
        P2 = Ps[i+1]

        x1n, x2n = find_matches(imA_path, imB_path, roma_model, K_inv, device=device)
        for j in range(x1n.shape[-1]):
            Xi, _ = triangulate_3D_point_DLT(x1n[:2, j], x2n[:2, j], P1, P2)
            X.append(Xi)
    
    X = np.array(X)
    X, _ = remove_3D_outliers(X)
    return X

def match_3d_to_image(X, descX, image_path):
    kp_new, desc_new = sift.detectAndCompute(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), None)
    matches = bf.knnMatch(descX, desc_new, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    X_indices = [m.queryIdx for m in good_matches]
    image_points = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
    matched_3d_points = X[:, X_indices]

    return matched_3d_points, image_points
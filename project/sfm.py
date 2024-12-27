from utils.get_dataset_info import get_dataset_info
import cv2 as cv
import numpy as np
from utils import general, pflat, estimate_T_robust, levenberg_marquardt, triangulate_3D_point_DLT, viz
from romatch import roma_outdoor
import torch
import os
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------
# Questions
# ------------------------------------------------------------------------------------------------
# - Can I change so RoMa extracts the features instead of SiFT? Feels much more compelling
# - When should I choose only the camera version with the 3D points in front of them? Lol

# Left to do:
# - Implement the 2 point DLT resectioning (when R is known) with RANSAC
# - Change Bundle Adjustment to adjust the translation vectors (camera centers) rather than the 3D points
# - Test on more datasets?
# - Implement 5 point algorithm
# - Implement some of the algorithms in CUDA

# Optional:
# - Implementation of 5-point solver in COLMAP: https://github.com/colmap/colmap/blob/main/src/colmap/estimators/essential_matrix.h
#   (-) Can we implement this in CUDA?

def sfm(dataset):
    # ------------------------------------------------------------------------------------------------
    # (0) Load the data and the models
    # ------------------------------------------------------------------------------------------------
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)
    K_inv = np.linalg.inv(K)

    # Load SIFT and naive matcher (correspondences come from RoMa matching)
    sift = cv.SIFT_create()
    bf = cv.BFMatcher(normType=cv.NORM_L2, crossCheck=True)

    # Ensure directories exist for this dataset
    os.makedirs(f"./storage/{dataset}", exist_ok=True)

    # Load images
    image_paths = ['./project_data/'+img_name for img_name in img_names]

    # Normalize pixel threshold
    focal_length = (K[0,0] + K[1 ,1])/2 # Average over the diagonal
    epipolar_treshold =          pixel_threshold / focal_length
    homography_threshold =   3 * pixel_threshold / focal_length
    translation_threshold  = 3 * pixel_threshold / focal_length

    # Initialize RoMa model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    roma_model = roma_outdoor(device=device)

    # ------------------------------------------------------------------------------------------------
    # (1) Calculate relative orientations (R_i, i+1 | T_i, i+1) between images i and i+1 
    # ------------------------------------------------------------------------------------------------
    try:
        relative_rotations = np.load(f"./storage/{dataset}/relative_rotations.npy")
    except:
        relative_rotations = []
        for i in range(len(image_paths)-1):
            imA_path = image_paths[i]
            imB_path = image_paths[i+1]

            x1u, x2u = general.find_matches(imA_path, imB_path, roma_model, device=device)

            # Better if we can replace with RoMa features
            _, desc1 = sift.compute(cv.imread(imA_path,cv.IMREAD_GRAYSCALE), [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in x1u.T])
            _, desc2 = sift.compute(cv.imread(imB_path,cv.IMREAD_GRAYSCALE), [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in x2u.T])

            np.save(f"./storage/{dataset}/desc{i}.npy", desc1)
            np.save(f"./storage/{dataset}/desc{i+1}.npy", desc2)

            x1n = pflat.pflat(K_inv @ x1u)
            x2n = pflat.pflat(K_inv @ x2u)

            R = general.find_relative_rotation_and_translation(x1n, x2n, epipolar_treshold)[:3, :3]
            relative_rotations.append(R)
            np.save(f"./storage/{dataset}/relative_rotations.npy", relative_rotations)
    
    # ------------------------------------------------------------------------------------------------
    # (2) Upgrade to absolute rotations R_i
    # ------------------------------------------------------------------------------------------------
    absolute_rotations = general.upgrade_to_absolute_rotation(relative_rotations)

    # ------------------------------------------------------------------------------------------------
    # (3) Reconstruct initial 3D points from an initial image pair i_1 and i_2
    # ------------------------------------------------------------------------------------------------

    imA_path = image_paths[init_pair[0]-1]
    imB_path = image_paths[init_pair[1]-1]

    x1u, x2u = general.find_matches(imA_path, imB_path, roma_model, device=device)
    _, descX = sift.compute(cv.imread(imA_path,cv.IMREAD_GRAYSCALE), [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in x1u.T])

    x1n = pflat.pflat(K_inv @ x1u)
    x2n = pflat.pflat(K_inv @ x2u)

    X0 = general.triangulate_initial_points(x1n, x2n, epipolar_treshold)

    # Rotate X0 to world coordinate frame
    X0 = X0@absolute_rotations[init_pair[0]-1].T

    # ------------------------------------------------------------------------------------------------
    # (4) For each image i robustly calculate the camera center Ci / translation Ti...
    # ...Reduced Camera Resectioning problem (since Ri is known at this stage)
    # ------------------------------------------------------------------------------------------------
    Ps = []
    Xs = []
    xs = []
    for i in range(len(image_paths)):
        desci = np.load(f"./storage/{dataset}/desc{i}.npy")
        matches = bf.match(desci, descX)
        matches_2d_3d = np.array([(m.queryIdx, m.trainIdx) for m in matches])
        x = x1n[:, matches_2d_3d[:, 0]]
        X = X0.T[:, matches_2d_3d[:, 1]]
        
        R = absolute_rotations[i]
        # Make this work, is it the absolute rotations that are bad or the algorithm??
        T, _ = estimate_T_robust.estimate_T_robust(x, X, R, translation_threshold, iterations=10)
        P = np.hstack((R, T))
        Ps.append(P)
        Xs.append(general.make_homogenous(X.T))
        xs.append(x)

    X0, _ = general.remove_3D_outliers(X0)
    viz.plot_scene(X0, Ps, name=f"scene_reconstruction_before_BA_{dataset}.png")
    # ------------------------------------------------------------------------------------------------
    # (5) Refine camera centers (or translation vectors) using Levenberg-Marquardt method...
    # ...here we instead optimize the entire pose (including rotation matrix)
    # ------------------------------------------------------------------------------------------------
    Ps = levenberg_marquardt.perform_bundle_adjustment(Xs, xs, Ps)
    
    viz.plot_scene(X0, Ps, name=f"scene_reconstruction_after_BA_{dataset}.png")
    # ------------------------------------------------------------------------------------------------
    # (6) Triangulate points for all pairs (i, i+1) and visualize 3D points + cameras
    # ------------------------------------------------------------------------------------------------
    X = []
    for i in tqdm(range(len(image_paths)-1), desc='Performing triangulation...'):
        imA_path = image_paths[i]
        imB_path = image_paths[i+1]

        P1 = Ps[i]
        P2 = Ps[i+1]

        x1u, x2u = general.find_matches(imA_path, imB_path, roma_model, device=device)
        x1n = pflat.pflat(K_inv @ x1u)
        x2n = pflat.pflat(K_inv @ x2u)
        N = x.shape[-1]
        for j in range(N):
            Xi, _ = triangulate_3D_point_DLT.triangulate_3D_point_DLT(x1n[:2, j], x2n[:2, j], P1, P2)
            X.append(Xi)

    X = np.array(X)
    X, _ = general.remove_3D_outliers(X)
    viz.plot_scene(X, Ps, name=f"scene_reconstruction_{dataset}.png")
    # ------------------------------------------------------------------------------------------------
    # (-1) End...
    # ------------------------------------------------------------------------------------------------
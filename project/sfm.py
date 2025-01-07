from utils.get_dataset_info import get_dataset_info
import numpy as np
from utils import general, levenberg_marquardt, viz
from romatch import roma_outdoor
import torch
import os

# ------------------------------------------------------------------------------------------------
# Questions
# ------------------------------------------------------------------------------------------------
# - I have this issue with triangulating initial 3D points that sometimes are orthogonal to the relative rotations...
#   ...I believe this is related to that I triangulate without using the relative rotations, so the points can end up with the wrong orientation
# - How to triangulate when you have the camera matrices? If i just append the coordinates I get too many points and some with wierd depths
# - Can I change so RoMa extracts the features instead of SiFT? Feels much more compelling

# Left to do:
# - Test on more datasets?
# - Implement 5 point algorithm
# - Implement some of the algorithms in CUDA (RANSAC or something)

# Optional:
# - Implementation of 5-point solver in COLMAP: https://github.com/colmap/colmap/blob/main/src/colmap/estimators/essential_matrix.h
#   (-) Can we implement this in CUDA?


# ------------------------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------------------------

def sfm(dataset):
    # ------------------------------------------------------------------------------------------------
    # (0) Load the data and the models
    # ------------------------------------------------------------------------------------------------
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)
    K_inv = np.linalg.inv(K)

    # Ensure directories exist for this dataset
    os.makedirs(f"./storage/{dataset}", exist_ok=True)

    # Load images
    image_paths = ['./project_data/'+img_name for img_name in img_names]

    # Normalize pixel threshold
    focal_length = (K[0,0] + K[1 ,1])/2 # Average over the diagonal
    epipolar_treshold      =     pixel_threshold / focal_length
    homography_threshold   = 3 * pixel_threshold / focal_length
    translation_threshold  = 3 * pixel_threshold / focal_length

    # Initialize RoMa model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    roma_model = roma_outdoor(device=device)

    # Initialize reference frames
    reference_imA_path = image_paths[init_pair[0]-1]
    reference_imB_path = image_paths[init_pair[1]-1]

    # ------------------------------------------------------------------------------------------------
    # (1) Calculate relative orientations (R_i, i+1 | T_i, i+1) between images i and i+1 
    # ------------------------------------------------------------------------------------------------
    try:
        relative_rotations = np.load(f"./storage/{dataset}/relative_rotations.npy")
    except:
        relative_rotations = general.find_relative_rotations(image_paths, K_inv, roma_model, epipolar_treshold, device=device)
        np.save(f"./storage/{dataset}/relative_rotations.npy", relative_rotations)
    
    # ------------------------------------------------------------------------------------------------
    # (2) Upgrade to absolute rotations R_i
    # ------------------------------------------------------------------------------------------------
    absolute_rotations = general.upgrade_to_absolute_rotations(relative_rotations)

    # ------------------------------------------------------------------------------------------------
    # (3) Reconstruct initial 3D points from an initial image pair i_1 and i_2
    # ------------------------------------------------------------------------------------------------

    X, descX = general.perform_initial_scene_reconstruction(reference_imA_path, reference_imB_path, K_inv, epipolar_treshold, init_pair, absolute_rotations)
    viz.plot_scene_old(X.T, [np.hstack((R, np.zeros((3,1)))) for R in absolute_rotations], name=f"{dataset}_absolute_rotations.png")
    # ------------------------------------------------------------------------------------------------
    # (4) For each image i robustly calculate the camera center Ci / translation Ti...
    # ...Reduced Camera Resectioning problem (since Ri is known at this stage)
    # ------------------------------------------------------------------------------------------------

    Ps, xs, Xs = general.reduced_camera_resectioning(X, descX, absolute_rotations, image_paths, K_inv, translation_threshold)

    # ------------------------------------------------------------------------------------------------
    # (5) Refine camera centers (or translation vectors) using Levenberg-Marquardt method...
    # ...(Optional) Nonlinear refinement of camera pose (max. 20 points)
    # ------------------------------------------------------------------------------------------------

    Ps = levenberg_marquardt.perform_extrinsic_bundle_adjustment(Xs, xs, Ps)
    viz.plot_scene_old(X.T, Ps, name=f"{dataset}_scene_reconstruction_after_BA.png")

    # ------------------------------------------------------------------------------------------------
    # (6) Triangulate points for all pairs (i, i+1) and visualize 3D points + cameras
    # ------------------------------------------------------------------------------------------------

    X = general.triangulate_scene(image_paths, Ps, roma_model, K_inv, device='cuda:0')
    viz.plot_scene_old(X, Ps, name=f"{dataset}_scene_reconstruction_after_full_triangulation.png")

    # ------------------------------------------------------------------------------------------------
    # (-1) End...
    # ------------------------------------------------------------------------------------------------
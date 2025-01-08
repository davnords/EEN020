from utils.get_dataset_info import get_dataset_info
import numpy as np
from utils import general, levenberg_marquardt, viz
from romatch import roma_outdoor
import torch
import os

# ------------------------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------------------------

def sfm(args):
    # ------------------------------------------------------------------------------------------------
    # (0) Load the data and the models
    # ------------------------------------------------------------------------------------------------
    dataset = args.dataset
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)
    K_inv = np.linalg.inv(K)

    # Ensure directories exist for this dataset
    os.makedirs(f"./storage/{dataset}", exist_ok=True)
    os.makedirs(f"./plots/{dataset}", exist_ok=True)

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
        relative_rotations = general.find_relative_rotations(image_paths, K_inv, roma_model, epipolar_treshold, device=device, matcher='sift')
        np.save(f"./storage/{dataset}/relative_rotations.npy", relative_rotations)
    
    # ------------------------------------------------------------------------------------------------
    # (2) Upgrade to absolute rotations R_i
    # ------------------------------------------------------------------------------------------------
    absolute_rotations = general.upgrade_to_absolute_rotations(relative_rotations)

    # ------------------------------------------------------------------------------------------------
    # (3) Reconstruct initial 3D points from an initial image pair i_1 and i_2
    # ------------------------------------------------------------------------------------------------

    X, descX = general.perform_initial_scene_reconstruction(reference_imA_path, reference_imB_path, K_inv, epipolar_treshold, init_pair, absolute_rotations, dataset)
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

    # ------------------------------------------------------------------------------------------------
    # (6) Triangulate points for all pairs (i, i+1) and visualize 3D points + cameras
    # ------------------------------------------------------------------------------------------------
    if args.plots == 'full':
        X = general.triangulate_scene(image_paths, Ps, roma_model, K_inv, device='cuda:0')
        viz.plot_scene(general.remove_3D_outliers(np.vstack(X))[0], Ps, name=f"./{dataset}/full_scene_reconstruction.png", title="Full Scene reconstruction SfM")
        viz.plot_colored_scene(X, Ps, name=f"./{dataset}/colored_scene_reconstruction.png", title="Colored Scene reconstruction SfM")

    x1n, x2n = general.find_matches(reference_imA_path, reference_imB_path, roma_model, K_inv, device='cuda:0', num=10000)
    
    viz.plot_scene(general.triangulate_correspondences(x1n, x2n, Ps[init_pair[0]-1], Ps[init_pair[1]-1]),
                        Ps, name=f"./{dataset}/single_view_scene_reconstruction.png",
                        title="Scene reconstruction from reference view")

    # ------------------------------------------------------------------------------------------------
    # (-1) End...
    # ------------------------------------------------------------------------------------------------
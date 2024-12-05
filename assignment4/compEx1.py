from PIL import Image
from scipy.io import loadmat
import numpy as np
from pflat import pflat
from estimate_F_DLT import estimate_F_DLT
from enforce_fundamental import enforce_fundamental
from enforce_essential import enforce_essential
from convert_E_to_F import convert_E_to_F
from matplotlib import pyplot as plt

if __name__ == "__main__":
    im1, im2 = Image.open('./A4data/round_church1.jpg'), Image.open('./A4data/round_church2.jpg')
    images = [im1, im2]
    mat = loadmat('./A4data/compEx1data.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)
    x = mat['x']
    x = np.array([pflat(x[0][0]), pflat(x[0][1])])

    x_normalized = np.array([pflat(K_inv@x[0]), pflat(K_inv@x[1])])
    E = estimate_F_DLT(x_normalized[0], x_normalized[1])
    enforce_fundamental(E)
    E = enforce_essential(E)

    U, S, Vt = np.linalg.svd(E)
    assert np.allclose(S, np.array([1, 1, 0])), 'Essential matrix is not valid!'

    F = convert_E_to_F(E, K, K)
    enforce_fundamental(F)

    epipolar_constraint = x[1].T@F@x[0].diagonal()
    assert np.allclose(epipolar_constraint, np.zeros_like(epipolar_constraint), atol=5-3), f'Epipolar constraint not fulfilled! Max deviation: {np.max(np.abs(epipolar_constraint))}'

    from compute_epipolar_errors import compute_epipolar_errors

    # Plot 1
    plt.figure()
    plt.title('Histogram of distances to epipolar lines (first image)')
    e1 = compute_epipolar_errors(F, x[0], x[1])
    print('Mean distance to epipolar lines (first image): ', np.mean(e1))
    plt.savefig('./plots/compEx1_plot1.png')
    # plt.show()
    plt.close()

    # Plot 2
    plt.figure()
    plt.title('Histogram of distances to epipolar lines (second image)')
    e2 = compute_epipolar_errors(F.T, x[1], x[0])
    print('Mean distance to epipolar lines (second image): ', np.mean(e2))
    plt.savefig('./plots/compEx1_plot2.png')
    # plt.show()
    plt.close()


    # Plot 3 & 4 
    from rital import rital
    for i in range(2):
        indices = np.random.choice(x[0].shape[-1], size=20, replace=False)
        sampled_points = np.array([pflat(x[0][:, indices]), pflat(x[1][:, indices])])
        sampled_points_x1 = sampled_points[0]
        sampled_points_x2 = sampled_points[1]
        if i == 0:    
            sampled_lines = F.T@sampled_points_x2
        else:
            sampled_lines = F@sampled_points_x1
            

        img_width = images[i].size[0]
        img_height = images[i].size[1]

        plt.figure()
        plt.title(f"Plotting 20 sampled points in image {i+1}")
        plt.plot(sampled_points[i, 0, :], sampled_points[i, 1, :], 'ro')
        plt.imshow(images[i].convert("RGB"))
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)  # Invert the y-axis to match image coordinates

        for line in sampled_lines.T:
            # Normalize the line
            line = line / np.sqrt(line[0]**2 + line[1]**2)
            
            # Find two points on the line that lie within the image bounds
            x_coords = np.array([0, img_width])
            y_coords = (-line[2] - line[0]*x_coords) / line[1]

            points = np.vstack((x_coords, y_coords, np.ones(2)))
            rital(points)

        plt.savefig(f"./plots/compEx1_plot{3+i}.png")
        # plt.show()
        plt.close()


    from estimate_E_robust import estimate_E_robust
    inlier_threshold_px = 2
    eps = inlier_threshold_px * 2 / ( K[0,0] + K[1 ,1])
    E, inliers = estimate_E_robust(x_normalized[0], x_normalized[1], eps, iterations=1)
    
    F = convert_E_to_F(E, K, K)
    enforce_fundamental(F)

     # Plot 5
    plt.figure()
    plt.title('Histogram of distances to epipolar lines (first image) for RANSAC')
    e1 = compute_epipolar_errors(F, x[0], x[1])
    print('Mean distance to epipolar lines (first image) for RANSAC: ', np.mean(e1))
    plt.savefig('./plots/compEx1_plot5.png')
    plt.show()
    plt.close()

    # Plot 6
    plt.figure()
    plt.title('Histogram of distances to epipolar lines (second image) for RANSAC')
    e2 = compute_epipolar_errors(F.T, x[1], x[0])
    print('Mean distance to epipolar lines (second image) for RANSAC: ', np.mean(e2))
    plt.savefig('./plots/compEx1_plot6.png')
    plt.show()
    plt.close()

    inliers_indices = np.where(inliers)[0]
    x_inliers = x[:, :, inliers_indices]

    # Plot 7 & 8 
    for i in range(2):
        indices = np.random.choice(x_inliers[0].shape[-1], size=20, replace=False)
        sampled_points = np.array([pflat(x_inliers[0][:, indices]), pflat(x_inliers[1][:, indices])])
        sampled_points_x1 = sampled_points[0]
        sampled_points_x2 = sampled_points[1]
        if i == 0:    
            sampled_lines = F.T@sampled_points_x2
        else:
            sampled_lines = F@sampled_points_x1
            

        img_width = images[i].size[0]
        img_height = images[i].size[1]

        plt.figure()
        plt.title(f"Plotting 20 sampled inliers points in image {i+1}  (RANSAC)")
        plt.plot(sampled_points[i, 0, :], sampled_points[i, 1, :], 'ro')
        plt.imshow(images[i].convert("RGB"))
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)  # Invert the y-axis to match image coordinates

        for line in sampled_lines.T:
            # Normalize the line
            line = line / np.sqrt(line[0]**2 + line[1]**2)
            
            # Find two points on the line that lie within the image bounds
            x_coords = np.array([0, img_width])
            y_coords = (-line[2] - line[0]*x_coords) / line[1]

            points = np.vstack((x_coords, y_coords, np.ones(2)))
            rital(points)

        plt.savefig(f"./plots/compEx1_plot{7+i}.png")
        plt.show()
        plt.close()


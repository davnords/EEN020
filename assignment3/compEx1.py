from scipy.io import loadmat
from pflat import pflat
import numpy as np
from normalization import compute_normalization_matrix
from estimate_F_DLT import estimate_F_DLT
from enforce_fundamental import enforce_fundamental
import random
from matplotlib import pyplot as plt
from PIL import Image
import os
from rital import rital
from compute_epipolar_errors import compute_epipolar_errors

if __name__ == "__main__":
    mat=loadmat('./A3data/compEx1data.mat')
    x = mat['x']
    x = np.array([pflat(x[0][0]), pflat(x[1][0])])

    if not os.path.exists('plots'):
        os.makedirs('plots')    

    N1 = compute_normalization_matrix(x[0])
    N2 = compute_normalization_matrix(x[1])

    x1_normalized = pflat(N1@x[0])
    x2_normalized = pflat(N2@x[1])

    x_normalized = np.array([x1_normalized, x2_normalized])
    
    F = estimate_F_DLT(x_normalized[0], x_normalized[1])
    F = N2.T@F@N1
    enforce_fundamental(F)

    assert np.allclose(np.mean(x_normalized[0], axis=0), 0, atol=3), "Mean of normalized x is not zero"
    assert np.allclose(np.mean(x_normalized[1], axis=0), 0, atol=3), "Mean of normalized x is not zero"
    assert np.allclose(np.std(x_normalized[0], axis=0), 1, atol=3), "Standard deviation of normalized x is not one"
    assert np.allclose(np.std(x_normalized[1], axis=0), 1, atol=3), "Standard deviation of normalized x is not one"

    # Verify epipolar constraint
    epipolar_constraint = x[0].T@F@x[1]
    assert np.allclose(epipolar_constraint, np.zeros_like(epipolar_constraint), atol=5e1), f'Epipolar constraint not fulfilled! Max deviation: {np.max(np.abs(epipolar_constraint))}'

    l = F @ x[0]

    indices = np.random.choice(x[0].shape[-1], size=20, replace=False)
    sampled_points = pflat(x[1][:, indices])
    sampled_lines = F@sampled_points

    im1, im2 = Image.open('./A3data/kronan1.JPG'), Image.open('./A3data/kronan2.JPG')

    img_width = im2.size[0]
    img_height = im2.size[1]

    plt.figure()
    plt.title('Plotting 20 sampled points in image 2')
    plt.plot(sampled_points[0, :], sampled_points[1, :], 'ro')
    plt.imshow(im2.convert("RGB"))

    for line in sampled_lines.T:
        # Normalize the line
        line = line / np.sqrt(line[0]**2 + line[1]**2)
        
        # Find two points on the line that lie within the image bounds
        x_coords = np.array([0, img_width])
        y_coords = (-line[2] - line[0]*x_coords) / line[1]

        points = np.vstack((x_coords, y_coords, np.ones(2)))
        rital(points)

    plt.savefig('./plots/compEx1_plot1.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Histogram of distances to epipolar lines')
    errors = compute_epipolar_errors(F, x[0], x[1])
    print('Mean distance to epipolar lines: ', np.mean(errors))
    plt.savefig('./plots/compEx1_plot2.png')
    plt.show()
    plt.close()

    # Non-normalized case

    x = mat['x']
    x = np.array([pflat(x[0][0]), pflat(x[1][0])])
    F = estimate_F_DLT(x[0], x[1])
    enforce_fundamental(F)

    epipolar_constraint = x[0].T@F@x[1].diagonal()
    assert np.allclose(epipolar_constraint, np.zeros_like(epipolar_constraint), atol=2e1), f'Epipolar constraint not fulfilled! Max deviation: {np.max(np.abs(epipolar_constraint))}'   
    errors = compute_epipolar_errors(F, x[0], x[1])

    print('Fundamental matrix F for the original (un-normalized) points', F/F[-1, -1])

    plt.close()
    plt.figure()
    plt.title('Histogram of distances to epipolar lines (non-normalized case)')
    errors = compute_epipolar_errors(F, x[0], x[1])
    print('Mean distance to epipolar lines (non-normalized): ', np.mean(errors))
    plt.savefig('./plots/compEx1_plot3.png')
    plt.show()
    plt.close()
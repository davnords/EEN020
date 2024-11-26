from scipy.io import loadmat
import numpy as np
from pflat import pflat
from estimate_F_DLT import estimate_F_DLT
from enforce_fundamental import enforce_fundamental
from PIL import Image
from matplotlib import pyplot as plt 
from rital import rital
from compute_epipolar_errors import compute_epipolar_errors
from enforce_essential import enforce_essential
from convert_E_to_F import convert_E_to_F

if __name__ == "__main__":
    mat1=loadmat('./A3data/compEx1data.mat')
    x = mat1['x']
    x = np.array([pflat(x[0][0]), pflat(x[1][0])])
    x_unnormalized = x.copy()

    mat = loadmat('./A3data/compEx2data.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)

    x = np.array([K_inv@x[0], K_inv@x[1]])
    F = estimate_F_DLT(x[0], x[1])
    enforce_fundamental(F)
    E = K.T@F@K
    U, S, Vt = np.linalg.svd(E)
    E = U@np.diag([1,1,0])@Vt

    np.save('variables/E.npy', E)

    epipolar_constraint = x[1].T@E@x[0]
    assert np.allclose(epipolar_constraint, np.zeros_like(epipolar_constraint), atol=5e-1), f'Epipolar constraint not fulfilled! Max deviation: {np.max(np.abs(epipolar_constraint))}'

    F = convert_E_to_F(E, K, K)
    F_unnormalized = K.T @ F @ K
    
    indices = np.random.choice(x_unnormalized[0].shape[-1], size=20, replace=False)
    sampled_points = pflat(x_unnormalized[1][:, indices])
    sampled_lines = F_unnormalized@sampled_points

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

    plt.savefig('./plots/compEx2_plot1.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Histogram of distances to epipolar lines')
    errors = compute_epipolar_errors(F, x[0], x[1])
    print('Mean distance to epipolar lines: ', np.mean(errors))
    plt.savefig('./plots/compEx1_plot2.png')
    plt.show()
    plt.close()
    
    enforce_essential(E)
    


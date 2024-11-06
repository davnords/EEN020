from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
from rital import rital, compute_line
import numpy as np 
from compute_intersection import compute_intersection
from point_line_distance_2D import point_line_distance_2D
import random

if __name__ == "__main__":
    img = Image.open('A1data/compEx2.jpg')
    H, W = img.size
    random_point = [random.randrange(0, H), random.randrange(0, W)]
    mat = loadmat('A1data/compEx2.mat')

    p1 = mat['p1']
    p2 = mat['p2']
    p3 = mat['p3']

    rital(p1)
    rital(p2)
    rital(p3)

    intersection = compute_intersection(p2, p3)
    plt.plot(intersection[0], intersection[1], 'ro')

    line1 = compute_line(p1[:, 0], p1[:, 1])
    d = point_line_distance_2D([intersection[0], intersection[1]], line1)
    d_random = point_line_distance_2D(random_point, line1)
    print('Distance between intersection and line 1:', d)
    print('Distance between random point and line 1:', d_random)

    plt.imshow(img.convert("RGB"))
    plt.savefig('plots/ce_2_plot_1.png')
    plt.show()
    
from PIL import Image
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = Image.open('A1data/compEx4.jpg')
    mat = loadmat('A1data/compEx4.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)
    v = mat['v']
    P = np.column_stack((K, np.zeros((3, 1))))
    corners = mat['corners']
    plt.axis('equal')
    plt.plot(corners[0], corners[1], 'ro')
    plt.imshow(img.convert("RGB"))
    plt.savefig('plots/compEx4_plot1.png')
    plt.show()

    corners = np.dot(K_inv, corners)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.plot(corners[0], corners[1], 'ro')
    plt.imshow(img.convert("RGB"))
    plt.savefig('plots/compEx4_plot2.png')
    plt.show()
from matplotlib import pyplot as plt
from plot_camera import plot_camera, plot_3d_points
from PIL import Image
from scipy.io import loadmat
from camera_center_and_axis import camera_center_and_axis
import numpy as np
from pflat import pflat

if __name__=="__main__":
    im1 = Image.open('A1data/compEx3im1.jpg')
    im2 = Image.open('A1data/compEx3im2.jpg')

    mat = loadmat('A1data/compEx3.mat')
        
    U = mat['U']
    U = pflat(U)

    P1 = mat['P1']
    P2 = mat['P2']

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plot_camera(P1, 5, ax)
    plot_camera(P2, 5, ax)
    plot_3d_points(U, ax)
    
    ax.view_init(elev=20, azim=30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Principal Axis')
    ax.legend()
    plt.savefig('plots/compEx3_plot1.png')
    plt.show()

    UP1 = pflat(np.dot(P1, U))
    UP2 = pflat(np.dot(P2, U))
    # create subplots for the images
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('Image 1')
    axs[1].set_title('Image 2')
    axs[0].plot(UP1[0], UP1[1], 'ro', alpha=0.2, markersize=0.5)
    axs[1].plot(UP2[0], UP2[1], 'ro', alpha=0.2, markersize=0.5)
    axs[0].imshow(im1.convert("RGB"))
    axs[1].imshow(im2.convert("RGB"))
    plt.savefig('plots/compEx3_plot2.png')
    plt.show()





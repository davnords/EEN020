from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

from .plotcams import plotcams

def plot_scene_old(Xs, Ps, title='Scene reconstruction SfM', name='scene_reconstruction.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(Ps, ax=ax, scale=0.5)
    plt.axis('equal')
    plt.title(title)
    plt.savefig(f"plots/{name}")
    plt.show()

def plot_scene(X, Ps, image_paths, scaling=0.5):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_xlim3d([-2, 4])
    # ax.set_ylim3d([-2, 4])
    # ax.set_zlim3d([-0.5, 15])

    ax.plot(X[0, :], X[1, :], X[2, :], 'bo', alpha=0.2, markersize=0.5)

    for P, image_path in zip(Ps, image_paths):
        img = np.array(Image.open(image_path))
        plotImage(ax, img, P, size=np.array((1, img.shape[0] / img.shape[1]))*scaling)

    plotPose(ax, np.eye(3), np.zeros((1,3)), scale=np.array([1, 1, 1]))
    plt.show()

def plotPose(ax, R, t, scale=np.array((1, 1, 1)), l_width=2, text=None):
    """
    plot an coordinate system to visualize Pose (R|t)
    
    ax      : matplotlib axes to plot on
    R       : Rotation as roation matrix
    t       : translation as np.array (1, 3)
    scale   : Scale as np.array (1, 3)
    l_width : linewidth of axis
    text    : Text written at origin
    """
    x_axis = np.array(([0., 0, 0], [1., 0, 0])) * scale
    y_axis = np.array(([0., 0, 0], [0., 1, 0])) * scale
    z_axis = np.array(([0., 0, 0], [0., 0, 1])) * scale

    x_axis += t
    y_axis += t
    z_axis += t

    x_axis = x_axis @ R
    y_axis = y_axis @ R
    z_axis = z_axis @ R

    ax.plot3D(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], color='red', linewidth=l_width)
    ax.plot3D(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], color='green', linewidth=l_width)
    ax.plot3D(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color='blue', linewidth=l_width)

    if (text is not None):
        ax.text(x_axis[0, 0], x_axis[0, 1], x_axis[0, 2], "red")

    return None


def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)

def plotImage(ax, img, P, size=np.array((1, 1)), scale=np.array((1, 1, 1)), img_scale=4):
    """
    Plot image plane in 3D with camera visualization
    Added show_camera parameter to toggle camera visualization
    """
    R = P[:3, :3]
    t = P[:, 3].reshape(1, 3)

    c = np.linalg.svd(P)[2][-1]  
    c /= c[3]
    
    z_axis = np.array(([0., 0, 0], [0., 0, 1])) * scale
    
    # Original image plotting code
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))
    corners = np.array(([0., 0, 0], [0, size[0], 0], 
                    [size[1], 0, 0], [size[1], size[0], 0]))

    camera_axis_length = 0.5
    
    # Calculate the center of the image plane
    center = np.array([size[1] / 2, size[0] / 2, 0])
    
    # Shift the corners so the image center is at (0, 0, 0)
    corners -= center

    # Apply 90-degree rotation to corners
    # R_rotate = np.array([[0, 1, 0],
                        # [-1, 0, 0],
                       #  [0, 0, 1]])
    # corners = corners @ R_rotate.T

    # Displace the corners by one unit in the z direction to give room for camera center
    corners += c.reshape(1,4)[:, :3] + np.array([0, 0, camera_axis_length])
    z_axis += t
    corners = corners @ R
    z_axis = z_axis @ R

    ax.plot(c[0], c[1], c[2], 'bo', markersize=1)
    #ax.plot3D(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color='blue', linewidth=2)

    for corner in corners:
        ax.plot(
           [c[0], corner[0]], 
           [c[1], corner[1]], 
           [c[2], corner[2]], 
           'r--', alpha=0.5
        )

    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]
    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]
    
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    
    return None

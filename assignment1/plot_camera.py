import numpy as np
import matplotlib.pyplot as plt
from pflat import pflat
from camera_center_and_axis import camera_center_and_axis

def plot_camera(P, s, ax=None):
    """Plots the principal axis of the camera scaled by s, from the camera center."""
    if P.shape != (3, 4):
        raise ValueError('Input must be a 3x4 matrix')

    C, a = camera_center_and_axis(P)
    a = s*a

    if ax==None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    ax.scatter(C[0], C[1], C[2], color='r')
    ax.quiver(C[0], C[1], C[2], a[0], a[1], a[2], length=1, color='b')

    if ax==None:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Principal Axis')

def plot_3d_points(points, ax=None):
    if ax==None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    ax.plot3D(points[0], points[1], points[2], 'o', markersize=0.5)
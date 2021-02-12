import numpy as np
import matplotlib.pyplot as plt
from math import radians

#
# Tip: Define functions to create the basic 4x4 transformations
#
def translate_x(x):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]
                     [0, 0, 0 ,1]])

def translate_y(y):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, y],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate_z(z):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def rotate_x(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0,  0],
                     [0, c, -s, 0],
                     [0, s, c,  0],
                     [0, 0, 0,  1]])

def rotate_y(phi_deg):
    phi_rad = np.deg2rad(phi_deg)
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    return np.array([[c, 0, s,  0],
                     [0, 1, 0,  0],
                     [-s, 0, c, 0],
                     [0, 0, 0,  1]])

def rotate_z(psi_deg):
    psi_rad = np.deg2rad(psi_deg)
    c, s = np.cos(psi_rad), np.sin(psi_rad)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])

# Note that you should use np.array, not np.matrix,
# as the latter can cause some unintuitive behavior.
#
# translate_x/y/z could alternatively be combined into
# a single function.

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """

    # Tip: Use the @ operator for matrix multiplication, the *
    # operator on arrays performs element-wise multiplication!

    #
    # Placeholder code (replace with your implementation)
    #
    X_tilde = K @ X[:3, :]
    uv = X_tilde[:2, :] / X_tilde[2:3,:]
    return uv

def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    This uses your project function, so implement it first.

    Control the length of the axes using 'scale'.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis

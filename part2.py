from TF import TF
from homography import estimate_H, decompose_H
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from common import rotate_x, rotate_y, rotate_z
from quanser import Quanser
from generate_quanser_summary import generate_quanser_summary
from methods import gauss_newton, levenberg_marquardt


# %% Loading and defining variables
K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
xy = platform_to_camera[:2]

#platform_corners_metric =  [X; Y, 0; 1] = XY01
XY01 = np.loadtxt('../data/platform_corners_metric.txt')
XY = XY01[:2]
XY1 = np.vstack((XY, np.ones(np.shape(XY[1]))))

#platform_corners_image = [u; v; 1] = uv1
# uv = np.loadtxt('../data/platform_corners_image.txt')
platform_corners_image = np.loadtxt('../data/platform_corners_image.txt')

detections = np.loadtxt('../data/detections.txt')



# %% Task 2.1 

"""
Estimate T-camera-platform using the linear algorithm from HW4. Compute the predicted 
image location using the two transformations (a) and (b)
"""
platform_corners_image = np.vstack((platform_corners_image,
                                    np.ones(platform_corners_image.shape[1])))

"""
xy declared
"""

platform_corners_camera = np.linalg.inv(K) @ platform_corners_image

"""
 H is the homoegraphy function that maps something from 3d to 2d. 
"""
H = estimate_H(xy, XY) #dim: 3x3

# T is composed of R(3x3) and t. Dim: 4x4
T = decompose_H(H)

camera = TF('camera')
platform = camera.new_TF_from_t_mat('platform', platform_to_camera)

# (a)
u_tilde1 = K @ H @ XY1
print('u_tilde1:')
print(u_tilde1)

#(b): 
u_tilde2 = K @ T[3:3] @ XY1


# %% Task 2.2
""" 
Estimate T_camera platform by minimizing the sum of squared reprojection errors with
with respect to R and t using L-M. 

You will need to parametrize the pose as a vector. A rotation parameterization as
eq 15 may be used. The p1, p2 and p3 in the tip is NOT roll, pitch and yaw of the
helicopter, as they are not involved in part 2(Piazza tip)

H flytter noe fra 3d til 2d space. T [R, t], u tilde er predicted image location

""" 

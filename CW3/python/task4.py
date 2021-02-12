# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:28:17 2021

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from math import radians

from common import project


def translation(translation, rx=0, ry=0, rz=0):
    R = Rotation.from_euler('xyz', (rx, ry, rz)).as_matrix()
    t = np.array(translation)[:, None]
    T = np.block([[R, t],
                  [np.zeros(3), 1]])
    return T


plt.close('all')
img = cv2.cvtColor(cv2.imread('../data/quanser.jpg'), cv2.COLOR_BGR2RGB)

K = np.loadtxt('../data/heli_K.txt')
T_plat_cam = np.loadtxt('../data/platform_to_camera.txt')
heli_points = np.loadtxt('../data/heli_points.txt')

d = 0.1145
screws_plat = np.array([[0, 0],
                        [d, 0],
                        [d, d],
                        [0, d]]).T

padding = np.stack(4*[[0, 1]], 1)
screws_plat = np.concatenate((screws_plat, padding), 0)

psi = radians(11.6)
theta = radians(28.9)
phi = radians(0)

#Task 4.3
T_base_platform = translation([0.05725, 0.05725, 0], rz=psi)
#Task 4.4
T_hinge_base = translation([0, 0, 0.325], ry=theta)
#Task 4.5
T_arm_hinge = translation([0, 0, -0.05])
#Task 4.6
T_rot_arm = translation([0.65, 0, -0.03], rx=phi)


def draw_frame(T, ax):
    d = 0.05
    o = T @ np.array([[0, 0, 0, 1]]).T
    x = T @ np.array([[d, 0, 0, 1]]).T
    y = T @ np.array([[0, d, 0, 1]]).T
    z = T @ np.array([[0, 0, d, 1]]).T
    ax.plot(*project(K, np.concatenate((o, y), 1)), c='g')
    ax.plot(*project(K, np.concatenate((o, z), 1)), c='b')
    ax.plot(*project(K, np.concatenate((o, x), 1)), c='r')


fig, ax = plt.subplots(1, 1)
T_plat2base = translation([d/2, d/2, 0])

ax.imshow(img)
# current = np.eye(4)
# for T in [T_plat_cam, T_base_platform, T_hinge_base, T_arm_hinge,
#           T_rot_arm]:
#     current = current @ T
#     draw_frame(current, ax)


ax.scatter(*project(K, T_plat_cam @ screws_plat), c='y')

# ax.scatter(*project(K, T_plat_cam@T_base_platform @ T_hinge_base @
#                     T_arm_hinge @ heli_points[:3].T), c='y')

# ax.scatter(*project(K, T_plat_cam@T_base_platform @ T_hinge_base @
#                     T_arm_hinge @ T_rot_arm @ heli_points[3:].T), c='y')
plt.show()
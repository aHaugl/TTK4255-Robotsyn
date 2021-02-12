# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:27:39 2021

"""

import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.spatial.transform import Rotation
from math import radians

# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt('../data/task2K.txt')
X = np.loadtxt('../data/task3points.txt')

T = translate_z(6) @ rotate_x(15) @ rotate_y(45)


# Task 2.2: Implement the project function
X = T @ X

u,v = project(K, X)

# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400

#
# Figure for Task 2.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
draw_frame(K, T)
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!!!!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
plt.show()

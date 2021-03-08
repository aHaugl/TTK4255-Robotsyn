from sympy.physics.vector import frame
from TF import TF
import sympy as sp
import numpy as np
from numbaprinter import numbafy
from itertools import product, repeat

from matplotlib import pyplot as plt
import matplotlib
from utils import project, flatten

img_idx = 123
matplotlib.use('Qt5Agg')
plt.close('all')

K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
heli_points = np.loadtxt('../data/heli_points.txt')
img = plt.imread(f'../data/quanser_image_sequence/video{img_idx:04d}.jpg')
logs = np.loadtxt('../data/logs.txt')

camera = TF('camera')
platform = camera.new_TF_from_t_mat('platform', platform_to_camera)
base = platform.new_TF('base', rotation_order='yxz')
hinge = base.new_TF('hinge', rotation_order='xzy')
arm = hinge.new_TF('arm', [0, 0, -0.05], [0, 0, 0])
rotors = arm.new_TF('rotors', rotation_order='yzx')

tfs = [platform, base, hinge, arm, rotors]

base.set_position([0.1145/2, 0.1145/2, 0.0])
hinge.set_position([0.00, 0.00,  0.325])
rotors.set_position([0.65, 0.00, -0.030])

base.set_orientation_body([0, 0, None])
hinge.set_orientation_body([0, None,  0])
rotors.set_orientation_body([None, 0, 0])

arm_points = arm.new_points('P_arm', heli_points[:3, :3])
rotors_points = rotors.new_points('P_rotors', heli_points[3:, :3])

axis_points = [tf.new_points(f'{tf}_ax', 0.05*np.eye(3)) for tf in tfs]
axis_points_w_origos = [[tf.origo, point] for tf, axis in zip(tfs, axis_points)
                        for point in axis]


axis_points = camera.project_points(flatten(axis_points_w_origos))
axis_projections = project(K, axis_points)
f = sp.lambdify(rotors.free_symbols, axis_projections)

fig, ax = plt.subplots(1, 1)

compensation = np.array([11.6, 28.9, 0.0])*np.pi/180 - logs[0, 1:]
rotations = logs[img_idx*16, 1:] + compensation
lines = f(*rotations)
x, y = lines
plt.imshow(img)

for i in range(0, lines.shape[1], 2):
    ax.plot(*lines[:, i:i+2], color='rgb'[(i//2) % 3], linewidth=2)

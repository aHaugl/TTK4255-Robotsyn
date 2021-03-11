from TF import TF
import sympy as sp
import numpy as np
from numbaprinter import numbafy, create_numba_file
import matplotlib
from utils import project, flatten


img_idx = 123
K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
heli_points = np.loadtxt('../data/heli_points.txt')
# img = plt.imread(f'../data/quanser_image_sequence/video{img_idx:04d}.jpg')
logs = np.loadtxt('../data/logs.txt')
detections = np.loadtxt('../data/detections.txt')

camera = TF('camera')
platform = camera.new_TF_from_t_mat('platform', platform_to_camera)
base = platform.new_TF('base', rotation_order='yxz')
hinge = base.new_TF('hinge', rotation_order='xzy')
arm = hinge.new_TF('arm', [0, 0, -0.05], [0, 0, 0])
rotors = arm.new_TF('rotors', rotation_order='yzx')

tfs = [platform, base, hinge, arm, rotors]
# base.set_position([0.1145/2, 0.1145/2, 0.0])
# hinge.set_position([0.00, 0.00,  0.325])
# rotors.set_position([0.65, 0.00, -0.030])

# base.set_orientation_body([0, 0, None])
# hinge.set_orientation_body([0, None,  0])
# rotors.set_orientation_body([None, 0, 0])

arm_points = arm.new_points('P_arm', heli_points[:3, :3])
rotors_points = rotors.new_points('P_rotors', heli_points[3:, :3])


# mypoint = rotors.new_point('mypoint', sp.symbols('mypoint_x:z'))
rotors_t = camera.t_mat(base)
rotors_t_flat = rotors_t.reshape(16, 1)
rotors_t_diff = rotors_t_flat.jacobian(sp.Matrix(rotors.free_symbols))

arm_t = camera.t_mat(arm)
arm_t_flat = rotors_t.reshape(16, 1)
arm_t_diff = rotors_t_flat.jacobian(sp.Matrix(rotors.free_symbols))

create_numba_file(
    [['rotors_t', rotors_t, rotors.free_symbols],
     ['rotors_t_diff', rotors_t_diff, rotors.free_symbols],
     ['arm_t', arm_t, rotors.free_symbols],
     ['arm_t_diff', arm_t_diff, rotors.free_symbols]])
# rotors_points = sp.MatrixSymbol('P', 4, 3)

# image_points = project(K, camera.project_points(arm_points+rotors_points))

# detection_syms = sp.symbols('detection_x:y0:7')
# detections = sp.Matrix(detection_syms)
# weight_syms = sp.symbols('weight:7')
# weights = sp.Matrix(weight_syms * 2)

# res = (image_points.reshape(14, 1) - detections).multiply_elementwise(
#     weights)

# variables = (rotors.free_symbols
#              + list(detection_syms)
#              + list(weight_syms))
# print('hello')
# jac = res.jacobian(sp.Matrix(rotors.free_symbols))
# # print(sp.pycode(jac))

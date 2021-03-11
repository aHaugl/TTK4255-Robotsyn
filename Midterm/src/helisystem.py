from math import log
import sympy as sp
import numpy as np
from sympy.vector import BodyOrienter
from TF import TF
from matplotlib import pyplot as plt
import matplotlib
from common import project
matplotlib.use('Qt5Agg')


class HeliSysyem:
    def __init__(self):
        self.K = np.loadtxt('../data/K.txt')
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
        p2c = self.platform_to_camera

        self.camera = TF('camera')
        self.platform = self.camera.new_TF_from_t_mat('platform', p2c)
        self.base = self.platform.new_TF('base', rotation_order='yxz')
        self.hinge = self.base.new_TF('hinge', rotation_order='xzy')
        self.arm = self.hinge.new_TF('arm', [0, 0, -0.05], [0, 0, 0])
        self.rotors = self.arm.new_TF('rotors', rotation_order='yzx')

    @property
    def free_symbols(self):
        return self.rotors.free_symbols

    def project(self, points):
        uvz = self.K @ self.camera.project_points(points)

        uv = uvz[:2, :]
        for i in range(uv.shape[1]):
            uv[:, i] = uv[:, i] / uvz[2, i]
        return uv


heli_points = np.loadtxt('../data/heli_points.txt')
K = np.loadtxt('../data/K.txt')
img = plt.imread(f'../data/quanser_image_sequence/video{0:04d}.jpg')

helisys = HeliSysyem()
helisys.base.set_position([0.1145/2, 0.1145/2, 0.0])
helisys.hinge.set_position([0.00, 0.00,  0.325])
helisys.rotors.set_position([0.65, 0.00, -0.030])

helisys.base.set_orientation_body([0, 0, None])
helisys.hinge.set_orientation_body([0, None,  0])
helisys.rotors.set_orientation_body([None, 0, 0])


arm_points = helisys.arm.new_points('P_arm', heli_points[:3, :3])
rotors_points = helisys.rotors.new_points('P_rotors', heli_points[3:, :3])


axproj = sp.lambdify(helisys.free_symbols, helisys.rotors.axis_projection())

markers = helisys.hinge.new_points('p', [[0, 0, 0],
                                         [0.1, 0, 0],
                                         [0, 0.1, 0],
                                         [0, 0, 0.1]])
markerproj = sp.lambdify(helisys.free_symbols,
                         helisys.project(markers))
logs = np.loadtxt('../data/logs.txt')
plt.close('all')
plt.imshow(img)


axis = project(K, axproj(0.5, 0.3, 0.7))

# axis = markerproj(0.5, 0.3, 0.7)
plt.plot([axis[0, 0], axis[0, 1]], [axis[1, 0], axis[1, 1]])
plt.plot([axis[0, 0], axis[0, 2]], [axis[1, 0], axis[1, 2]])
plt.plot([axis[0, 0], axis[0, 3]], [axis[1, 0], axis[1, 3]])
# plt.scatter(*markerproj(0, 0, 0))
# helisys.project(arm_points + rotors_points)
plt.show()
hs = helisys
hs.camera.project_point(hs.base.origo)

from TF import TF
import sympy as sp
import numpy as np
from numbaprinter import create_numba_file

platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

camera = TF('camera')
platform = camera.new_TF('platform', None, None)

variables = platform.free_symbols
if __name__ == '__main__':

    platform_t = camera.t_mat(platform)
    platform_t_flat = platform_t.reshape(16, 1)
    platform_t_jac = platform_t_flat.jacobian(sp.Matrix(variables))
    create_numba_file('platfrom',
                      [['get_platform_T', platform_t, variables],
                       ['get_platform_T_diff', platform_t_jac, variables]])

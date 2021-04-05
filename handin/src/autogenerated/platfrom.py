import numpy as np
import numba as nb


@nb.njit(cache=True)
def get_platform_T(states):
    """
    platform_x, platform_y, platform_z, platform_p, platform_q, platform_r
    """

    platform_x = states[0]
    platform_y = states[1]
    platform_z = states[2]
    platform_p = states[3]
    platform_q = states[4]
    platform_r = states[5]

    x0 = np.cos(platform_q)
    x1 = np.cos(platform_r)
    x2 = np.sin(platform_r)
    x3 = np.cos(platform_p)
    x4 = x2*x3
    x5 = np.sin(platform_q)
    x6 = np.sin(platform_p)
    x7 = x1*x6
    x8 = x2*x6
    x9 = x1*x3

    return np.array([[x0*x1, -x4 + x5*x7, x5*x9 + x8, platform_x],
                     [x0*x2, x5*x8 + x9, x4*x5 - x7, platform_y],
                     [-x5, x0*x6, x0*x3, platform_z],
                     [0., 0., 0., 1.]])


@nb.njit(cache=True)
def get_platform_T_diff(states):
    """
    platform_x, platform_y, platform_z, platform_p, platform_q, platform_r
    """

    platform_x = states[0]
    platform_y = states[1]
    platform_z = states[2]
    platform_p = states[3]
    platform_q = states[4]
    platform_r = states[5]

    x0 = np.sin(platform_q)
    x1 = np.cos(platform_r)
    x2 = x0*x1
    x3 = np.sin(platform_r)
    x4 = np.cos(platform_q)
    x5 = x3*x4
    x6 = np.sin(platform_p)
    x7 = x3*x6
    x8 = np.cos(platform_p)
    x9 = x2*x8 + x7
    x10 = x1*x6
    x11 = x1*x8
    x12 = -x0*x7 - x11
    x13 = x3*x8
    x14 = x2*x6
    x15 = x0*x13

    return np.array([[0., 0., 0., 0., -x2, -x5],
                     [0., 0., 0., x9, x10*x4, x12],
                     [0., 0., 0., x13 - x14, x11*x4, x10 - x15],
                     [1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., -x0*x3, x1*x4],
                     [0., 0., 0., -x10 + x15, x5*x6, -x13 + x14],
                     [0., 0., 0., x12, x5*x8, x9],
                     [0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., -x4, 0.],
                     [0., 0., 0., x4*x8, -x0*x6, 0.],
                     [0., 0., 0., -x4*x6, -x0*x8, 0.],
                     [0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.]])

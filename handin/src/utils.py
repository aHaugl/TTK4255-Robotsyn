import numpy as np
import numba as nb


@nb.njit(parallel=True, cache=True)
def jacobian_matmul(T_jac, points):
    """
    the jacobian are with respect to (n) variables

    T_jac: jacobian of a flattened T matrix, has shape (4*4, n)
    points: matrix of points, has shape (4, c)

    output: jacobian of the flattened points, has shape (3*c, n)
    """
    n_points = points.shape[1]
    n_args = T_jac.shape[1]

    output = np.zeros((3*n_points, n_args), np.float64)
    for var in nb.prange(n_args):
        for point in range(n_points):
            for ax in range(3):
                for i in range(4):
                    output[point + ax * n_points, var] += (
                        T_jac[ax * 4 + i, var] * points[i, point])
    return output


@nb.njit(parallel=True, cache=True)
def get_uv_jac(K, T, T_jac, points):
    """
    Finds the jacobian of the image coordinates
    """
    n_points = points.shape[1]
    n_states = T_jac.shape[1]

    padded_eye = np.zeros((4, 3))
    padded_eye[:3, :3] = np.eye(3)

    output_state = np.zeros((2*n_points, n_states), np.float64)
    output_points = np.zeros((2*n_points, 3*n_points), np.float64)
    P_jac = jacobian_matmul(T_jac, points)
    xyz = K @ (T @ points)[:3]
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    n_points = P_jac.shape[0]//3
    for var in nb.prange(n_states):
        xyz_dot = K @ np.ascontiguousarray(P_jac[:, var]).reshape(3, -1)
        x_dot = xyz_dot[0]
        y_dot = xyz_dot[1]
        z_dot = xyz_dot[2]
        u_dot = (x_dot * z - x * z_dot)/z**2
        v_dot = (y_dot * z - y * z_dot)/z**2
        output_state[:, var] = np.hstack((u_dot, v_dot))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    xyz_dot = K@(T@padded_eye)[:3]
    x_dot = xyz_dot[0].reshape(1, -1)
    y_dot = xyz_dot[1].reshape(1, -1)
    z_dot = xyz_dot[2].reshape(1, -1)
    u_dot = (x_dot * z - x * z_dot)/z**2
    v_dot = (y_dot * z - y * z_dot)/z**2

    xyz_arg = np.arange(3) * n_points
    for point in nb.prange(n_points):
        for i, arg in enumerate(xyz_arg):
            output_points[point, point + arg] = u_dot[point, i]
            output_points[n_points + point, point + arg] = v_dot[point, i]
    return output_state, output_points


@nb.njit(cache=True)
def project(K, points):
    xyz = K @ points[:3]
    uv = xyz[:2]/xyz[2:3]
    return uv

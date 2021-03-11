from numba.np.ufunc import parallel
import numpy as np
import numba as nb
from accelerated import rotors_t, rotors_t_diff
from methods import finite_jacobian


K = np.loadtxt('../data/K.txt')


@nb.njit('float64[:,::1](float64[:,::1], float64[:,:])',
         parallel=True, cache=True)
def point_jacobian(T_diff, points):
    """
    T_diff: jacobian of a flattened T matrix
    points: matrix of points 

    output: jacobian of the flattened points
    """
    n_points = points.shape[1]
    n_args = T_diff.shape[1]
    # T_diff = rotors_t_diff(system_args)
    output = np.zeros((3*n_points, n_args), np.float64)
    for var in nb.prange(n_args):
        for point in nb.prange(n_points):
            for ax in nb.prange(3):
                for i in range(4):
                    output[point + ax * n_points, var] += (
                        T_diff[ax * 4 + i, var] * points[i, point])
    return output


if __name__ == '__init__' or 1:
    system_args = np.zeros(18, dtype=np.float64)
    points = np.random.random((4, 50))

    x = np.random.random(18)
    T_diff = rotors_t_diff(x)

    def foo(vars):
        return np.ravel((rotors_t(vars) @ points)[:3])
    d = finite_jacobian(foo, x, 1e-9)
    d2 = point_jacobian(T_diff, points)
    d3 = d-d2
    print(np.mean(d3))

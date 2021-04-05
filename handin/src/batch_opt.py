import numpy as np
import numba as nb
from quanser_fast import get_resiudials, get_resiudials_diff


@nb.njit(cache=True, parallel=True)
def get_J(weights, states, points,
          dynamic_idx, static_idx):
    """
    Get the static and dynamic part of the J matrix.
    """
    n_points = points.shape[1]
    n_states = states.shape[1]
    static_idx = np.hstack((static_idx, n_states+np.arange(n_points*3)))
    n_static = static_idx.shape[0]
    n_dynamic = dynamic_idx.shape[0]
    image_n = states.shape[0]  # l

    static_blocks = np.empty((image_n, n_points*2, n_static))
    dynamic_blocks = np.empty((image_n, n_points*2, n_dynamic))

    for img in nb.prange(image_n):
        resiudials_diff = get_resiudials_diff(
            weights[img], states[img], points)

        static_blocks[img] = resiudials_diff[:, static_idx]
        dynamic_blocks[img] = resiudials_diff[:, dynamic_idx]
    return static_blocks, dynamic_blocks


@nb.njit(cache=True, parallel=True)
def get_all_resudials(uv_arr, weight_arr, states, points, out):
    """
    Computes 
    """
    for img in nb.prange(out.shape[0]):
        out[img, :, 0] = get_resiudials(
            uv_arr[img], weight_arr[img], states[img], points)


def levenberg_marquardt_batch(uv_arr, weight_arr, states, points,
                              dynamic_idx, static_idx):
    states = states.copy()
    states_next = states.copy()
    points = points.copy()
    points_next = points.copy()

    n_points = points.shape[1]
    n_static = static_idx.shape[0]
    n_dynamic = dynamic_idx.shape[0]
    n_image = states.shape[0]

    residual = np.empty((n_image, n_points*2, 1))
    residual_next = np.empty((n_image, n_points*2, 1))

    max_iterations = 1000
    mu = None

    D_inv = np.empty((n_image, 3, 3))
    diag_D_idx = np.arange(n_dynamic)
    diag_A_idx = np.arange(n_static)

    get_all_resudials(uv_arr, weight_arr, states, points,
                      out=residual)
    residual_squaresum = np.sum(residual**2)

    for iteration in range(max_iterations):

        static, dynamic = get_J(weight_arr, states, points,
                                dynamic_idx, static_idx)

        static_full = static.reshape(-1, static.shape[2])

        A = static_full.T @ static_full
        B = (np.transpose(static, (0, 2, 1))
             @ dynamic)
        C = np.transpose(B, (0, 2, 1))
        D = np.transpose(dynamic, (0, 2, 1)) @ dynamic

        if mu is None:
            maximum = 0
            maximum = np.maximum(maximum, np.amax(np.diag(A)))
            D_diag = D[:, np.arange(3), np.arange(3)]
            maximum = np.maximum(maximum, np.amax(D_diag))
            mu = 1e-3 * maximum

        A[diag_A_idx, diag_A_idx] += mu
        for img in range(n_image):
            D[img, diag_D_idx, diag_D_idx] += mu
            D_inv[img] = np.linalg.inv(D[img])

        JTstatic = -np.sum(np.transpose(static, (0, 2, 1)) @ residual, axis=0)
        JTdyn = -np.transpose(dynamic, (0, 2, 1)) @ residual

        B_D_inv = B @ D_inv
        LHS = A - np.sum(B_D_inv @ np.transpose(B, (0, 2, 1)), axis=0)
        RHS = JTstatic - np.sum(B_D_inv @ JTdyn, axis=0)

        static_delta = np.linalg.solve(LHS, RHS)
        dynamic_delta = D_inv @ (JTdyn - C @ static_delta)
        points_delta = static_delta[n_static:, 0].reshape(3, -1)

        states_next[:, static_idx] = (
            states[:, static_idx] + static_delta[None, :n_static, 0])

        states_next[:, dynamic_idx] = (
            states[:, dynamic_idx] + dynamic_delta[:, :, 0])

        points_next[:3, :] = points[:3, :] + points_delta

        get_all_resudials(uv_arr, weight_arr, states_next, points_next,
                          out=residual_next)

        if np.sum(residual_next**2) < residual_squaresum:
            states = states_next.copy()
            points = points_next.copy()
            residual = residual_next.copy()
            residual_squaresum = np.sum(residual**2)
            mu /= 3
        else:
            mu *= 2

        stepsize = np.amax(np.abs(static_delta))
        if stepsize <= 1e-6:
            break
    print(f'Optimization finished after {iteration+1} iterations')
    return states, points, np.squeeze(residual)

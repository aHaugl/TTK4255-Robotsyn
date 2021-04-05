import numpy as np


def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.
    A = np.zeros((2 * XY.shape[1], 9), np.float32)
    for i in range(XY.shape[1]):
        A[2*i, 0:3] = (*XY[:, i], 1)
        A[2*i, -3:] = -xy[0, i] * np.array((*XY[:, i], 1))

        A[2*i+1, 3:6] = (*XY[:, i], 1)
        A[2*i+1, -3:] = -xy[1, i] * np.array((*XY[:, i], 1))

    U, s, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)  # Placeholder, replace with your implementation
    return H


def decompose_H(H):
    # Tip: Use np.linalg.norm to compute the Euclidean length
    scaling = np.linalg.norm(H[:, 0])
    H = H/scaling

    r3 = np.cross(H[:, 0], H[:, 1])

    # R1 = np.block([[H[:, :2], r3[:, None]]])
    # R2 = np.block([[-H[:, :2], r3[:, None]]])
    R1 = closest_rotation_matrix(np.block([[H[:, :2], r3[:, None]]]))
    R2 = closest_rotation_matrix(np.block([[-H[:, :2], r3[:, None]]]))

    # Placeholder, replace with your implementation
    T1 = np.block([[R1, H[:, -1:]],
                   [0, 0, 0, 1]])

    T2 = np.block([[R2, -H[:, -1:]],
                   [0, 0, 0, 1]])
    return T1 if T1[2, 3] > 0 else T2


def closest_rotation_matrix(Q):
    U, s, VT = np.linalg.svd(Q)
    R = U@VT
    return R

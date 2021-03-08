import sympy as sp


def project(K, points):
    uvz = K @ points
    uv = uvz[:2, :]
    for i in range(uv.shape[1]):
        uv[:, i] = uv[:, i] / uvz[2, i]
    return uv


def flatten(list_):
    return [i for sublist in list_ for i in sublist]

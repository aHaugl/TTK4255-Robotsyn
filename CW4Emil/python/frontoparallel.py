from numpy.lib.ufunclike import _deprecate_out_named_y
from common import estimate_H
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline
import scipy


def interpolate(img, x, y):
    splines = [
        RectBivariateSpline(
            np.arange(img.shape[0]), np.arange(img.shape[1]), img[..., color])
        for color in range(3)]

    return np.stack((spline(y, x, grid=False) for spline in splines), axis=-1)


I = plt.imread(f'../data/image{image_number:04d}.jpg')

k = np.amax(XY[0])/5.
xs = np.linspace(-k, k*7, 8*128)
ys = np.linspace(-k, k*5, 6*128)

x, y = np.meshgrid(xs, ys, indexing='xy')

args = np.stack((x, y, np.ones(x.shape)), -1)

# transform = np.linalg.inv(H)
transform = K @ H
# transform = np.eye(3)

args_transformed = np.squeeze(transform[None, None, ...]@args[..., None])
args_transformed = args_transformed[..., :2] / args_transformed[..., -1:]
alpha = 1
selected = args_transformed * alpha + args[..., :2] * (1-alpha)
# selected = args_transformed
img2 = interpolate(I, selected[..., 0], selected[..., 1])

plt.close('all')

plt.imshow(img2/255.)

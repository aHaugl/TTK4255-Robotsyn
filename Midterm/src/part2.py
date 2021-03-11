from TF import TF
from homography import estimate_H, decompose_H
import sympy as sp
import numpy as np


K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

platform_corners_metric = np.loadtxt('../data/platform_corners_metric.txt')
platform_corners_image = np.loadtxt('../data/platform_corners_image.txt')


platform_corners_image = np.vstack((platform_corners_image,
                                    np.ones(platform_corners_image.shape[1])))
platform_corners_camera = np.linalg.inv(K) @ platform_corners_image


H = estimate_H(platform_corners_camera[:2], platform_corners_metric[:2])
T = decompose_H(H)
camera = TF('camera')
# platform = camera.new_TF_from_t_mat('platform', platform_to_camera)

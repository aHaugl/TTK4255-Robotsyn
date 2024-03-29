import matplotlib.pyplot as plt
import numpy as np
from common import *


class Quanser:
    def __init__(self):
        self.K = np.loadtxt('../data/K.txt')
        self.heli_points = np.loadtxt('../data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

    def residuals(self, uv, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge = translate(0.00, 0.00, -0.050)
        rotors_to_arm = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera = self.platform_to_camera@base_to_platform
        self.hinge_to_camera = self.base_to_camera@hinge_to_base
        self.arm_to_camera = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:, :3]
        p2 = self.rotors_to_camera @ self.heli_points[:, 3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat  # Save for use in draw()

        #
        # TASK: Compute the vector of residuals.
        #
        r = np.ravel((uv_hat - uv) * weights[None, :])
        return r

    def draw(self, uv, weights, image_number):
        img = plt.imread(
            '../data/quanser_image_sequence/video%04d.jpg' % image_number)
        plt.imshow(img)
        plt.scatter(*uv[:, weights == 1], linewidths=1,
                    edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' %
                  image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, img.shape[1]])
        plt.ylim([img.shape[0], 0])
        plt.savefig('../output/out_reprojection.png')

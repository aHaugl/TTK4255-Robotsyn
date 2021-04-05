import matplotlib.pyplot as plt
import numpy as np

from generate_quanser_summary_full import generate_quanser_summary
from methods import levenberg_marquardt
from quanser_fast import Quanser, get_resiudials_diff

plt.close('all')

detections = np.loadtxt('../data/detections.txt')
default_points = np.loadtxt('../output/heli_points_optimized.txt')

run_until = detections.shape[0]

visualize_number = 100

quanser = Quanser()


default_states = np.loadtxt('../output/states_optimized.txt')
p_args = np.array([4, 7, 11])  # base_r, hinge_q, rotors_p
p = default_states[p_args]

all_residuals = []
trajectory = np.zeros((run_until, 3))

states_arr = np.repeat(default_states[None, :], run_until, axis=0)
weight_arr = np.empty((run_until, 7))
uv_arr = np.empty((run_until, 2, 7))

states = default_states.copy()
for image_number in range(run_until):
    weights = detections[image_number, :: 3].copy()
    uv = np.vstack((detections[image_number, 1:: 3],
                    detections[image_number, 2::3]))

    def residualsfun(p):
        states[p_args] = p
        return quanser.residuals(uv, weights, states, default_points)

    def diff(p):
        states[p_args] = p
        return get_resiudials_diff(weights, states, default_points)[:, p_args]

    cost_logger = None
    p = levenberg_marquardt(residualsfun, diff, p, cost_logger=cost_logger)

    states_arr[image_number, p_args] = p
    weight_arr[image_number] = weights
    uv_arr[image_number] = uv

    all_residuals.append(residualsfun(p))


generate_quanser_summary(states_arr, all_residuals, detections, p_args)

fig, ax = plt.subplots(1, 1)


def show_img(img_n):
    plt.sca(ax)
    plt.cla()
    states = states_arr[img_n]
    quanser.residuals(uv_arr[img_n], weight_arr[img_n], states, default_points,
                      True)
    quanser.draw(uv_arr[img_n], weight_arr[img_n], img_n)


show_img(150)
plt.show()

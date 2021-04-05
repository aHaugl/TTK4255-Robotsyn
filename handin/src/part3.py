import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from generate_quanser_summary_full import generate_quanser_summary
from batch_opt import levenberg_marquardt_batch
from system_full import variables
from methods import levenberg_marquardt
from quanser_fast import Quanser, get_resiudials_diff


ONLY_MODEL_LENGTHS = False

detections = np.loadtxt('../data/detections.txt')
default_points = np.loadtxt('../data/heli_points.txt').T

run_until = detections.shape[0]

quanser = Quanser()
p = np.array([11.6, 28.9, 0.0])*np.pi/180  # Optimal for image number 0

variables = np.array([str(variable) for variable in variables])
"""
The following variables can be altered. 
base_x, base_y, 
base_p, base_q, base_r

hinge_z, 
hinge_p, hinge_q

rotors_x, rotors_y, rotors_z, 
rotors_p, rotors_q, rotors_r

To modify this; edit and run system_full.py
"""

default_states = np.array(
    [0.1145/2, 0.1145/2,
     0, 0, 0,
     0.325,
     0, 0,
     0.65, 0.00, -0.030,
     0, 0, 0,
     ])

p_args = np.array([4, 7, 11])  # base_r, hinge_q, rotors_p

default_states[p_args] = p
dynamic_idx = p_args


static_idx = np.array(
    [i for i in range(default_states.shape[0]) if i not in dynamic_idx])

if ONLY_MODEL_LENGTHS:
    static_idx = np.array([0, 5, 8, 10])

dynamic_names = ', '.join([str(variables[i]) for i in dynamic_idx])
static_names = ', '.join([str(variables[i]) for i in static_idx])
print('')
print(f"Dynamic variables:\n{dynamic_names}\n")
print(f"Static variables:\n{static_names}\n")

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
        states[dynamic_idx] = p
        return quanser.residuals(uv, weights, states, default_points)

    def diff(p):
        states[dynamic_idx] = p
        return get_resiudials_diff(weights, states, default_points)[:, p_args]

    cost_logger = None
    p = levenberg_marquardt(residualsfun, diff, p, cost_logger=cost_logger)

    states_arr[image_number, dynamic_idx] = p
    weight_arr[image_number] = weights
    uv_arr[image_number] = uv

    all_residuals.append(residualsfun(p))

points = default_points

t0 = time.time()

states_arr, points, all_residuals = levenberg_marquardt_batch(
    uv_arr, weight_arr, states_arr, default_points,
    dynamic_idx, static_idx)

print(f'Optimization took {time.time()-t0}s\n')

generate_quanser_summary(states_arr, all_residuals, detections, p_args)

print('\n\nChanges from optimization\n')

print('Static parameters:')
for i in static_idx:
    variable = str(variables[i])
    init = default_states[i]
    final = states_arr[0, i]
    print(f'{str(variables[i]):<10} {init:10.6f} => {final:10.6f}')

print('\nPoints:\n')
for j, i in product(range(points.shape[1]), range(3)):
    variable = f"P{j}_{'xyz'[i]}"
    init = default_points[i, j]
    final = points[i, j]
    print(f'{variable} {init:10.6f} => {final:10.6f}')


fig, ax = plt.subplots(1, 1)


def show_img(img_n):
    plt.sca(ax)
    plt.cla()
    states = states_arr[img_n]
    quanser.residuals(uv_arr[img_n], weight_arr[img_n], states, points,
                      True)
    quanser.draw(uv_arr[img_n], weight_arr[img_n], img_n)


np.savetxt('../output/heli_points_optimized.txt', points)
np.savetxt('../output/states_optimized.txt', states_arr[0])

show_img(86)

plt.show()

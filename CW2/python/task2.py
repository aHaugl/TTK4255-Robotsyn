import numpy as np
import matplotlib.pyplot as plt
from common import *

# This bit of code is from HW1.
edge_threshold = 0.015
blur_sigma     = 1
filename       = '../data/grid.jpg'
I_rgb          = plt.imread(filename)
I_rgb          = im2double(I_rgb) #Ensures that the image is in floating-point with pixel values in [0,1].
I_gray         = rgb_to_gray(I_rgb)
Ix, Iy, Im     = derivative_of_gaussian(I_gray, sigma=blur_sigma) # See HW1 Task 3.6
x,y,theta      = extract_edges(Ix, Iy, Im, edge_threshold)

# You can adjust these for better results
line_threshold = 0.2
N_rho          = 1000
N_theta        = 600

###########################################
#
# Task 2.1: Determine appropriate ranges
#
###########################################
# Tip: theta is computed using np.arctan2. Check that the
# range of values returned by arctan2 matches your chosen
# ranges (check np.info(np.arctan2) or the internet docs).
rho_max   = np.sqrt(I_gray.shape[0]**2 + I_gray.shape[1]**2)*np.pi/2 # Placeholder value
rho_min   = -I_rgb.shape[1] # Placeholder value
theta_min = 0 # Placeholder value
theta_max = np.pi# Placeholder value

###########################################
#
# Task 2.2: Compute the accumulator array
#
###########################################
# Zero-initialize an array to hold our votes
H = np.zeros((N_rho, N_theta))

# 1) Compute rho for each edge (x,y,theta)
thetas = np.linspace(theta_min, theta_max, N_theta)
theta_vecs = np.stack((np.cos(thetas), np.sin(thetas)), 0)
edge_args = np.stack((x, y), -1)
rhos = np.dot(edge_args, theta_vecs)

# 2) Convert to discrete row,column coordinates
# Tip: Use np.floor(...).astype(np.int) to floor a number to an integer type

row = np.floor(N_rho * (rhos - rho_min) /
               (rho_max - rho_min)).astype(np.uint32)

H = np.apply_along_axis(lambda x: np.bincount(
    x, minlength=N_rho), axis=0, arr=row)

# 3) Increment H[row,column]
# Tip: Make sure that you don't try to access values at indices outside
# the valid range: [0,N_rho-1] and [0,N_theta-1]

###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
# 1) Call extract_local_maxima
#extract_local_maxima(H, line_threshold)

rho_arg, theta_arg = extract_local_maxima(H, 0.6)

# 2) Convert (row, column) back to (rho, theta)
rho_values = ((rho_arg+0.5) / N_rho) * (rho_max - rho_min) + rho_min
theta_values = ((theta_arg+0.5) / N_theta) * \
    (theta_max - theta_min) + theta_min
    
maxima_rho = rho_values  # Placeholder
maxima_theta = theta_values  # Placeholder

###########################################
#
# Figure 2.2: Display the accumulator array and local maxima
#
###########################################
plt.figure()
plt.imshow(H, extent=[theta_min, theta_max, rho_max, rho_min], aspect='auto')
plt.colorbar(label='Votes')
plt.scatter(maxima_theta, maxima_rho, marker='.', color='red')
plt.title('Accumulator array')
plt.xlabel('$\\theta$ (radians)')
plt.ylabel('$\\rho$ (pixels)')
# plt.savefig('out_array.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

###########################################
#
# Figure 2.3: Draw the lines back onto the input image
#
###########################################
plt.figure()
plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for theta,rho in zip(maxima_theta,maxima_rho):
    draw_line(theta, rho, color='yellow')
plt.title('Dominant lines')
# plt.savefig('out_lines.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

plt.show()

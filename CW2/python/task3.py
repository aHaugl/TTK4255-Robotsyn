import matplotlib.pyplot as plt
import cv2
import numpy as np
from common import *

# Note: the sample image is naturally grayscale
I = rgb_to_gray(im2double(plt.imread('../data/calibration.jpg')))
# I = rgb_to_gray(im2double(plt.imread('../data/test1.jpg')))
# I = rgb_to_gray(im2double(plt.imread('../data/test2.png')))

###########################################
#
# Task 3.1: Compute the Harris-Stephens measure
#
###########################################
sigma_D = 1
sigma_I = 3
alpha = 0.06


# I_diff_smooth = cv2.blur(I, (sigma_D,sigma_D))
I_diff_smooth = gaussian(I, sigma_D)
I_dx, I_dy, _ = derivative_of_gaussian(I, sigma_D)

I_dxx = I_dx**2
I_dxyy = I_dy**2
I_dxy = I_dx*I_dy

I_dxx_smooth = gaussian(I_dxx, sigma_I)
I_dyy_smooth = gaussian(I_dxyy, sigma_I)
I_dxy_smooth = gaussian(I_dxy, sigma_I)

det = I_dxx_smooth * I_dyy_smooth - I_dxy_smooth**2
trace = I_dxx_smooth + I_dyy_smooth

response = det - (alpha * trace**2)

###########################################
#
# Task 3.4: Extract local maxima
#
###########################################
corners_y = [0] # Placeholder Hente ut vertikale hjørner?
corners_x = [0] # Placeholder Hente ut horisontale hjørnekomponenter?

corners_y, corners_x = extract_local_maxima(response, 0.001)

###########################################
#
# Figure 3.1: Display Harris-Stephens corner strength
#
###########################################
plt.figure(figsize=(13,5))
plt.imshow(response)
plt.colorbar(label='Corner strength')
plt.tight_layout()
# plt.savefig('out_corner_strength.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

###########################################
#
# Figure 3.4: Display extracted corners
#
###########################################
plt.figure(figsize=(10,5))
plt.imshow(I, cmap='gray')
plt.scatter(corners_x, corners_y, linewidths=1, edgecolor='black', color='yellow', s=9)
plt.tight_layout()
# plt.savefig('out_corners.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

plt.show()

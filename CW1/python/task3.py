import numpy as np
import matplotlib.pyplot as plt
import scipy
from filtering import * 

threshold = 0.03   # todo: choose an appropriate value
sigma     = 2 # todo: choose an appropriate value
filename  = '../data/grid.jpg'



I_rgb = plt.imread(filename)
I_rgb = I_rgb/255.0
I_gray = rgb_to_gray(I_rgb)
I_blur = gaussian(I_gray, sigma)
Ix, Iy, Im = central_difference(I_blur)
x, y, theta = extract_edges(Ix, Iy, Im, threshold)

x, y = y, I_rgb.shape[0] - x
fig, axes = plt.subplots(1,6,figsize=[15,4], sharey='row')
plt.set_cmap('gray')
axes[0].imshow(I_rgb)
axes[1].imshow(I_blur)
axes[2].imshow(Ix, vmin=-0.05, vmax=0.05)
axes[3].imshow(Iy, vmin=-0.05, vmax=0.05)
axes[4].imshow(Im, vmin=+0.00, vmax=0.10, interpolation='bilinear')
edges = axes[5].scatter(x, y, s=1, c=theta, cmap='hsv')
fig.colorbar(edges, ax=axes[5], orientation='horizontal', label='$\\theta$ (radians)')
for a in axes:
    a.set_xlim([300, 600])
    a.set_ylim([I_rgb.shape[0], 0])
    a.set_aspect('equal')
axes[0].set_title('Original input')
axes[1].set_title('Blurred input')
axes[2].set_title('Gradient in x')
axes[3].set_title('Gradient in y')
axes[4].set_title('Gradient magnitude')
axes[5].set_title('Extracted edges')
plt.tight_layout()
# plt.savefig('out_edges.png') # Uncomment to save figure to working directory
plt.show()


def rgb_to_gray(I_rgb):
    return np.mean(I_rgb, axis=-1)

#Return the gradient images and gradient magnitude for a grayscaled image
def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    kernel = np.array([[1/2, 0, -1/2]])
    Ix = scipy.signal.convolve2d(I, kernel, mode='same', boundary='symm')
    Iy = scipy.signal.convolve2d(I, kernel.T, mode='same', boundary='symm')
    Im = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, Im

#Convolve a grayscale image with the 2-d Gaussian
def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations:
    h = 2*np.ceil(3*sigma) + 1.
    x, y = np.meshgrid(*([np.arange(-h, h+1)]*2), indexing='ij')
    kernel = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    result = scipy.signal.convolve2d(I, kernel.T, mode='same', boundary='symm')
    return result
    

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    args = np.where(Im >= threshold)
    out = np.stack((*args, np.arctan2(Iy[args], Ix[args])))
    return out
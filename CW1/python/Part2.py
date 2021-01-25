# -*- coding: utf-8 -*-
"""

Coursework 1 - Assignment 2
Created on Mon Jan 18 13:17:41 2021

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from PIL import Image

import sys 
import skimage.color
import skimage.io
import skimage.viewer

# %% 2.1) Load an image and print height and width
# myImg = plt.imread('../data/grass.jpg')

img = Image.open('../data/grass.jpg')

#img.show()

print("width, height, mode and format are: ", img.size, img.mode, img.format)


# %% 2.2) Plot the three single-channel images and determine which is the green channel
img = mpimg.imread('../data/grass.jpg')
#print(img2b)
imgplot = plt.imshow(img)

plot1 = plt.figure(1)
plt.imshow(img[:,:,0], cmap='gray')

plot2 = plt.figure(2) #This is the green channel
plt.imshow(img[:,:,1], cmap='gray')

plot3 = plt.figure(3)
plt.imshow(img[:,:,2], cmap='gray')

# %% 2.3) Thresholding: Find a threshold that isolates all the pixels belonging to the sugar beet leaves
# print (img[:, :, 1] > 150)

image = skimage.io.imread('../data/grass.jpg', as_gray=True)
# viewer = skimage.viewer.ImageViewer(image)
# viewer.show()

histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plot4 = plt.figure(4)
plt.plot(bin_edges[0:-1], histogram)  # <- or here

# For this image it is not possible to find a threshold that isolates the pixels of the sugar beet leaves
# As it is hard to differ the luminance and hue of the colour in the green channel, thus we can't use
# it alone as a way to determine greenness. The histogram below also illustrates two peaks, I'm not quite sure
# what it displays, but my guess is that it can say something about that there are no single threhsold value 
# that holds.


# %% 2.4) Normalized rgb

img_normalized = img / np.sum(img, axis=-1)[:,:,None]
plot5 = plt.figure(5)
plt.imshow(img_normalized)

plot6 = plt.figure(6)
plt.imshow(img[:,:,0], cmap='gray')

plot7 = plt.figure(7) #This is the green channel
plt.imshow(img[:,:,1], cmap='gray')

plot8 = plt.figure(8)
plt.imshow(img[:,:,2], cmap='gray')


# plt.imshow(img[:,:,0], cmap='gray')
# %% 2.5 Threshold the normalized rgb image


histogram, bin_edges = np.histogram(img_normalized, bins=256, range=(0, 1))

# configure and draw the histogram figure (to display where the threshold might be)
plot9 = plt.figure(9)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.plot(bin_edges[0:-1], histogram)  # <- or here+
plt.show()

# Print the thresholded image

plot10 = plt.figure(10)
plt.imshow(img_normalized[:,:,1] > 0.43, cmap='gray')




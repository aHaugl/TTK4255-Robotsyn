# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:18:33 2021

@author: Andreas
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
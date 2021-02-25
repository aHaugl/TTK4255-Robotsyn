import matplotlib.pyplot as plt
import numpy as np
from common import *

K           = np.loadtxt('../data/K.txt')
detections  = np.loadtxt('../data/detections.txt')
XY          = np.loadtxt('../data/XY.txt').T
n_total     = XY.shape[1] # Total number of markers (= 24)

fig = plt.figure(figsize=plt.figaspect(0.35))
XY = np.vstack((XY, np.ones(n_total)))
#for image_number in range(23): # Use this to run on all images
for image_number in [4]: # Use this to run on a single image

    # Load data
    # valid : Boolean mask where valid[i] is True if marker i was detected
    #     n : Number of successfully detected markers (<= n_total)
    #    uv : Pixel coordinates of successfully detected markers
    valid = detections[image_number, 0::3] == True
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    uv = uv[:, valid]
    n = uv.shape[1]

    # Tip: The 'valid' array can be used to perform Boolean array indexing,
    # e.g. to extract the XY values of only those markers that were detected.
    # Use this when calling estimate_H and when computing reprojection error.

    # Tip: Helper arrays with 0 and/or 1 appended can be useful if
    # you want to replace for-loops with array/matrix operations.
    uv = np.vstack((uv, np.ones(n)))
    
    # XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))

    
    xy = np.linalg.inv(K)@uv
    xy = xy/xy[2,:]
    
    # TASK: Implement this function
    H = estimate_H(xy.T, XY[:, valid].T)   
    uv_hat = (K @ H @ XY)
    
    # TASK: Compute predicted pixel coordinates using H
    uv_from_H = (uv_hat/uv_hat[2,:]).T 
    
    # TASK: Implement this function
    T1,T2 = decompose_H(H) 

  
    # TASK: Choose solution (try both T1 and T2 for Task 3.1, but choose automatically for Task 3.2)
    T = determine_pose(T1, T2) 
    error = reprojection_error(uv, uv_from_H.T)
    print("Max error: ", np.max(error))
    print("Min error: ", np.min(error))
    print("Avg error: ", np.average(error))
    
    # The figure should be saved in the data directory as out0000.png, etc.
    # NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
    plt.clf()
    generate_figure(fig, image_number, K, T, uv, uv_from_H.T, XY)
    plt.savefig('../data/out%04d.png' % image_number)

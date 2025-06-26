'''
smoothing.py
Author: Aditya Mallepalli
Date: 9/5/24

This function smooths a grayscaled image using a 3x3 matrix divided by 9
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

#read in image using opencv
img = cv2.imread("testImage.jpg")

#get the dimensions of the image
height, width, _ = img.shape

#create empty grayimage matrix 
grayimage = np.zeros((height, width), dtype=img.dtype)

#loop through every pixel and apply the BT.709 formula on them.
#reconstruct the grayscale image with the new grayscale values.
for i in range(height):
    for j in range(width):
        #grabbing the blue green and red intensities per pixel
        blue = img[i, j][0]
        green = img[i, j][1]
        red = img[i, j][2]
        
        #applying BT.709 formula and reconstructing image
        grayscale = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        grayimage[i, j] = grayscale

#creating a smoothing kernel by taking its mean
smoothing_kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=np.float32) / 9

#empty image
smoothed_image = np.zeros((height, width), dtype=img.dtype)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        #extracts a 3x3 window that we will apply filter to
        window = grayimage[i-1:i+2, j-1:j+2]

        #multipy the window by the smoothing kernel and add up matrix
        smoothed_pixel = np.sum(window * smoothing_kernel)

        #reconstruct image with smoothed pixels
        smoothed_image[i, j] = smoothed_pixel
        
#normailzes the pixel values so that it is between 0 and 255
smoothed_image = cv2.normalize(smoothed_image, None, 0, 255, cv2.NORM_MINMAX)

#display the smoothed image
plt.imshow(smoothed_image, cmap='gray')
plt.show()
'''
hog.py
Author: Aditya Mallepalli
Date: 9/5/24

This function implements HOG
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#read in image using opencv
img = cv2.imread("../test_img_resized.png")

#get the dimensions of the image
height, width, _ = img.shape
pixels_per_cell = 8


#resize the image in terms of 2:1 or 1:2 aspect ratio depending on image size
if (width > height):
    width = 128
    height = 64
else:
    width = 64
    height = 128

cells_x = width // pixels_per_cell
cells_y = height // pixels_per_cell
resized_image = cv2.resize(img, (width, height))

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

#compute the gradients in the x and y directions

'''
Sobel edge detection taken from Donny Weintz
'''

# kernels
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# based on kernel size, find dimension of new image (crop)
dim_x = grayimage.shape[0]
dim_y = grayimage.shape[1]

# arrays for gradient results
grad_x = np.zeros((dim_x, dim_y))
grad_y = np.zeros((dim_x, dim_y))
total_grad = np.zeros((dim_x, dim_y))

# perform window operation
for i in range(dim_x - 2):
    for j in range(dim_y - 2):
        grad_x[i, j] = np.sum(
            sobel_kernel_x * img[i:i + sobel_kernel_x.shape[0], j:j + sobel_kernel_y.shape[1]])
        grad_y[i, j] = np.sum(
            sobel_kernel_y * img[i:i + sobel_kernel_x.shape[0], j:j + sobel_kernel_y.shape[1]])

# compute gradient and magintude
mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
ang = np.arctan2(grad_y, grad_x) * 180 / np.pi

mag = np.int32(mag)  
ang = np.int32(ang % 180)  


#create empty feature vector
feature_vector = []

#looping through the each cell
for i in range(cells_y):
    for j in range(cells_x):
        #grab the current magnitues and angles of the 8x8 cell
        curr_magnitudes = mag[i * pixels_per_cell : (i+1) * pixels_per_cell, j * pixels_per_cell : (j+1) * pixels_per_cell]
        curr_angles = ang[i * pixels_per_cell : (i+1) * pixels_per_cell, j * pixels_per_cell : (j+1) * pixels_per_cell]
        
        #create the empty histogram. seperate arrays for mangintude and angles because
        #the magnitude is the only one that is being appended to. Angle array is used for
        #finding index and splitting the magnitdues properly
        histogram_mag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        histogram_ang = [0, 20, 40, 60, 80, 100, 120, 140, 160]

        #looping through the pixels in each 8x8 cell
        for k in range(pixels_per_cell):
            for l in range(pixels_per_cell):
                #grab the pixel mangitude and angles
                pixel_mag = curr_magnitudes[k, l]
                pixel_ang = curr_angles[k, l]
                upper_bin = 0
                lower_bin = 8
                #get the upper and lower bins by looping through the angles array
                for m in range(len(histogram_ang)):
                    if (histogram_ang[m] > pixel_ang):
                        upper_bin = m
                        lower_bin = m - 1
                        break
                
                #Edge cases if the pixel angle is out of range of the angle array
                if (pixel_ang > 160):
                    histogram_mag[upper_bin] += pixel_mag * (
                                (pixel_ang - 160) / 20)
                    histogram_mag[lower_bin] += pixel_mag * (
                                (180 - pixel_ang) / 20)
                else:
                    #Splitting the values between the lower and upper bins.
                    histogram_mag[lower_bin] += pixel_mag * ((histogram_ang[upper_bin] - pixel_ang) / (histogram_ang[upper_bin] - histogram_ang[lower_bin]))
                    histogram_mag[upper_bin] += pixel_mag * ((pixel_ang - histogram_ang[lower_bin]) / (histogram_ang[upper_bin] - histogram_ang[lower_bin]))
        
        #adding histogram of mangitudes to the feature vector
        feature_vector.append(histogram_mag)

feature_vector = np.array(feature_vector)
print(feature_vector)

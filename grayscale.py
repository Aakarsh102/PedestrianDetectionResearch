'''
grayscale.py
Author: Aditya Mallepalli
Date: 9/5/24

This function grayscales an image using the BT.709 formula
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

def grayimage(img_path):
    #get the dimensions of the image
    img = cv2.imread(img_path)
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
    
    return grayimage
'''
img = cv2.imread('testImage.jpg')
gray = grayimage(img)
#display the grayscale image
plt.imshow(gray, cmap="gray")
plt.show()
'''
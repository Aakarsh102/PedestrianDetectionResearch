'''
grayscale.py
Author: Aditya Mallepalli
Date: 9/7/24

This script is used to delete some of the images without pedestrians to make it
only be 10% of our total data.
'''
import os

#path of the file where images are stored
folder_path = '../ad_train/ad_all/'

#storing the annotations text file with only dashcam images to an array
dataset = open("SVM_pyerragu/scrubbed_train_bbox.txt", "r")
dataset_arr = dataset.readlines()
dataset.close()

#counter for pictures with no pedestrians
no_pedestrians = 0

#this code segment loops through the text file and gets rid of 27479 images
#with no pedestrians, cleaning up the data to have 10% images with no pedestrains
for i in range(len(dataset_arr)):
    if " " not in dataset_arr[i]:
        no_pedestrians += 1
        if no_pedestrians == 27479:
            break
        else:
            os.remove(folder_path + dataset_arr[i].strip())

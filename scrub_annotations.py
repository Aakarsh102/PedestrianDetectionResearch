'''
scrub_annotations.py
Author: Preetham Yerragudi
Date: 09/05/2024


This script removes surveillance annotations from the annotation txt file and also collected data about our dataset
'''


total_images = 0  # The total number of images excluding the surveillance images
pedestrian_images = 0  # The number of images containing pedestrians excluding the surveillance images
# adding all the lines of data into an array for further use
dataset = open("train_bbox.txt", "r")
dataset_arr = dataset.readlines()
dataset.close()
# writing to the file called scrubbed_train_bbox.txt
dataset = open("SVM_pyerragu/scrubbed_train_bbox.txt", "w")
for i in range(len(dataset_arr)):
   # if the line doesn't contain sur (which means that it's from a surveillance camera) than we write it to the file
   if not "sur" in dataset_arr[i]:
       dataset.write(dataset_arr[i])
       total_images += 1
       # if the line has a space, that means it has a pedestrian
       if " " in dataset_arr[i]:
           pedestrian_images += 1
dataset.close()


num_images_without_ped = total_images - pedestrian_images # the number of images without pedestrians is total images minus the number of images with pedestrians
no_pedestrian_percentage = num_images_without_ped / total_images # the percentage of total images without pedestrians
num_images_removed = (num_images_without_ped - (0.1 * total_images)) / 0.9; # the number of images that need to be removed to make the annotated images 10% of the dataset


# printing out some statistics about the dataset for further scrubbing of the dataset
print(f"The current # of images without pedestrians is {num_images_without_ped}")
print(f"The current % of images without any pedestrians is {round(no_pedestrian_percentage * 100)}%")
print(f"The # of images that you need to get rid of are {num_images_removed}")
print(f"The # of images without pedestrians would be {num_images_without_ped - num_images_removed} and the number of images with pedestrians would be {pedestrian_images}")
print(f"The # of images left would be {total_images - num_images_removed}")

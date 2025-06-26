'''
Author: Aditya Mallepalli
Data: 10/14/24

This code takes in an image and extracts the pedestrian using the annotations
text file. It also resizes them to be consistent sizes so that we can put them
into HOG. It also extracts random sections in an image if there is no pedestrian.
'''
import cv2
from matplotlib import pyplot as plt
import os
import random

#function to extract pedestrians from their bounding boxes using annotations file so that they can be fed to hog.
#also extracts 4 random images from the bounding boxes without any pedestrians
def extract_pedestrians_and_subimages(annotations_path, img_path, num_images, pedestrian_percentage):
    #final array that stores the images and their classifications
    cropped_images = []
    #annotations dataset
    dataset = open(annotations_path, "r")
    dataset_arr = dataset.readlines()
    dataset.close()
    #loop through the annotations file
    num_pedestrians = int(pedestrian_percentage * num_images)
    num_non_pedestrians = int((1 - pedestrian_percentage) * num_images)
    ped_count = 0
    nonped_count = 0
    for i in range(len(dataset_arr)):
        #exit condition
        if ped_count >= num_pedestrians and nonped_count >= num_non_pedestrians:
            break
        #case if there are pedestrians
        if " " in dataset_arr[i]:
            if ped_count >= num_pedestrians:
                continue
            #splits the line by ' ' so that we can get name and coordinates
            #also get the path of the image and grayscale it with the name
            parts = dataset_arr[i].split(' ')
            curr_name = parts[0]
            curr_img_path = os.path.join(img_path, curr_name)
            img = cv2.imread(curr_img_path)
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #get the coordinates of the of pedestrains
            numbers = list(map(int, parts[1:]))
            coordinates = [(numbers[i], numbers[i+1], numbers[i+2], numbers[i+3]) for i in range(0, len(numbers), 4)]
            #loop to extract the pedestrians based on their coordinates also resizes them
            for i, (x1, y1, x2, y2) in enumerate(coordinates):
                #extracts the pedestrian and the top and bottom half
                top_half = grayimg[y1: y1 + int(y2 / 2), x1: x1 + x2]
                bottom_half = grayimg[y1 + int(y2 / 2): y1 + y2, x1: x1 + x2]
                cropped_img = grayimg[y1: y1 + y2, x1:x1 + x2]
                height, width = 128, 64 
                #resizing pedestrian and storing them in the final array that is returned
                top_resize = cv2.resize(top_half, (width, height))
                bottom_resize = cv2.resize(bottom_half, (width, height))
                resized_image = cv2.resize(cropped_img, (width, height))
                cropped_images.append([top_resize, 1])
                cropped_images.append([bottom_resize, 1])
                cropped_images.append([resized_image, 1])
                ped_count += 3
        #case where there are no pedestrians
        else:
            if nonped_count >= num_non_pedestrians:
                continue
            #in the case of no pedestrians don't need to split the line since only the name exits
            #grab image name and grayscale it
            curr_name = str(dataset_arr[i]).strip()
            curr_img_path = os.path.join(img_path, curr_name)
            img = cv2.imread(curr_img_path)
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # #set some of the constant values that will be used to extract random subimages in the non pedestrains images
            number_sub_images = 4
            min_width = 10
            min_height = 20
            max_width = 128
            max_height = 256
            # fraction_of_height = 3

            height, width = grayimg.shape
            # obtain middle 1/3 of image with respect to full image height
            # start_idx = height // fraction_of_height
            # end_idx = 2 * (width // fraction_of_height)

            # middle_third = grayimg[start_idx:end_idx, :]
            sub_images = []
            # # find 4 sub-images in the middle region
            # for i in range(number_sub_images):
            #     sub_image = middle_third[0:height, (width // number_sub_images) * i:(width // number_sub_images) * (i + 1)]
            #     sub_images.append(sub_image)
            for i in range(number_sub_images):
                region_width = random.randint(min_width, max_width)
                region_height = random.randint(min_height, max_height)

                x = random.randint(0, width - region_width)
                y = random.randint(0, height - region_height)

                sub_images.append(img[y:y + region_height, x:x + region_width])
            # resize the images
            for i in range(len(sub_images)):
                height, width = 128, 64 
                sub_images[i] = cv2.cvtColor(sub_images[i], cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(sub_images[i], (width, height))
                cropped_images.append([resized, -1])
            nonped_count += number_sub_images
    print('Finished Preprocessing')
    print('Pedestrians', ped_count)
    print('Non Pedestrians', nonped_count)
    return cropped_images

# cropped_images = extract_pedestrians_and_subimages('scrubbed_train_bbox.txt', '', 8, .5)
# print(len(cropped_images))
# plt.imshow(cropped_images[6][0], cmap='gray')
# plt.show()

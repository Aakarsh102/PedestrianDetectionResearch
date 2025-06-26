'''
Author: Aditya Mallepalli
Data: 10/14/24

This code implements the sliding window technique to detect pedestrians and draw
bounding boxes over them efficiently.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn import svm
from joblib import load
import pickle as pkl
# loading the SVM classifier
svm_classifier = load("pedestrian_detector.joblib")

# Function to run sliding window operation
def sliding_window(img_path, window_size, stride, scale_factor):
    # these are the x and y dimensions of the window sizes that will be passed in as a parameter
    window_x = window_size[0]
    window_y = window_size[1]
    
    #read the image, gray it, and blur it
    img = cv2.imread(img_path)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    height, width = img.shape
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    # hog = cv2.HOGDescriptor((64, 128), (32, 32), (16, 16), (16, 16), 9)
    #this is a counter that keeps track of the current scale
    current_scale = 1.0
    bounding_boxes = []
    #While statement to break out of the image based on the dimensions of the image
    while(height > window_y and width > window_x):
        print('width: ', width)
        print('height', height)
        
        #loop to loop through the image for the sliding window
        for y in range(0, (height - window_y), stride):
            for x in range(0, (width - window_x), stride):
                #crop out the image window and get the feature vectors
                img_window = img[y: y + window_y, x: x + window_x]
                img_window = cv2.resize(img_window, (64, 128))
                feature_vectors = hog.compute(img_window)
                #reshape feature vector
                feature_vectors = feature_vectors.reshape(1, -1)
                #classify the feature vector
                classification = svm_classifier.predict(feature_vectors)
                #detects pedestrian
                if classification == 1:
                    #calculates how far the prediction is from the decision boundary
                    threshold_score = svm_classifier.decision_function(feature_vectors)
                    #thresholding to make sure not to get too many false positvies
                    if threshold_score > 0.99:
                        #rescales the points of the bounding box so that they are drawn correct
                        x_1 = int(x * current_scale)
                        x_2 = int((x + window_x) * current_scale)
                        y_1 = int(y * current_scale)
                        y_2 = int((y + window_y) * current_scale)
                        bounding_boxes.append([x_1, y_1, x_2, y_2, threshold_score])
                        #draw the rectangle
                        # cv2.rectangle(img_copy, (x_1, y_1), (x_2, y_2), (0, 255, 255), 2)
                        # plt.imshow(img_copy)
                        # plt.show()
        #scale down the image
        height = int(height / scale_factor)
        width = int(width / scale_factor)
        #track the current scale
        current_scale *= scale_factor

        img = cv2.resize(img, (width, height))
    bounding_boxes = non_max_suppression(bounding_boxes, 0.3)
    for box in bounding_boxes:
        x_1, y_1, x_2, y_2 = box[:4]
        cv2.rectangle(img_copy, (x_1, y_1), (x_2, y_2), (0, 255, 255), 2)
    return img_copy

def non_max_suppression(bounding_boxes, iou_threshold):
    new_bounding_boxes = []
    bounding_boxes = sorted(bounding_boxes, reverse=True, key = lambda x : x[4])
    while len(bounding_boxes) > 0:
        current_box = bounding_boxes.pop(0)
        new_bounding_boxes.append(current_box)
        for box in bounding_boxes:
            iou = IOU(current_box[:4], box[:4])
            if iou > iou_threshold:
                bounding_boxes.remove(box)
    return new_bounding_boxes

'''
Utilizing code from https://github.com/vineeth2309/IOU/blob/main/IOU.py
'''
def IOU(box1, box2):
    """ We assume that the box follows the format:
        box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
        where (x1,y1) and (x3,y3) represent the top left coordinate,
        and (x2,y2) and (x4,y4) represent the bottom right coordinate """
    x1, y1, x2, y2 = box1	
    x3, y3, x4, y4 = box2
    print("Box1:", x1, y1, x2, y2)
    print("Box2:", x3, y3, x4, y4)
    x_inter1 = max(x1, x3)
    print("x_inter1", x_inter1)
    y_inter1 = max(y1, y3)
    print("y_inter1", y_inter1)
    x_inter2 = min(x2, x4)
    print("x_inter2", x_inter2)
    y_inter2 = min(y2, y4)
    print("y_inter2", y_inter2)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    print("box 1 area:", area_box1)
    print("box 2 area:", area_box2)
    print("box inter area:", area_inter)
    print("area union", area_union)
    if (area_union == 0):
        return 1
    iou = area_inter / area_union
    return iou

img = sliding_window('ad10084.jpg', (64, 128), 32, 1.5)
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
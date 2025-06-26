
import random

import cv2
import joblib
import numpy as np

from skimage.feature import hog
from sklearn.svm import SVC

'''
Author: Jianing Wwang
Date: 11/2/2024
'''
def load_data(folder_path, file_name, samples):
    # Loads annotations file
    dataset = open(file_name, "r")
    dataset_arr = dataset.readlines()
    dataset.close()

    # Data points and their classifications
    images = []
    labels = []

    # Number of images to iterate through
    nPos = 0
    nNeg = 0

    # Read each line of scrubbed annotations file
    for i in range(len(dataset_arr)):
        annotation = dataset_arr[i].strip()
        # If image contains a bounding box
        if " " in annotation and nPos < samples:
            annotation_split = annotation.split()
            file_name = annotation_split[0]
            image = cv2.imread(folder_path + file_name)

            if image is not None:
                numbers = annotation_split[1:]
                numbers = list(map(int, numbers))
                bounding_box_arr = [numbers[i: i + 4] for i in range(0, len(numbers), 4)]
                # Loop through each bounding box
                for j in range(len(bounding_box_arr)):
                    # Coordinates
                    x, y, w, h = bounding_box_arr[j]
                    # Resize to standard dimensions and convert from BGR to RGB
                    if w >= 32 and h >= 64:
                        images.append(cv2.cvtColor(cv2.resize(image[y: y + h, x: x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
                        labels.append(1)
                        nPos += 1

        # If image contains no bounding boxes
        elif nNeg < samples + 5000:
            file_name = annotation.split()[0]
            image = cv2.cvtColor(cv2.imread(folder_path + file_name), cv2.COLOR_BGR2GRAY)

            if image is not None:
                # Height and width of entire image
                H, W = image.shape
                # Generates randomly sized boxes in random locations
                for i in range(4):
                    w = random.randint(32, 256)
                    h = w * 2
                    x = random.randint(0, W - w)
                    y = random.randint(0, H - h)
                    images.append(cv2.resize(image[y: y + h, x: x + w], (64, 128)))
                    labels.append(-1)
                    nNeg += 1

        # Break if the desired number of images have been trained
        if nPos >= samples and nNeg >= samples + 5000:
            break

    return images, labels


def train(samples):
    images, labels = load_data("../ad_train/ad_all_nonscrubbed/", "../ad_train/scrubbed_train_bbox.txt", samples)
    print("Done loading data")
    nPos = 0
    nNeg = 0
    for l in labels:
        if l == 1:
            nPos += 1
        else:
            nNeg += 1

    print("Pos:", nPos)
    print("Neg:", nNeg)

    hog_features_list = []
    hog_labels = []

    for img, label in zip(images, labels):
        # Extract HOG features and append to list
        features = hog(img,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       orientations=9,
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       feature_vector=True,
                       )
        hog_features_list.append(features)
        hog_labels.append(label)
    print("Done extracting HOG")

    X = np.array(hog_features_list)
    y = np.array(hog_labels)

    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale')
    svm_model.fit(X, y)
    # Saves the model to avoid having to rerun it
    joblib.dump(svm_model, "pedestrian_detector.pkl")
    print("Done training")

    return svm_model


if __name__ == '__main__':
    svm_model = train(5000)

# import random
# import cv2
# import joblib
# import numpy as np
# from skimage.feature import hog
# from sklearn.svm import SVC
# import os
#
#
# def augment_bounding_box(image, x, y, w, h, max_shift=10, max_scale=0.1):
#     # Apply random shift to each side
#     dx = random.randint(-max_shift, max_shift)
#     dy = random.randint(-max_shift, max_shift)
#     dw = random.randint(-int(w * max_scale), int(w * max_scale))
#     dh = random.randint(-int(h * max_scale), int(h * max_scale))
#
#     # Calculate new coordinates and dimensions, ensuring they stay within image bounds
#     new_x = max(0, x + dx)
#     new_y = max(0, y + dy)
#     new_w = max(32, min(image.shape[1] - new_x, w + dw))
#     new_h = max(64, min(image.shape[0] - new_y, h + dh))
#
#     # Return None if the new bounding box is invalid
#     if new_x + new_w > image.shape[1] or new_y + new_h > image.shape[0]:
#         return None
#
#     # Crop and resize to standard dimensions (64x128)
#     cropped_img = cv2.resize(image[new_y:new_y + new_h, new_x:new_x + new_w], (64, 128))
#     return cropped_img
#
# def load_data(folder_path, file_name, samples, hard_negatives=None):
#     dataset = open(file_name, "r")
#     dataset_arr = dataset.readlines()
#     dataset.close()
#
#     images, labels, image_paths = [], [], []
#     nPos, nNeg = 0, 0
#
#     for i in range(len(dataset_arr)):
#         annotation = dataset_arr[i].strip()
#
#         # If the image contains a bounding box (positive sample)
#         if " " in annotation and nPos < samples:
#             annotation_split = annotation.split()
#             file_name = annotation_split[0]
#             image = cv2.imread(folder_path + file_name)
#             if image is not None:
#                 numbers = list(map(int, annotation_split[1:]))
#                 bounding_boxes = [numbers[i:i + 4] for i in range(0, len(numbers), 4)]
#
#                 for box in bounding_boxes:
#                     x, y, w, h = box
#                     if w >= 32 and h >= 64:
#                         # images.append(
#                         #     cv2.cvtColor(cv2.resize(image[y: y + h, x: x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
#                         # labels.append(1)
#                         # image_paths.append(folder_path + file_name)
#                         # nPos += 1
#                         augmented_img = augment_bounding_box(image, x, y, w, h)
#                         if augmented_img is not None:
#                             images.append(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY))
#                             labels.append(1)
#                             nPos += 1
#
#         # If the image does not contain bounding boxes (negative sample)
#         elif nNeg < samples:
#             file_name = annotation.split()[0]
#             image = cv2.cvtColor(cv2.imread(folder_path + file_name), cv2.COLOR_BGR2GRAY)
#             if image is not None:
#                 H, W = image.shape
#                 for i in range(4):
#                     w = random.randint(32, 256)
#                     h = w * 2
#                     x = random.randint(0, W - w)
#                     y = random.randint(0, H - h)
#                     images.append(cv2.resize(image[y: y + h, x: x + w], (64, 128)))
#                     labels.append(-1)
#                     image_paths.append(folder_path + file_name)
#                     nNeg += 1
#
#         if nPos >= samples and nNeg >= samples:
#             break
#
#     # Add hard negatives if provided
#     if hard_negatives:
#         for img in hard_negatives:
#             images.append(cv2.resize(img, (64, 128)))
#             labels.append(-1)
#             image_paths.append("hard_negative")
#
#     return images, labels, image_paths
#
#
# def extract_hog_features(images, labels):
#     hog_features_list = []
#     hog_labels = []
#     for img, label in zip(images, labels):
#         features = hog(img,
#                        pixels_per_cell=(8, 8),
#                        cells_per_block=(2, 2),
#                        orientations=9,
#                        block_norm='L2-Hys',
#                        transform_sqrt=True,
#                        feature_vector=True)
#         hog_features_list.append(features)
#         hog_labels.append(label)
#     return np.array(hog_features_list), np.array(hog_labels)
#
#
# def detect_hard_negatives(model, new_neg_images):
#     hard_negatives = []
#     for img in new_neg_images:
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         H, W = img_gray.shape
#         for _ in range(5):  # Generate 5 random regions per image
#             w = random.randint(32, 256)
#             h = w * 2
#             x = random.randint(0, W - w)
#             y = random.randint(0, H - h)
#             region = cv2.resize(img_gray[y: y + h, x: x + w], (64, 128))
#             features = hog(region, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
#                            orientations=9, block_norm='L2-Hys', transform_sqrt=True,
#                            feature_vector=True)
#             if model.predict([features]) == 1:  # Classified as positive (false positive)
#                 hard_negatives.append(region)
#     return hard_negatives
#
#
# def load_new_negative_images(neg_folder_path, num_images):
#     neg_images = []
#     for i, img_name in enumerate(os.listdir(neg_folder_path)):
#         if i >= num_images:
#             break
#         img_path = os.path.join(neg_folder_path, img_name)
#         image = cv2.imread(img_path)
#         if image is not None:
#             neg_images.append(image)
#     return neg_images
#
#
# def train(samples):
#     images, labels, image_paths = load_data("../ad_train/ad_all_nonscrubbed/", "../ad_train/scrubbed_train_bbox.txt",
#                                             samples)
#     print("Done loading data")
#
#     X, y = extract_hog_features(images, labels)
#     print("Done extracting HOG features")
#
#     svm_model = SVC(kernel='rbf', C=10.0, gamma='scale')
#     svm_model.fit(X, y)
#     print("Initial training complete")
#
#     # Hard negative mining with new unseen negative images
#     new_neg_images = load_new_negative_images("../ad_train/ad_02/", 5000)  # Load 5000 new negative images
#     hard_negatives = detect_hard_negatives(svm_model, new_neg_images)
#
#     # Retrain with hard negatives
#     if hard_negatives:
#         images, labels, _ = load_data("../ad_train/ad_all_nonscrubbed/", "../ad_train/scrubbed_train_bbox.txt", samples,
#                                       hard_negatives)
#         X, y = extract_hog_features(images, labels)
#         svm_model.fit(X, y)
#         print("Retraining with hard negatives complete")
#
#     # Save the model
#     joblib.dump(svm_model, "pedestrian_detector_with_hard_negatives_offset.pkl")
#     print("Model saved with hard negative mining applied")
#
#     return svm_model
#
#
# if __name__ == '__main__':
#     svm_model = train(5000)

import json
import os

import cv2
import random

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC

images_file_path = "../leftImg8bit_trainvaltest/peeps/"
annotations_file_path = "../leftImg8bit_trainvaltest/peeps_annotations/"


def generate_random_box(W, H, existing_bboxes):
    max_attempts = 1000  # Limit attempts to avoid infinite loops

    # Function to check overlap between two boxes
    def is_overlapping(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Check if there is no overlap
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    # Attempt to generate a valid box
    for _ in range(max_attempts):
        # Random width and height within the specified range
        w = random.choice([32, 64, 128, 256])
        h = w * 2

        # Random x and y ensuring the box fits within the image dimensions
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        # New box in (x, y, width, height) format
        new_box = (x, y, w, h)

        # Check if it overlaps with any existing boxes
        if all(not is_overlapping(new_box, bbox) for bbox in existing_bboxes):
            return new_box

    # If no valid box is found, return None
    return None


def load_data(pos_samples, neg_samples):
    directory = os.fsencode(images_file_path)

    images = []
    labels = []
    n_pos = 0
    n_neg = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        annotation = filename.replace("leftImg8bit.png", "gtBboxCityPersons.json")

        img = cv2.imread(images_file_path + filename)
        H, W, _ = img.shape

        with open(annotations_file_path + annotation, 'r') as file:
            data = json.load(file)
        filtered_bbox_list = [obj['bbox'] for obj in data['objects'] if obj['label'] != "ignore"]

        if n_pos < pos_samples:
            for x, y, w, h in filtered_bbox_list:
                if x >= 0 and y >= 0 and w >= 32 and h >= 64:
                    images.append(cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
                    labels.append(1)
                    n_pos += 1

        if n_neg < neg_samples:
            for i in range(4):
                x, y, w, h = generate_random_box(W, H, filtered_bbox_list)
                images.append(cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
                labels.append(-1)
                n_neg += 1

        print(n_pos, n_neg)

        if n_pos >= pos_samples and n_neg >= neg_samples:
            break

    return images, labels


def train(pos_samples, neg_samples):
    images, labels = load_data(pos_samples, neg_samples)
    print("Done loading data")

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
                       feature_vector=True
                       )
        hog_features_list.append(features)
        hog_labels.append(label)
    print("Done extracting HOG")

    X = np.array(hog_features_list)
    y = np.array(hog_labels)

    svm_model = SVC(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X, y)
    # Saves the model to avoid having to rerun it
    joblib.dump(svm_model, "../pedestrian_detector_new_dataset_weighted.pkl")
    print("Done training")

    return svm_model


def train_hard_negative(pos_samples, neg_samples):
    directory = os.fsencode(images_file_path)

    images = []
    labels = []
    n_pos = 0
    n_neg = 0

    svm_model = SVC(kernel='rbf', C=10, gamma='scale')
    trained = False

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        annotation = filename.replace("leftImg8bit.png", "gtBboxCityPersons.json")

        img = cv2.imread(images_file_path + filename)
        H, W, _ = img.shape

        with open(annotations_file_path + annotation, 'r') as file:
            data = json.load(file)
        bbox_list = [obj['bbox'] for obj in data['objects'] if obj['label'] != "ignore"]
        pedestrian_bbox_list = [obj['bbox'] for obj in data['objects'] if obj['label'] != "ignore" and obj['label'] != "person group"]

        if n_pos < pos_samples or n_neg < int(neg_samples * 0.75):
            if n_pos < pos_samples:
                for x, y, w, h in pedestrian_bbox_list:
                    if x >= 0 and y >= 0 and w >= 32 and h >= 64:
                        padding = 5
                        if x + w + padding < W:
                            w += padding
                        if y + h + padding < H:
                            h += padding
                        if x - padding > 0:
                            x -= padding
                        if y - padding > 0:
                            y -= padding
                        images.append(cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
                        labels.append(1)
                        n_pos += 1

            if n_neg < int(neg_samples * 0.75):
                for i in range(4):
                    x, y, w, h = generate_random_box(W, H, bbox_list)
                    images.append(cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (64, 128)), cv2.COLOR_BGR2GRAY))
                    labels.append(-1)
                    n_neg += 1

            print(n_pos, n_neg)
        else:
            if not trained:
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
                                   feature_vector=True
                                   )
                    hog_features_list.append(features)
                    hog_labels.append(label)

                X = np.array(hog_features_list)
                y = np.array(hog_labels)

                svm_model.fit(X, y)
                trained = True
            else:
                if n_neg < neg_samples:
                    for i in range(8):
                        x, y, w, h = generate_random_box(W, H, pedestrian_bbox_list)
                        image = cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (64, 128)), cv2.COLOR_BGR2GRAY)
                        features = hog(image,
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2),
                                       orientations=9,
                                       block_norm='L2-Hys',
                                       transform_sqrt=True,
                                       feature_vector=True
                                       )
                        features = features.reshape(1, -1)
                        if svm_model.decision_function(features) > 0.5:
                            images.append(image)
                            labels.append(-1)
                            n_neg += 1
                else:
                    break

    print("Done loading data")

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

    svm_model.fit(X, y)
    # Saves the model to avoid having to rerun it
    joblib.dump(svm_model, "../pedestrian_detector_new_dataset_hard_negative_threshold_2.pkl")
    print("Done training")

    return svm_model



if __name__ == '__main__':
    # svm = train(10000, 10000)
    svm = train_hard_negative(5000, 10000)

import numpy as np
import matplotlib.pyplot as plt

'''
Author: Jianing Wwang
Date: 10/14/2024
'''

# Dataset directory
folder_path = '../ad_train/ad_all_nonscrubbed/'

# Loads annotations file
dataset = open("SVM_pyerragu/scrubbed_train_bbox.txt", "r")
dataset_arr = dataset.readlines()
dataset.close()


def remove_outliers(data):
    data = np.array(data)

    # Calculate Q1 and Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    # Calculate the IQR
    IQR = Q3 - Q1
    # Outlier fences
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove data points that are below the lower fence or above the upper fence
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    return filtered_data

nBig = 0

aspectRatioList = []
areaList = []
widthList = []
heightList = []
for i in range(len(dataset_arr)):
    annotation = dataset_arr[i].strip()
    # If image contains a bounding box
    if " " in annotation:
        annotation_split = annotation.split()
        numbers = annotation_split[1:]
        numbers = list(map(int, numbers))
        bounding_box_arr = [numbers[i: i + 4] for i in range(0, len(numbers), 4)]
        # Loop through each bounding box
        for j in range(len(bounding_box_arr)):
            # Coordinates
            x, y, w, h = bounding_box_arr[j]
            aspectRatioList.append(w / h)
            areaList.append(w * h)
            widthList.append(w)
            heightList.append(h)
            if w >= 32 and h >= 64:
                nBig += 1

aspectRatioList = remove_outliers(aspectRatioList)
areaList = remove_outliers(areaList)
widthList = remove_outliers(widthList)
heightList = remove_outliers(heightList)

print("nBig:", nBig)
print("Aspect Ratio Mean:", np.mean(aspectRatioList))
print("Aspect Ratio Median:", np.median(aspectRatioList))
print("Aspect Ratio Min:", np.min(aspectRatioList))
print("Aspect Ratio Max:", np.max(aspectRatioList))
print()
print("Area Mean:", np.mean(areaList))
print("Area Median:", np.median(areaList))
print("Area Min:", np.min(areaList))
print("Area Max:", np.max(areaList))
print()
print("Width Mean:", np.mean(widthList))
print("Width Median:", np.median(widthList))
print("Width Min:", np.min(widthList))
print("Width Max:", np.max(widthList))
print()
print("Height Mean:", np.mean(heightList))
print("Height Median:", np.median(heightList))
print("Height Min:", np.min(heightList))
print("Height Max:", np.max(heightList))

# Plots histogram
plt.hist(aspectRatioList)
plt.title("Histogram of aspect ratio of pedestrian bounding box")
plt.ylabel("Frequency")
plt.xlabel("w / h")
plt.show()

# Plots histogram
plt.hist(widthList)
plt.title("Histogram of width of pedestrian bounding box")
plt.ylabel("Frequency")
plt.xlabel("w")
plt.show()

# Plots histogram
plt.hist(heightList)
plt.title("Histogram of height of pedestrian bounding box")
plt.ylabel("Frequency")
plt.xlabel("h")
plt.show()
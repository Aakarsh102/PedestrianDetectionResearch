# Jianing Wang 9/6/2024

import cv2

# Constants
BOX_COLOR = (0, 255, 255)
BOX_THICKNESS = 2

# Loads annotations file
dataset = open("SVM_pyerragu/scrubbed_train_bbox.txt", "r")
dataset_arr = dataset.readlines()
dataset.close()

# Draws a bounding box around each pedestrian labeled in the annotations file
def draw_bounding_box(img, name):
    # Creates a copy of img
    img_bounding_box = img.copy()
    for i in range(len(dataset_arr)):
        if name in dataset_arr[i]:
            annotation = dataset_arr[i]
            print(annotation)
            # Checks if the annotation contains any bounding boxes
            if " " in dataset_arr[i]:
                # Interprets the annotation and returns a matrix of coordinates
                # Used OpenAI
                numbers = annotation.split()[1:]
                numbers = list(map(int, numbers))
                bounding_box_arr = [numbers[i: i + 4] for i in range(0, len(numbers), 4)]
                print(bounding_box_arr)
                # Loops through each bounding box
                for j in range(len(bounding_box_arr)):
                    # x y w h
                    x, y, w, h = bounding_box_arr[j]
                    # Draws bounding boxes with the coordinates
                    cv2.rectangle(img_bounding_box, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
            break
    return img_bounding_box


if __name__ == '__main__':
    # File information
    file_name = "ad40907.jpg"
    file_path = "../ad_train/ad_all/"

    img_ad = cv2.imread(file_path + file_name)

    cv2.imshow(file_name, draw_bounding_box(img_ad, file_name))

    # Delete windows on key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

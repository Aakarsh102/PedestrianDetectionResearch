import cv2
import numpy as np
from block_normalizer import block_normalize

path_to_image = "this_image_dude.jpeg"
bins = 9
window_for_cell = 8
cell_size = 8
image = cv2.imread(path_to_image)
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Compute the gradients in the x and y directions
# took the help of chatgpt for this part 
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

# This function makes calculating gradients and angles much faster and easier. 
# We're focussing on implementing HOG so we're not going to be implementing this function
mag, ang = cv2.cartToPolar(gx, gy)



# We're doing bin - 1 because bins corresponds to the numbers not not the spaces between them
# we need to normalize the angles to be in the range [0, bins - 1]
bin_values = ((bins - 1) * ang) / (2 * (np.pi))
# Now we'll be constructing a histograms of gradients for each cell
def create_histograms(mag, bin_values, cell_size, block_size, gray):
    h, w = gray.shape
    # Calculate the number of cells in both dimensions
    num_cells_x = w // cell_size
    num_cells_y = h // cell_size
    
    # Initialize the feature vector to hold the histograms for all cells
    feature_vector = []
    # count = 0
    
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Extract the magnitude and bin values for the current cell
            cell_mag = mag[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_bin = bin_values[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            
            # Initialize the histogram for this cell
            hist = np.zeros(bins)
            feature_desciptor = np.zeros(block_size * bins)
            # Populate the histogram by summing the magnitudes for each bin
            for x in range(cell_size):
                for y in range(cell_size):
                    # if (int(cell_bin[x, y]) - cell_bin[x, y] >= 1):
                    #     print(cell_bin[x, y])
                    #     print("bro something is totally wrong!!")

                    # This part is handling the division of magnitudes between the current and next bin
                    bin_idx = int(cell_bin[x, y]) % bins
                    extra = (cell_bin[x, y] - bin_idx) * cell_mag[x, y]
                    residual = cell_mag[x, y] - extra
                    hist[bin_idx] += residual
                    hist[bin_idx + 1] += extra 
            
            feature_vector.append(hist)
            # #print(feature_desciptor.shape)
            # feature_desciptor[count * bins: (count + 1) * bins] = hist
            
            # count += 1
            # if (count == block_size):
            #     print(feature_desciptor.shape)
            #     count = 0
            #     # print("#########")
            #     # print(len(np.array(feature_desciptor).flatten()))
            #     # print(np.array(feature_desciptor).flatten())
            #     # print(hist.shape)
            #     # print("$$$$$$$")
            #     # feature_desciptor = np.array(feature_desciptor)
            #     # feature_vector.append(feature_desciptor)
            #     # feature_desciptor = []
                
            
    
    return feature_vector

l = create_histograms(mag, bin_values, cell_size, 4, gray)
# range_of_data = 173 ** 2
# print((np.array(l)[:range_of_data].reshape(173, 173, 9)).shape)
# print(int(np.sqrt(len(l))))  # Check the number of cells in one dimension
block_features = block_normalize(l, 4)
final_feature_vector = block_features.flatten()
print(final_feature_vector.shape)  # Check the shape of the final feature vector
# print(ang)
# print(mag.shape, gray.shape)
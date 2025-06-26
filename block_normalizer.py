import numpy as np

def block_normalize(feature_vectors, block_size):

    num_cells = len(feature_vectors)
    cells_per_row = int(np.sqrt(num_cells))

    # Since the number of features vectors won't necessarily be a perfect square,
    # we'll be pruning of the excess cells. 
    range_considered = cells_per_row ** 2
    
    # Converting the list of features vectors into a 2d numpy array with each element
    # representing a feature vector for a cell
    cell_grid = (np.array(feature_vectors)[:range_considered]).reshape(cells_per_row, cells_per_row, -1)

    normalized_blocks = []
    for i in range(cells_per_row - block_size + 1):
        for j in range(cells_per_row - block_size + 1):

            # We're going to extract a block of size block_size x block_size
            # We'll append them all together to get just 1 vector.
            block = cell_grid[i:i+block_size, j:j+block_size].flatten()
            # This function finds the magnitude of the block vector.
            norm = np.linalg.norm(block)
            if norm != 0:
                block /= norm
            normalized_blocks.append(block)

    return np.array(normalized_blocks)


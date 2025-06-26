'''
hog_vision.py
Author: Donny Weintz
Date: 9/20/24

This script creates a visualization of the 9x1 histograms 
for each cell in HOG.
'''

'''
visualize_hog(histogram_list, x_dim, y_dim)

Args:
    histogram list (list of lists/arrays)
        A list containing 9x1 lists/arrays for each cell. The 9x1
        histograms should only contain the weighted magnitude values.
        This is assuming you use [0, 20, 40, 60, 80, 100, 120, 140, 160]
        as you bins.
    x_dim (int)
        The number of histograms on the x-axis.
    y_dim (int)
        The number of histograms on the y-axis.
    plot_size (tuple of 2 ints)
        Tuple containing the dimensions of the output plot.
        Example: (5, 10), creates 5x10 plot.
Return:
    None. Will plot a grid containing the histograms for each cell
    based on input.
'''

import matplotlib.pyplot as plt     # graphing/display
import numpy as np                  # matrices

def visualize_hog(histogram_list, x_dim, y_dim, plot_size):
    # define the angle bins
    angles1 = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    angles2 = [180, 200, 220, 240, 260, 280, 300, 320, 340]

    # for angles less than 180 degrees
    angles_rad1 = np.deg2rad(angles1)
    x_components1 = np.cos(angles_rad1)
    y_components1 = np.sin(angles_rad1)

    # for angles greater than 180 degrees
    angles_rad2 = np.deg2rad(angles2)
    x_components2 = np.cos(angles_rad2)
    y_components2 = np.sin(angles_rad2)

    # create plot for hog features
    fig, ax = plt.subplots(figsize = plot_size)
    ax.set_facecolor('black')
    ax.set_title('Histograms for Each Cell')
   
    # plot the gradient field
    histogram_idx = 0
    for y in range(y_dim, 0, -1):
        for x in range(x_dim):
            for k in range(len(histogram_list[0])):
                arrow_length = (histogram_list[histogram_idx])[k]

                ax.quiver(x, y, 
                        arrow_length * y_components1[k].T, 
                        arrow_length * x_components1[k].T, 
                        color = 'white', scale = 15, alpha = 0.7, 
                        headwidth = 1, headlength = 0.1, width = 0.003)
                ax.quiver(x, y, 
                        arrow_length * y_components2[k].T, 
                        arrow_length * x_components2[k].T, 
                        color = 'white', scale = 15, alpha = 0.7, 
                        headwidth = 1, headlength = 0.1, width = 0.003)
            histogram_idx += 1
    plt.show()
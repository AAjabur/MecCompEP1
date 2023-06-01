import numpy as np
from matplotlib import pyplot as plt

class PlotUtils:
    @staticmethod
    def plot_quiver_decreasing_density(x_matrix, y_matrix, i_matrix, j_matrix, num_of_vectors):
        subset_size = int(np.sqrt(len(x_matrix)*len(y_matrix) / num_of_vectors))

        fix, ax = plt.subplots()
        ax.quiver(
            x_matrix[::subset_size, ::subset_size],
            y_matrix[::subset_size, ::subset_size], 
            i_matrix[::subset_size, ::subset_size], 
            j_matrix[::subset_size, ::subset_size],
            angles="xy",
            pivot="mid",
            scale=1700,
            headwidth=4
        )
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_points(plotter, points, color='green', size=15):
    """
    Plot a set of points in a 3D plotter.

    Parameters:
    - plotter: pyvista.Plotter
        The 3D plotter object to add the points to.
    - points: pyvista.PolyData
        The points to be plotted.
    - color: str, optional
        The color of the points. Default is 'green'.
    - size: int, optional
        The size of the points. Default is 15.

    Returns:
    None
    """
    import pyvista
    plotter.add_mesh(points, color=color, point_size=size, render_points_as_spheres=True)


def plot_adjacency(plotter, adj_matrix, points):
    """
    Plot the adjacency of points using a given plotter.

    Parameters:
    - plotter: The plotter object used for visualization.
    - adj_matrix: The adjacency matrix representing the connections between points.
    - points: The list of points to be plotted.

    Returns:
    None
    """

    import pyvista
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))


def plot_similarity_matrix(similarity_matrix, labels, add_values=True, filename=None):
    """
    Plots a similarity matrix.

    Parameters:
    similarity_matrix (numpy.ndarray): The similarity matrix to be plotted.
    labels (list): The labels for the rows and columns of the matrix.
    add_values (bool, optional): Whether to add the values of the matrix as text. Defaults to True.
    filename (str, optional): The filename to save the plot as an image. If not provided, the plot will be displayed.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(similarity_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)

    fig.suptitle('Similarity matrix')

    if add_values:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                ax.text(j, i, format(similarity_matrix[i, j], ".2f"), 
                        ha="center", va="center", 
                        color="w" if np.abs(similarity_matrix[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_training_curves(epochs, train_loss, val_loss=None, test_loss=None, loss_label='MSE', filename=None, log_scale=False):
    """
    Plots the training curves for a given set of epochs and loss values.

    Parameters:
    - epochs (list): A list of integers representing the epochs.
    - train_loss (list): A list of floats representing the training loss values.
    - val_loss (list, optional): A list of floats representing the validation loss values. Default is None.
    - test_loss (list, optional): A list of floats representing the test loss values. Default is None.
    - loss_label (str, optional): The label for the loss values. Default is 'MSE'.
    - filename (str, optional): The filename to save the plot. Default is None.
    - log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.

    Returns:
    - None

    """
    
    fig, ax = plt.subplots(1)

    ax.plot(epochs, train_loss, label='train')
    if val_loss:
        ax.plot(epochs, val_loss, label='val')
    if test_loss:
        ax.plot(epochs, test_loss, label='test')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('epochs')
    ax.set_ylabel(loss_label) 
    ax.legend()

    fig.suptitle("Training Curves")

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_dataframe_distributions(df, save_dir=None, col_names=None):
    """Plots the distributions of the columns in a dataframe."""

    if col_names is None:
        numeric_columns = df.select_dtypes(include=['number']).columns
    else:
        numeric_columns = col_names

    num_cols = len(numeric_columns)
    
    if num_cols == 0:
        print("No numeric columns to plot.")
        return

    os.makedirs(save_dir,exist_ok=True)
    
    for i, col in enumerate(numeric_columns):
        property_name=col.split(':')[0]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Create a subplot for each type of plot

        # Histogram
        sns.histplot(df[col], kde=False, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        # Density Plot
        sns.kdeplot(df[col], ax=axes[1], fill=True)
        axes[1].set_title(f'Density Plot of {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Density')

        # Box Plot
        sns.boxplot(x=df[col], ax=axes[2])
        axes[2].set_title(f'Box Plot of {col}')
        axes[2].set_xlabel(col)
        axes[2].set_ylabel('Value')

        # Violin Plot
        sns.violinplot(x=df[col], ax=axes[3])
        axes[3].set_title(f'Violin Plot of {col}')
        axes[3].set_xlabel(col)
        axes[3].set_ylabel('Value')

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir,f'{property_name}.png'))
        else:
            plt.show()
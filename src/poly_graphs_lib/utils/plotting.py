import numpy as np
import matplotlib.pyplot as plt

def plot_points(plotter, points, color='green', size=15):
    import pyvista
    plotter.add_mesh(points,color=color,point_size=size,render_points_as_spheres=True)

def plot_adjacency(plotter, adj_matrix, points):
    import pyvista
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))


def plot_similarity_matrix( similarity_matrix, labels, add_values=True, filename=None):
    fig, ax = plt.subplots()
    cax = ax.matshow(similarity_matrix, cmap='coolwarm')  # Change the color map here
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

def plot_training_curves(epochs, train_loss, val_loss=None, test_loss=None,loss_label='MSE', filename=None):
    fig,ax = plt.subplots(1)

    ax.plot(epochs,train_loss , label = 'train')
    if val_loss:
        ax.plot(epochs,val_loss , label = 'val')
    if test_loss:
        ax.plot(epochs,test_loss , label = 'test')

    ax.set_xlabel('epochs')
    ax.set_ylabel(loss_label) 
    ax.legend()

    fig.suptitle("Training Curves")

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

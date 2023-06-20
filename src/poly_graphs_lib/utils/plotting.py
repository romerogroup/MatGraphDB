import pyvista
import numpy as np

def plot_points(plotter, points, color='green', size=15):
    plotter.add_mesh(points,color=color,point_size=size,render_points_as_spheres=True)

def plot_adjacency(plotter, adj_matrix, points):
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))

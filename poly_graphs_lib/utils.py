import numpy as np


def rot_z(theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def plot_points(plotter, points, color='green', size=15):
    import pyvista as pv

    plotter.add_mesh(points,color=color,point_size=size,render_points_as_spheres=True)

def plot_adjacency(plotter, adj_matrix, points):
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))

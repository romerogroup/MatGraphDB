import os
import shutil
import random
import json
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

import pyvista as pv
from coxeter.shapes import ConvexPolyhedron
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from voronoi_statistics.voronoi_structure import VoronoiStructure

import random

def generate_polyhedra(num_vertices):
    """
    Generates a random polyhedron with the specified number of vertices.
    Returns the list of vertices and the list of faces.
    """
    # Generate random vertices
    vertices = []
    for i in range(num_vertices):
        vertex = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        vertices.append(vertex)

    # Generate random faces
    faces = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            for k in range(j + 1, num_vertices):
                # Check if the three vertices form a valid face
                face = (i, j, k)
                if is_valid_face(vertices, face):
                    faces.append(face)

    # Remove common edges
    faces = remove_common_edges(faces)

    return vertices, faces

def is_valid_face(vertices, face):
    """
    Checks if the three vertices form a valid face.
    Returns True if the face is valid, False otherwise.
    """
    v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    # Calculate the cross product of the two edges of the face
    # If the cross product is zero, the edges are parallel and the face is invalid
    cross_product = ((v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]),
                     (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]))
    if cross_product[0][1] * cross_product[1][2] - cross_product[0][2] * cross_product[1][1] == 0 \
            and cross_product[0][2] * cross_product[1][0] - cross_product[0][0] * cross_product[1][2] == 0 \
            and cross_product[0][0] * cross_product[1][1] - cross_product[0][1] * cross_product[1][0] == 0:
        return False
    return True

def remove_common_edges(faces):
    """
    Removes common edges from the list of faces.
    Returns the updated list of faces.
    """
    updated_faces = []
    for i, face1 in enumerate(faces):
        for j, face2 in enumerate(faces):
            if i != j:
                # Check if the two faces share an edge
                shared_vertices = set(face1) & set(face2)
                if len(shared_vertices) == 2:
                    # Remove the common edge from both faces
                    edge = tuple(shared_vertices)
                    new_face1 = tuple(v for v in face1 if v not in edge)
                    new_face2 = tuple(v for v in face2 if v not in edge)
                    if len(new_face1) == 3:
                        updated_faces.append(new_face1)
                    if len(new_face2) == 3:
                        updated_faces.append(new_face2)
                else:
                    updated_faces.append(face1)
    return updated_faces

def generate_random_polyhedron_2(n_points):
    points = []
    i=0
    while i < n_points:
        # Sample spherical coordinates uniformly at random
        z = 2 * np.random.random() - 1
        t = 2 * np.pi * np.random.random()
        r = np.sqrt(1 - z**2)
        x = r * np.cos(t)
        y = r * np.sin(t)

        if i == 0 :
            points.append((x, y, z))
            i+=1
        else:
            distances = np.linalg.norm(points - np.array([x,y,z]), axis=1)

            min_distance = np.min(distances)
            if not min_distance <= 0.1:
                
                points.append((x, y, z))
                i+=1
    return np.array(points)

# vertices, faces = generate_polyhedra(num_vertices=4)
random.seed(1)
verts = generate_random_polyhedron_2(n_points=30)
verts_cube = pv.Cube().points

verts = verts
poly = ConvexPolyhedron(vertices=verts)
poly.merge_faces(atol=1e-8, rtol=1e-1)
num_faces = len(poly.faces)
print(num_faces)
plotter = pv.Plotter()

plotter.add_mesh(pv.PolyData(verts).delaunay_3d(), render_points_as_spheres = True, point_size = 15)
plotter.show()
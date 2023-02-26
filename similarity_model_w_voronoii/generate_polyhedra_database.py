import os
import shutil
import json
import itertools
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


from voronoi_statistics.similarity_measure_random import similarity_measure_random


def moment_of_inertia(points):
    """
    Calculate the moment of inertia for a set of points.

    Args:
    points (list of lists or numpy array): The set of points.

    Returns:
    float: The moment of inertia for the set of points.
    """
    n = len(points)
    
    points = (points - points.mean(axis=0))/points.std(axis=0)
    center_of_mass = points.mean(axis=0)
    distances_squared = np.sum((points - center_of_mass)**2, axis=1)
    return float(distances_squared.sum())

def coulomb_energy_per_n_verts(points):
    energy = 0
    charges = [1]*len(points)
    num_particles = len(charges)

    points = (points - points.mean(axis=0))/points.std(axis=0)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            distance = np.linalg.norm(points[i] - points[j])
            energy += charges[i] * charges[j] / distance
    return float(energy) / num_particles

def three_body_energy(points, dihedral_matrix,sep_const=1,theta=1):
    energy = 0
    num_points = len(points)

    points = (points - points.mean(axis=0))/points.std(axis=0)
    for i in range(num_points):
        for j in range(i+1, num_points):
            if dihedral_matrix[i,j] > 0:
                sep_vec = points[i] - points[j]
                
                # print(np.linalg.norm(sep_vec)**2  ,dihedral_matrix[i,j])
                energy = sep_const*0.5*np.linalg.norm(sep_vec)**2 + 0.5 * (dihedral_matrix[i,j] - theta**2) **2
    # print(energy)
    return energy

def find_common_edges(face_1, face_2):
    result = []
    for i in face_1:
        if i in face_2:
            result.append(i)
    return result

def get_edge_face_features(points):

    poly = ConvexPolyhedron(vertices=points)
    
    num_faces = len(poly.faces)
    face_adjacency_matrix = np.zeros((num_faces, num_faces))

    dihedral_matrix = np.zeros((num_faces, num_faces))
    edge_length_matrix = np.zeros((num_faces, num_faces))
    face_centers_distance_matrix = np.zeros((num_faces, num_faces))

    dihedral_angles = []
    edge_lengths = []

    face_sides = []
    face_areas = []
    face_normals = np.zeros((num_faces, 3))
    
    edges=[]
    for i_face, i_neighbors in enumerate(poly.neighbors):
        n_face_sides = len(poly.faces[i_face])
        face_sides.append(n_face_sides)
        face_areas.append(poly.get_face_area(i_face)[0])
        face_normals[i_face,:] = poly.normals[i_face,:]

        for i_neighbor in i_neighbors:

            common_edge = find_common_edges(face_1=poly.faces[i_face], face_2=poly.faces[i_neighbor])

            edge_length = np.linalg.norm(poly.vertices[common_edge[0]] - poly.vertices[common_edge[1]])
            dihedral_angle = poly.get_dihedral(i_face, i_neighbor)
            if common_edge not in edges and common_edge[::-1] not in edges:
                edges.append(common_edge)


            edge_length_matrix[i_face, i_neighbor]+=edge_length
            face_adjacency_matrix[i_face, i_neighbor]+=1
            dihedral_matrix[i_face, i_neighbor]+=dihedral_angle
            face_centers_distance_matrix[i_face, i_neighbor] += np.linalg.norm(poly.normals[i_face,:] - poly.normals[i_neighbor,:])


    return dihedral_matrix, edge_length_matrix, face_centers_distance_matrix ,face_sides,face_areas,face_normals

def plot_adjacency(plotter:pv.Plotter, adj_matrix, points):
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                # lines.append([points[i], points[j]])
                plotter.add_lines(np.array([points[i], points[j]]))

def generate_pyg_node_feature(face_areas, face_sides):
    n_faces = len(face_areas)
    x = np.dstack((face_sides,face_areas))[0].T
    x = torch.tensor(x, dtype=torch.float)
    return x

def generate_pyg_edge_feature(dihedral_matrix, edge_length_matrix):
    n_faces = dihedral_matrix.shape[0]
    edge_index = []
    edge_attr = []
    for i_face in range(n_faces):
        for j_face in range(i_face,n_faces):
            if dihedral_matrix[i_face,j_face] > 0:
                edge_index.append([i_face,j_face])
                edge_attr.append([i_face,j_face])

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)


    edge_index = torch.tensor(edge_index, dtype=torch.float).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).t().contiguous()
    return edge_index, edge_attr

def rot_z(theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def collect_data_polyhedron(polyhedron_verts):
    dihedral_matrix, edge_length_matrix,face_centers_distance_matrix ,face_sides,face_areas,face_normals = get_edge_face_features(points=polyhedron_verts)
    energy = [coulomb_energy_per_n_verts(points=face_normals)]
    three_energy = three_body_energy(points=face_normals, dihedral_matrix=dihedral_matrix,sep_const=1,theta=1)

    data = {
            # 'coords': verts_list,
            'linkage_distance': moment_of_inertia(points = face_normals),
            'columb_energy': energy,
            'three_body_energy':three_energy,

            'dihedral_matrix':dihedral_matrix.tolist(),
            'edge_length_matrix':edge_length_matrix.tolist(),
            'face_centers_distance_matrix':face_centers_distance_matrix.tolist(),

            'face_sides':face_sides,
            'face_areas':face_areas,
            'face_normals':face_normals.tolist()
            }
    return data

def collect_data(polyhedra_verts,combination_indices, save_dir='',polyhedra_dir=''):

    for combination in combination_indices:
        index_a = combination[0]
        index_b = combination[1]

        verts_a = polyhedra_verts[index_a].tolist()
        verts_b = polyhedra_verts[index_b].tolist()

        similarity = similarity_measure_random(verts_a=verts_a,
                                            verts_b=verts_b, 
                                            n_cores=6, 
                                            n_points= 10000, 
                                            sigma = 2)

        data_a = collect_data_polyhedron(polyhedron_verts=verts_a)
        data_b = collect_data_polyhedron(polyhedron_verts=verts_b)



        data = {
            'similarity':similarity,
            'polyhedron_a':data_a,
            'polyhedron_b':data_b,
        }

        if not os.path.exists(f'{polyhedra_dir}{os.sep}{index_a}.json'):
            with open(f'{polyhedra_dir}{os.sep}{index_a}.json','w') as outfile:
                json.dump(data_a, outfile,indent=4)
        if not os.path.exists(f'{polyhedra_dir}{os.sep}{index_b}.json'):
            with open(f'{polyhedra_dir}{os.sep}{index_b}.json','w') as outfile:
                json.dump(data_b, outfile,indent=4)

        with open(f'{save_dir}{os.sep}{index_a}-{index_b}.json','w') as outfile:
            json.dump(data, outfile,indent=4)
        
    return None

def main():

    parent_dir = os.path.dirname(__file__)
    train_dir = f"{parent_dir}{os.sep}train"
    test_dir = f"{parent_dir}{os.sep}test"
    val_dir = f"{parent_dir}{os.sep}val"
    polyhedra_dir = f"{parent_dir}{os.sep}polyhedra"
    print(train_dir)
    print(test_dir)
    print(val_dir)
    print(polyhedra_dir)
    # Creating train,text, val directory
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    if os.path.exists(polyhedra_dir):
        shutil.rmtree(polyhedra_dir)
    os.makedirs(polyhedra_dir)

    verts_tetra = pv.Tetrahedron().points
    verts_cube = pv.Cube().points
    verts_oct = pv.Octahedron().points
    verts_dod = pv.Dodecahedron().points
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1


    polyhedra_verts = [verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot]
    data_indices = np.arange(0,len(polyhedra_verts))
    train_indices,val_indices = train_test_split(data_indices, test_size = 0.25, random_state=0)

    train_combinations = list(itertools.combinations(train_indices,2))
    val_combinations = list(itertools.combinations(val_indices,2))

    
    collect_data(polyhedra_verts=polyhedra_verts,combination_indices=train_combinations, save_dir=train_dir,polyhedra_dir=polyhedra_dir)
    collect_data(polyhedra_verts=polyhedra_verts,combination_indices=val_combinations, save_dir=val_dir,polyhedra_dir=polyhedra_dir)


if __name__ == '__main__':
    main()

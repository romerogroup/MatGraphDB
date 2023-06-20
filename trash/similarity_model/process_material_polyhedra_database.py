import os
import shutil
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


def moment_of_inertia(points):
    """
    Calculate the moment of inertia for a set of points.

    Args:
    points (list of lists or numpy array): The set of points.

    Returns:
    float: The moment of inertia for the set of points.
    """
    points = np.array(points)
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
            energy += charges[i] * charges[j] * distance
    return float(energy) / num_particles

def connected_energy_per_n_verts(points, adjacency_matrix):
    energy = 0
    charges = [1]*adjacency_matrix.shape[0]
    num_particles = adjacency_matrix.shape[0]

    points = (points - points.mean(axis=0))/points.std(axis=0)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if adjacency_matrix[i,j] > 0:
                distance = np.linalg.norm(points[i] - points[j])
                energy += charges[i] * charges[j] * distance
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
    print(energy)
    return energy
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

def collect_data(polyhedra_verts,indices, save_dir='',):
    for index in indices:
        verts_list = polyhedra_verts[index].tolist()

        dihedral_matrix, edge_length_matrix, face_centers_distance_matrix,face_sides,face_areas,face_normals = get_edge_face_features(points=polyhedra_verts[index])
        energy = [coulomb_energy_per_n_verts(points=face_normals)]
        three_energy = three_body_energy(points=face_normals, dihedral_matrix=dihedral_matrix,sep_const=1,theta=1)
        node_feature = generate_pyg_node_feature(face_areas, face_sides)
        edge_index, edge_attr = generate_pyg_edge_feature(dihedral_matrix, edge_length_matrix)
        
        data = {
                'coords': verts_list,
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

        with open(f'{save_dir}{os.sep}{index}.json','w') as outfile:
            json.dump(data, outfile,indent=4)
        
    return None


def main():
    dataset_dir = f'C:/Users/lllang/Desktop/Romero Group Research/Research Projects/crystal_generation_project/datasets'
    parent_dir = os.path.dirname(__file__)
    train_dir = f"{parent_dir}{os.sep}train"
    test_dir = f"{parent_dir}{os.sep}test"
    val_dir = f"{parent_dir}{os.sep}val"
    print(train_dir)
    print(test_dir)
    print(val_dir)
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


    polyhedra_verts = []
    cif_dir = f"{dataset_dir}{os.sep}mp_cif"

    cif_files = os.listdir(cif_dir)

    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]
        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{cif_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except:
            pass
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


    verts_tetra = pv.Tetrahedron().points
    verts_cube = pv.Cube().points
    verts_oct = pv.Octahedron().points
    verts_dod = pv.Dodecahedron().points
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

    polyhedra_verts.append(verts_tetra)
    polyhedra_verts.append(verts_cube)
    polyhedra_verts.append(verts_oct)
    polyhedra_verts.append(verts_dod)
    polyhedra_verts.append(verts_tetra_rot)


    data_indices = np.arange(0,len(polyhedra_verts))
    train_indices,val_indices = train_test_split(data_indices, test_size = 0.25, random_state=0)
    print(train_indices,val_indices)

    collect_data(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir)
    collect_data(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir)




if __name__ == '__main__':
    main()

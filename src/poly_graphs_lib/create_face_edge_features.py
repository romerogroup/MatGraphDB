import os
import json

from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt

from coxeter.shapes import ConvexPolyhedron

def rot_z(theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def plot_adjacency(plotter, adj_matrix, points):
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))
                
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
    return float(distances_squared.sum())/ n

def energy_per_n_verts(points):
    energy = 0
    charges = [1]*len(points)
    num_particles = len(charges)

    points = (points - points.mean(axis=0))/points.std(axis=0)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
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

                energy += sep_const*0.5*np.linalg.norm(sep_vec)**2 + 0.5 * (dihedral_matrix[i,j,0] - theta**2) **2
    return energy

def find_common_edges(face_1, face_2):
    result = []
    for i in face_1:
        if i in face_2:
            result.append(i)
    return result

def create_graph_node_featutres(node_feature_list):
    x = np.concatenate(node_feature_list,axis=1)
    return x

def create_graph_edge_featutres(edge_feature_list):
    n_faces = edge_feature_list[0].shape[0]
    edge_index = []
    edge_attr = []
    for i_face in range(n_faces):
        for j_face in range(i_face,n_faces):
            if edge_feature_list[0][i_face,j_face] > 0:
                edge_index.append([i_face,j_face])
                # edge_attr.append([i_face,j_face])
                tmp = []
                for edge_feature_matrix in edge_feature_list:
                    tmp.append(edge_feature_matrix[i_face,j_face])
                edge_attr.append(tmp)

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def create_graph_edge_featutres_2(edge_feature_list):

    n_faces = edge_feature_list[0].shape[0]

    edge_index = []
    edge_attr = []
    for i_face in range(n_faces):
        for j_face in range(i_face,n_faces):
            if edge_feature_list[0][i_face,j_face,0] > 0:
                edge_index.append([i_face,j_face])
                tmp = []
                for edge_feature_matrix in edge_feature_list:
                    tmp.extend(edge_feature_matrix[i_face,j_face,:])
                edge_attr.append(tmp)

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr


def gaussian_node_feature_bin_encoder(values, min_val:float=0, max_val:float=40, sigma:float= 2):
    """Creates bins continuous face features by gaussian method

    Parameters
    ----------
    values : : float
        The continuous value to bin
    min_val : float, optional
        The minimum value, by default 0
    max_val : float, optional
        The max value, by default 40
    sigma : float, optional
        The standard dev for the binning, by default 2

    Returns
    -------
    np.ndarray
        The binned feature
    """

    filter = np.arange(min_val, max_val + sigma, step=sigma)
    values = np.array(values)
    encoded_vec = np.exp(-(values - filter)**2 / sigma**2)
    return encoded_vec

def gaussian_edge_feature_bin_encoder(values, min_val:float=0, max_val:float=40, sigma:float= 2):
    """Creates bins continuous edge features by gaussian method

    Parameters
    ----------
    values : : float
        The continuous value to bin
    min_val : float, optional
        The minimum value, by default 0
    max_val : float, optional
        The max value, by default 40
    sigma : float, optional
        The standard dev for the binning, by default 2

    Returns
    -------
    np.ndarray
        The binned feature
    """
    filter = np.arange(min_val, max_val + sigma, step=sigma)
    values = np.array(values)
    encoded_vec = np.exp(-(values - filter)**2 / sigma**2)
    return encoded_vec

def face_sides_bin_encoder(node_values):
    """Ceates bins for the number of sides on a face

    Parameters
    ----------
    node_values : _type_
        The number of nodes(faces)

    Returns
    -------
    np.ndarray
        The encoded vector
    """
    n_nodes = len(node_values)
    encoded_vec = np.zeros(shape = (n_nodes, 8))

    for i_node,node_value in enumerate(node_values):
        if node_value <= 8:
            encoded_vec[i_node,node_value-3]  = 1
        else:
            encoded_vec[i_node,-1]  = 1
    return encoded_vec


##############################################
# Modify this function to change the datastes
###############################################
def get_pyg_graph_components(points, set_index: int = 0):

    """Creates the pygeometric graph components for points that form a convex polyhedron

    Parameters
    ----------
    points : np.ndarray
        The vertces of a convex polyhedron

    set_index : int
        The feature set to generate

    Returns
    -------
    Tuple
        The x, edge_index, edge_attr, pos , y, y_labels
    """

    poly = ConvexPolyhedron(vertices=points)
    
    num_faces = len(poly.faces)
    face_adjacency_matrix = np.zeros((num_faces, num_faces))

    dihedral_matrix = np.zeros((num_faces, num_faces))
    edge_length_matrix = np.zeros((num_faces, num_faces))
    face_centers_distance_matrix = np.zeros((num_faces, num_faces))

    face_sides = []
    face_areas = []
    face_normals = np.zeros((num_faces, 3))
    
    # Looping over faces and then adjacent faces
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
    
    face_sides = np.array(face_sides)[...,np.newaxis]
    face_areas = np.array(face_areas)[...,np.newaxis]
    dihedral_matrix = np.array(dihedral_matrix)[...,np.newaxis]
    face_centers_distance_matrix = np.array(face_centers_distance_matrix)[...,np.newaxis]

    if set_index == 0:
        node_feature_list = [face_areas,face_sides]
        edge_feature_list = [dihedral_matrix,face_centers_distance_matrix]

    elif set_index == 1:
        encoded_area_vec =  gaussian_node_feature_bin_encoder(values = face_areas, min_val=0, max_val=40, sigma=2)
        encoded_sides_vec = face_sides_bin_encoder(node_values = face_sides)
   
        encoded_dihedral_matrix = gaussian_edge_feature_bin_encoder(values = dihedral_matrix, min_val=0, max_val=2*np.pi, sigma= 0.1)
        encoded_face_centers_distance_matrix = gaussian_edge_feature_bin_encoder(values = face_centers_distance_matrix, min_val=0,  max_val=3, sigma= 0.1)


        node_feature_list = [encoded_area_vec,encoded_sides_vec]
        edge_feature_list = [encoded_dihedral_matrix,encoded_face_centers_distance_matrix]

    elif set_index == 2:
        # encoded_area_vec =  gaussian_node_feature_bin_encoder(values = face_areas, min_val=0, max_val=40, sigma=2)
        encoded_sides_vec = face_sides_bin_encoder(node_values = face_sides)
   
        encoded_dihedral_matrix = gaussian_edge_feature_bin_encoder(values = dihedral_matrix, min_val=0, max_val=2*np.pi, sigma= 0.1)
        encoded_face_centers_distance_matrix = gaussian_edge_feature_bin_encoder(values = face_centers_distance_matrix, min_val=0,  max_val=3, sigma= 0.1)


        node_feature_list = [face_areas,encoded_sides_vec]
        edge_feature_list = [dihedral_matrix,face_centers_distance_matrix]

    elif set_index == 3:
        # encoded_area_vec =  gaussian_node_feature_bin_encoder(values = face_areas, min_val=0, max_val=40, sigma=2)
        encoded_sides_vec = face_sides_bin_encoder(node_values = face_sides)
   
        encoded_dihedral_matrix = gaussian_edge_feature_bin_encoder(values = dihedral_matrix, min_val=0, max_val=2*np.pi, sigma= 0.1)
        encoded_face_centers_distance_matrix = gaussian_edge_feature_bin_encoder(values = face_centers_distance_matrix, min_val=0,  max_val=3, sigma= 0.1)


        node_feature_list = [face_areas,encoded_sides_vec]
        edge_feature_list = [dihedral_matrix]


    elif set_index == 4:
        # encoded_area_vec =  gaussian_node_feature_bin_encoder(values = face_areas, min_val=0, max_val=40, sigma=2)
        encoded_sides_vec = face_sides_bin_encoder(node_values = face_sides)
   
        encoded_dihedral_matrix = gaussian_edge_feature_bin_encoder(values = dihedral_matrix, min_val=0, max_val=2*np.pi, sigma= 0.1)
        encoded_face_centers_distance_matrix = gaussian_edge_feature_bin_encoder(values = face_centers_distance_matrix, min_val=0,  max_val=3, sigma= 0.1)


        node_feature_list = [face_areas,encoded_sides_vec]
        edge_feature_list = [encoded_dihedral_matrix, face_centers_distance_matrix]

    elif set_index == 5:
        # encoded_area_vec =  gaussian_node_feature_bin_encoder(values = face_areas, min_val=0, max_val=40, sigma=2)
        encoded_sides_vec = face_sides_bin_encoder(node_values = face_sides)
   
        encoded_dihedral_matrix = gaussian_edge_feature_bin_encoder(values = dihedral_matrix, min_val=0, max_val=2*np.pi, sigma= 0.1)
        encoded_face_centers_distance_matrix = gaussian_edge_feature_bin_encoder(values = face_centers_distance_matrix, min_val=0,  max_val=3, sigma= 0.1)


        node_feature_list = [face_areas,encoded_sides_vec]
        edge_feature_list = [encoded_dihedral_matrix]



    x = create_graph_node_featutres(node_feature_list = node_feature_list)
    # edge_index, edge_attr  = create_graph_edge_featutres(edge_feature_list = edge_feature_list)
    edge_index, edge_attr  = create_graph_edge_featutres_2(edge_feature_list=edge_feature_list)

    pos = face_normals

    # print(edge_attr.shape)

    # Getting the y value of the graph
    dihedral_energy = three_body_energy(points=pos, dihedral_matrix=dihedral_matrix,sep_const=1,theta=1)
    moi = moment_of_inertia(points=pos)
    energy_per_verts = energy_per_n_verts(points)
    y = [dihedral_energy,moi,energy_per_verts]
    y_labels = ['dihedral_energy','moi','energy_per_verts']

    return x, edge_index, edge_attr, pos , y, y_labels

def get_pyg_vertex_edge_components(points, set_index: int = 0):

    """Creates the pygeometric graph components for points that form a convex polyhedron

    Parameters
    ----------
    points : np.ndarray
        The vertces of a convex polyhedron

    set_index : int
        The feature set to generate

    Returns
    -------
    Tuple
        The x, edge_index, edge_attr, pos , y, y_labels
    """

    poly = ConvexPolyhedron(vertices=points)
    
    print(poly.vertices)
    n_nodes = len(poly.vertices)
    node_adjacency_matrix = np.zeros((n_nodes, n_nodes))

    # edge_matrix = np.zeros((num_faces, num_faces))
    # x = create_graph_node_featutres(node_feature_list = node_feature_list)
    # edge_index, edge_attr  = create_graph_edge_featutres(edge_feature_list = edge_feature_list)
    # edge_index, edge_attr  = create_graph_edge_featutres_2(edge_feature_list=edge_feature_list)

    # pos = face_normals

    # # print(edge_attr.shape)

    # # Getting the y value of the graph
    # dihedral_energy = three_body_energy(points=pos, dihedral_matrix=dihedral_matrix,sep_const=1,theta=1)
    # moi = moment_of_inertia(points=pos)
    # energy_per_verts = energy_per_n_verts(points)
    # y = [dihedral_energy,moi,energy_per_verts]
    # y_labels = ['dihedral_energy','moi','energy_per_verts']

    # return x, edge_index, edge_attr, pos , y, y_labels

def collect_data(polyhedra_verts:List[np.ndarray], 
                 indices:List[int], 
                 save_dir:str='',
                 feature_set_index:int=0,
                 graph_type:str='vert_edge'):
    """Convert a list of polyhedra 

    Parameters
    ----------
    polyhedra_verts : _type_
        A list of polyhedra vertices
    indices : List[int]
        A list of indices in
    save_dir : str, optional
        The save directory, by default ''
    feature_set_index : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    for index in indices:
        verts_list = polyhedra_verts[index].tolist()
        if graph_type == 'face_edge':
            x, edge_index, edge_attr, pos, y, y_labels = get_pyg_graph_components(points=polyhedra_verts[index], set_index = feature_set_index)
        elif graph_type =='vert_edge':
            x, edge_index, edge_attr, pos, y, y_labels = get_pyg_vertex_edge_components(points=polyhedra_verts[index], set_index = feature_set_index)
        # print('____________________')
        # print(x.shape,edge_index.shape, edge_attr.shape)
        data = {
                'coords': verts_list,
                'x':x.tolist(),
                'edge_index':edge_index.tolist(),
                'edge_attr':edge_attr.tolist(),
                'pos':pos.tolist(),
                }

        for i,y_label in enumerate(y_labels):
            data[y_label] = y[i]

        with open(f'{save_dir}{os.sep}{index}.json','w') as outfile:
            json.dump(data, outfile,indent=4)
        
    return None

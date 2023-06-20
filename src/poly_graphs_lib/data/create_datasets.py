import os
import shutil
import random
import json
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from coxeter.families import PlatonicFamily
from coxeter.shapes import ConvexPolyhedron
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from voronoi_statistics.voronoi_structure import VoronoiStructure

from ..poly_featurizer import PolyFeaturizer
from .. import encoder, utils
from ..create_face_edge_features import get_pyg_graph_components, rot_z, collect_data


def create_plutonic_dataset(data_dir:str, feature_set_index:int, val_size:float=0.20,graph_type:str='face_edge'):
    """A method to create the plutonic polyhedra dataset

    Parameters
    ----------
    feature_set_index : int
        _description_
    """

    ###########################################################################################################
    # Start of data generation
    ##########################################################################################################
    feature_dir = data_dir + os.sep + 'plutonic_polyhedra' + os.sep + graph_type + os.sep + f'feature_set_{feature_set_index}'
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)

    train_dir = f"{feature_dir}{os.sep}train"
    test_dir = f"{feature_dir}{os.sep}test"
    val_dir = f"{feature_dir}{os.sep}val"

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

    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1


    polyhedra_verts = [verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot]
    data_indices = np.arange(0,len(polyhedra_verts))
    train_indices,val_indices = train_test_split(data_indices, test_size = val_size, random_state=0)

    collect_data(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir,feature_set_index=feature_set_index)
    collect_data(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir,feature_set_index=feature_set_index)

def create_material_random_polyhedra_dataset(data_dir:str, mpcif_data_dir: str, feature_set_index:int,n_polyhedra:int=500, val_size:float=0.20):


    ###########################################################################################################
    # Start of data generation
    ###########################################################################################################

    feature_dir = data_dir + os.sep + 'material_random_polyhedra' + os.sep + f'feature_set_{feature_set_index}'
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)

    train_dir = f"{feature_dir}{os.sep}train"
    test_dir = f"{feature_dir}{os.sep}test"
    val_dir = f"{feature_dir}{os.sep}val"

    # Creating train,text, val directory
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    polyhedra_verts = []

    # Random polyhedra
    for x in range(n_polyhedra):
        n_random_verts = random.randint(4, 50)
        # poly_verts = generate_random_polyhedron(num_points=n_random_verts, bounds=[-1,1])
        poly_verts = generate_random_polyhedron_2(n_points=n_random_verts)

        polyhedra_verts.append(poly_verts)

    # Material polyhedra
    cif_files = os.listdir(mpcif_data_dir)
    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]
        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except:
            continue
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


    
    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

    test_poly = []
    test_poly.extend([verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot])
    test_poly.extend([verts_mp567387_Ti,verts_mp4019_Ti,verts_mp3397_Ti,verts_mp15502_Ba,verts_mp15502_Ti])
    data_indices = np.arange(0,len(polyhedra_verts))
    train_indices,val_indices = train_test_split(data_indices, test_size = val_size, random_state=0)
    test_indices = np.arange(0,len(test_poly))

    collect_data(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir,feature_set_index=feature_set_index)
    collect_data(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir,feature_set_index=feature_set_index)
    collect_data(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir,feature_set_index=feature_set_index)

def create_material_polyhedra_dataset(data_dir:str, mpcif_data_dir: str, feature_set_index:int, val_size:float=0.20):


    ###########################################################################################################
    # Start of data generation
    ###########################################################################################################

    feature_dir = data_dir + os.sep + 'material_polyhedra' + os.sep + f'feature_set_{feature_set_index}'
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)

    train_dir = f"{feature_dir}{os.sep}train"
    test_dir = f"{feature_dir}{os.sep}test"
    val_dir = f"{feature_dir}{os.sep}val"

    # Creating train,text, val directory
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    polyhedra_verts = []

    # Material polyhedra
    cif_files = os.listdir(mpcif_data_dir)
    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]

        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except Exception as e:
            print(e)
            continue
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


    # print(polyhedra_verts)
    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

    test_poly = []
    test_poly.extend([verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot])
    test_poly.extend([verts_mp567387_Ti,verts_mp4019_Ti,verts_mp3397_Ti,verts_mp15502_Ba,verts_mp15502_Ti])
    data_indices = np.arange(0,len(polyhedra_verts))

    train_indices,val_indices = train_test_split(data_indices, test_size = val_size, random_state=0)
    test_indices = np.arange(0,len(test_poly))
    
    collect_data(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir,feature_set_index=feature_set_index)
    collect_data(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir,feature_set_index=feature_set_index)
    collect_data(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir,feature_set_index=feature_set_index)


def create_material_polyhedra_dataset_2(data_dir:str, mpcif_data_dir: str, node_type:str='face', val_size:float=0.20,y_val = 'energy_per_node'):


    ###########################################################################################################
    # Start of data generation
    ###########################################################################################################

    feature_dir = data_dir + os.sep + y_val +os.sep+ 'material_polyhedra' + os.sep + node_type + '_nodes'
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)

    train_dir = f"{feature_dir}{os.sep}train"
    test_dir = f"{feature_dir}{os.sep}test"
    val_dir = f"{feature_dir}{os.sep}val"

    # Creating train,text, val directory
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    polyhedra_verts = []

    # Material polyhedra
    cif_files = os.listdir(mpcif_data_dir)
    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]

        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except Exception as e:
            print(e)
            continue
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


    # print(polyhedra_verts)
    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

    test_poly = []
    test_poly.extend([verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot])
    test_poly.extend([verts_mp567387_Ti,verts_mp4019_Ti,verts_mp3397_Ti,verts_mp15502_Ba,verts_mp15502_Ti])
    data_indices = np.arange(0,len(polyhedra_verts))

    train_indices,val_indices = train_test_split(data_indices, test_size = val_size, random_state=0)
    test_indices = np.arange(0,len(test_poly))
    
    process_polythedra(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir,node_type=node_type,y_val=y_val)
    process_polythedra(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir,node_type=node_type, y_val=y_val)
    process_polythedra(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir ,node_type=node_type,y_val=y_val)

def create_material_polyhedra_dataset_3(data_dir:str, mpcif_data_dir: str, node_type:str='face', val_size:float=0.20, y_val = 'energy_per_node'):


    ###########################################################################################################
    # Start of data generation
    ###########################################################################################################

    feature_dir = data_dir + os.sep + y_val + os.sep + 'material_polyhedra' + os.sep + node_type + '_nodes'
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)

    train_dir = f"{feature_dir}{os.sep}train"
    test_dir = f"{feature_dir}{os.sep}test"


    # Creating train,text, val directory
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    polyhedra_verts = []

    # Material polyhedra
    cif_files = os.listdir(mpcif_data_dir)
    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]

        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except Exception as e:
            print(e)
            continue
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


    # print(polyhedra_verts)
    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices
    verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

    test_poly = []
    test_poly.extend([verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot])
    test_poly.extend([verts_mp567387_Ti,verts_mp4019_Ti,verts_mp3397_Ti,verts_mp15502_Ba,verts_mp15502_Ti])
    data_indices = np.arange(0,len(polyhedra_verts))
    poly_name = ['tetra','cube','oct','dod','rotated_tetra',
                 'dod-like','cube-like','tetra-like','cube-like','oct-like']
    
    test_indices = np.arange(0,len(test_poly))
    print("Processing Train polyhedra")
    process_polythedra(polyhedra_verts=polyhedra_verts,indices=data_indices, save_dir=train_dir,node_type=node_type,y_val=y_val)
    print("Processing Test polyhedra")
    process_polythedra(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir ,node_type=node_type,y_val=y_val,labels=poly_name)


def process_polythedra(polyhedra_verts, indices, save_dir, node_type,y_val,labels=None):
    for index in indices:
        try:
            poly_vert = polyhedra_verts[index]
            if node_type =='face':
                obj = PolyFeaturizer(vertices=poly_vert)

                face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
                face_areas_features = obj.face_areas
                node_features = np.concatenate([face_areas_features,face_sides_features,],axis=1)
                
                dihedral_angles = obj.get_dihedral_angles()
                edge_features = dihedral_angles

                pos = obj.face_normals
                adj_mat = obj.faces_adj_mat

                if y_val == 'three_body_energy':
                    target_variable = obj.get_three_body_energy(pos,adj_mat)
                elif y_val =='energy_per_node':
                    target_variable = obj.get_energy_per_node(pos,adj_mat)

                obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
                if labels is not None:
                    label = labels[index]
                else:
                    label = None
                obj.set_label(label)
                filename = save_dir + os.sep + str(index) + '.json'
                obj.export_pyg_json(filename)

            elif node_type=='vert':
                    obj = PolyFeaturizer(vertices=poly_vert)

                    node_features = None

                    edge_lengths = obj.get_edge_lengths()
                    edge_features = edge_lengths

                    pos = obj.vertices
                    adj_mat = obj.verts_adj_mat

                    if y_val == 'three_body_energy':
                        target_variable = obj.get_three_body_energy(pos,adj_mat)
                    elif y_val =='energy_per_node':
                        target_variable = obj.get_energy_per_node(pos,adj_mat)


                    obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
                    if labels is not None:
                        label = labels[index]
                    else:
                        label = None
                    obj.set_label(label)
                    filename = save_dir + os.sep + str(index) + '.json'
                    obj.export_pyg_json(filename)
        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    # parameters
    feature_set_index = 5
    
    create_plutonic_dataset(feature_set_index=feature_set_index)

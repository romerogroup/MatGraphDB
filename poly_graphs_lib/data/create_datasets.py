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

def generate_random_polyhedron(num_points, bounds):
    # Generate a set of random points within the bounds
    points = np.random.uniform(bounds[0], bounds[1], (num_points, 3))
    # Compute the convex hull of the points
    
    hull = ConvexHull(points)
    # # Return the vertices of the convex hull
    return points[hull.vertices]

verts_mp567387_Ti = np.array([
                        [9.35032981971472,5.347268180285281,3.817163621730258],    
                        [9.565874129342053,6.5013130232365235,4.13423618046107],   
                        [8.806123370657946,6.8889623195389325,4.521885476763478],  
                        [10.563361819538931,6.501313023236524,5.131723870657948],  
                        [8.196284976763476,5.131723870657947,4.13423618046107],    
                        [7.808635680461068,5.891474629342055,4.521885476763478],   
                        [10.563361819538931,5.1317238706579476,6.5013130232365235],
                        [10.880434378269744,5.347268180285281,5.347268180285282],  
                        [10.15058366514502,4.54701433485498,4.547014334854982],    
                        [7.491563121730255,5.675930319714721,5.6759303197147215],
                        [8.806123370657946,4.521885476763477,6.888962319538932],
                        [9.565874129342054,4.134236180461069,6.5013130232365235],
                        [9.350329819714721,3.8171636217302574,5.3472681802852815],
                        [8.196284976763476,4.1342361804610706,5.131723870657947],
                        [7.808635680461068,4.521885476763478,5.891474629342055],
                        [9.02166768028528,7.206034878269745,5.6759303197147215],
                        [10.175712523236523,6.888962319538932,5.891474629342055],
                        [8.22141383485498,6.476184165145021,6.476184165145021],
                        [10.175712523236523,5.891474629342055,6.888962319538932],
                        [9.02166768028528,5.6759303197147215,7.206034878269744],
                        ])

verts_mp4019_Ti = np.array([
                    [-1.5610940596101637,3.007591755096396,0.6540881713940931],
                    [-0.12734937191639875,1.5018256393814862,-0.9681590877898649],
                    [-0.1472139553524771,4.059347306539453,0.9048358963742741],
                    [-1.1056018594606634,2.9118447037721467,-1.2628776782671443],
                    [-1.1347728981648033,2.9395716377943497,-1.2192138009483693],
                    [-1.1007870284255312,2.954760759147764,-1.2642544002698246],
                    [0.14721395535247794,1.4480816934605474,-0.9048358963742734],
                    [-0.04951849136340658,1.3380790861342804,-0.7558287522527248],
                    [1.1007870284255317,2.5526682408522365,1.2642544002698248],
                    [-0.4467834459029587,1.4014632463672834,0.9897970580129993],
                    [0.0495184913634068,4.1693499138657195,0.7558287522527258],
                    [1.1056018594606638,2.5955842962278535,1.2628776782671451],
                    [0.12734937191639872,4.005603360618514,0.9681590877898655],
                    [0.44678344590295893,4.105965753632717,-0.9897970580129986],
                    [1.1347728981648038,2.5678573622056504,1.219213800948371],
                    [1.5610940596101637,2.499837244903604,-0.6540881713940925],
                    ])

verts_mp3397_Ti = np.array([
                    [1.9306777251472496,3.1169782854024306,2.0563769867442585],
                    [1.8054822121351715,3.1240859126747322,1.9467541237138737],
                    [1.945910572533114,3.1566328366269816,2.04097287449931],
                    [2.464382954373896,2.1142163254329995,1.873274249487606],
                    [2.719479455568606,3.3121417289241086,1.5357261882866609],
                    [1.477186370678998,0.4852793063463494,-0.11811265953387831],
                    [2.441662682102672,0.42162704621215696,0.7204451394088339],
                    [2.673238587927734,0.41026612968851905,-0.40558219482153296],
                    [0.044601042266896096,2.699454623535892,0.054029062321303604],
                    [2.470522912474803,3.1838424381692065,-1.6721398974028676],
                    [3.251331161928193,3.3685101200246885,-0.9888726326663095],
                    [-0.07960104245943045,2.4443673007640045,-0.2300528137135605],
                    [0.04482841213036726,1.9658766754887975,-0.4364098368951282],
                    [-0.07903568131810079,2.4491544147108617,-0.633172809124098],
                    [3.1071433997753926,2.576182674590298,-1.5546818660746209],
                    [3.227423271315739,2.5675009030833937,-1.4709558145256372],
                    [3.1629591019092436,2.235525862106149,-1.407475241019251],
                    [0.01747261621426119,2.6504367321900224,-0.7551432628817729],
                    [0.030745792604394778,2.6778868113389396,-0.7459210615581939],
                    [0.15299435037045034,2.7026738080163524,-0.816650776407609],
                    ])

verts_mp15502_Ba = np.array([
                    [3.325675162276587,2.7721813740825554,-0.15032750642414405],
                    [0.2565651338603758,3.214599141464254,0.3838871671925155],
                    [0.3637623983845195,3.412715041834665,0.5442817367498695],
                    [3.4127150418346646,0.5442817367498698,0.3637623983845195],
                    [3.8596968881556304,0.7672924528764009,3.215555261577223],
                    [3.28736598188651,2.7442730601241747,-0.1716872284629094],
                    [3.2145991414642534,0.38388716719251437,0.25656513386037605],
                    [0.4449622889519995,2.083129333776001,0.44496228895199974],
                    [0.2917369334057738,3.106917762562171,0.2917369334057742],
                    [2.7442730601241743,-0.1716872284629094,3.28736598188651],
                    [0.4449622889520002,0.44496228895199974,2.083129333776001],
                    [2.083129333776001,0.44496228895199974,0.44496228895199996],
                    [3.106917762562171,0.291736933405774,0.2917369334057738],
                    [-0.17168722846290962,3.28736598188651,2.7442730601241756],
                    [-0.15032750642414472,3.325675162276587,2.772181374082556],
                    [0.2917369334057742,0.29173693340577356,3.106917762562171],
                    [0.38388716719251526,0.25656513386037605,3.2145991414642543],
                    [0.5993244100341759,3.440771806912349,3.673643525633595],
                    [3.2249138044700123,3.8538504852728357,3.334543547011789],
                    [3.5695246595817163,3.5695246595817163,3.569524659581716],
                    [3.8538504852728357,3.3345435470117883,3.2249138044700123],
                    [3.2155552615772236,3.859696888155631,0.7672924528764014],
                    [3.440771806912349,3.6736435256335946,0.599324410034177],
                    [0.5442817367498696,0.3637623983845193,3.412715041834666],
                    [0.7672924528764007,3.215555261577223,3.8596968881556313],
                    [3.334543547011787,3.2249138044700114,3.853850485272837],
                    [2.772181374082556,-0.1503275064241436,3.325675162276588],
                    [3.6736435256335938,0.5993244100341756,3.44077180691235],
                    ])

verts_mp15502_Ti = np.array([
                    [6.947501927571278,5.506972438490912,-0.32924188281113315],
                    [4.58269653692375,2.826614046054236,-0.3945087196480414],
                    [7.18110515366992,2.8078845592552617,2.0260123040021707],
                    [4.612639695997836,3.830767440744741,0.54245315366992],
                    [6.329802072428725,5.5069724384909104,0.3292418828111341],
                    [6.244143280351961,3.8120379539457647,2.0559554630762555],
                    [6.309410117188869,1.1316795615090895,-0.3088499275712768],
                    [6.9678938828111345,1.1316795615090887,0.3088499275712766],
                    [6.0961988463300845,2.807884559255262,-2.0260123040021703],
                    [8.69460746307626,2.8266140460542384,0.39450871964804124],
                    [7.033160719648042,3.812037953945764,-2.0559554630762564],
                    [8.664664304002175,3.830767440744741,-0.5424531536699175],
                    ])

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


def create_material_polyhedra_dataset_2(data_dir:str, mpcif_data_dir: str, node_type:str='face', val_size:float=0.20):


    ###########################################################################################################
    # Start of data generation
    ###########################################################################################################

    feature_dir = data_dir + os.sep + 'material_polyhedra' + os.sep + node_type + '_nodes'
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
    
    process_polythedra(polyhedra_verts=polyhedra_verts,indices=train_indices, save_dir=train_dir)
    process_polythedra(polyhedra_verts=polyhedra_verts,indices=val_indices, save_dir=val_dir)
    process_polythedra(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir)


def process_polythedra(polyhedra_verts, indices, save_dir, node_type):
    for index in indices:
        poly_vert = polyhedra_verts[index]
        if node_type =='face':
                obj = PolyFeaturizer(vertices=poly_vert)

                face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
                face_areas_features = obj.face_areas
                node_features = np.concatenate([face_areas_features,face_sides_features,],axis=1)
                
                dihedral_angles = obj.get_dihedral_angles()
                edge_features = dihedral_angles

                pos = obj.face_centers
                adj_mat = obj.faces_adj_mat

                target_variable = obj.get_energy_per_node(pos,adj_mat)

                obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)

                filename = save_dir + os.sep + str(index) + '.json'
                obj.export_pyg_json(filename)

        elif node_type=='vert':
                obj = PolyFeaturizer(vertices=poly_vert)

                node_features = None

                edge_lengths = obj.get_edge_lengths()
                edge_features = edge_lengths

                pos = obj.vertices
                adj_mat = obj.verts_adj_mat

                target_variable = obj.get_energy_per_node(pos,adj_mat)

                obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)

                filename = save_dir + os.sep + str(index) + '.json'
                obj.export_pyg_json(filename)

if __name__ == '__main__':
    # parameters
    feature_set_index = 5
    
    create_plutonic_dataset(feature_set_index=feature_set_index)

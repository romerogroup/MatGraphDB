import os
import json
import shutil
from glob import glob
import itertools
import random

import numpy as np
from coxeter.families import PlatonicFamily
from voronoi_statistics.voronoi_structure import VoronoiStructure

from ..utils.shapes import test_polys,test_names
from ..utils import math
from ..data.featurization import PolyFeaturizer

# from ..config import PROJECT_DIR
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(PROJECT_DIR)
class FeatureGeneratorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}data"
    
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"
    processed_dir : str = f"{data_dir}{os.sep}processed"

    dirname : str = "nelement_max_3_nsites_max_10_3d"
    raw_json_dir : str = f"{raw_dir}{os.sep}{dirname}"
    raw_test_dir : str = f"{raw_dir}{os.sep}test"

    interim_json_dir : str = f"{interim_dir}{os.sep}{dirname}"
    interim_test_dir : str = f"{interim_dir}{os.sep}test"


    n_cores : int = 40
    feature_set_index : int = 1

class FeatureGenerator:

    def __init__(self):
        self.config = FeatureGeneratorConfig() 

    def initialize_generation(self):
        print("___Initializing Feature Generation___")

        self._featurize_dir(dir=self.config.raw_json_dir,save_dir=self.config.interim_json_dir)
        self._featurize_dir(dir=self.config.raw_test_dir,save_dir=self.config.interim_test_dir)


    def _featurize_dir(self,dir,save_dir):
        os.makedirs(save_dir,exist_ok=True)

        filenames = dir + '/*.json'
        poly_files = glob(filenames)
        print("There are this many poly files: ", len(poly_files))
        for poly_file in poly_files:
            self._featurize_poly_file(poly_file,save_dir)

    def _featurize_poly_file(self,poly_file,save_dir):
        try:
            # Getting label information
            filename = poly_file.split(os.sep)[-1]
            save_path = save_dir + os.sep + filename

            # # Checking if interim file exists, if it exists update it
            # if os.path.exists(save_path):
            #     with open(poly_file) as f:
            #         poly_dict = json.load(f)
            # else:
            #     with open(poly_file) as f:
            #         poly_dict = json.load(f)
            if not os.path.exists(save_path):
                with open(poly_file) as f:
                    poly_dict = json.load(f)
                # self._feature_set_0(poly_dict, save_path)
                # self._feature_set_1(poly_dict, save_path)
                # self._feature_set_2(poly_dict, save_path)
                self._feature_set_3(poly_dict, save_path)
        except Exception as e:
            print(poly_file)
            print(e)

    def _feature_set_0(self,poly_dict, save_path):

        # Getting label information
        filename = save_path.split(os.sep)[-1]
        label = filename.split('.')[0]
        poly_vert = poly_dict['vertices']

        # Initializing Polyehdron Featureizer
        obj = PolyFeaturizer(vertices=poly_vert, norm = True)

        face_sides_features = math.encoder.face_sides_bin_encoder(obj.face_sides)
        face_areas_features = obj.face_areas
        node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)
        
        dihedral_angles = obj.get_dihedral_angles()
        edge_features = dihedral_angles

        pos = obj.face_centers
        adj_mat = obj.faces_adj_mat
        
        target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
                                                    face_normals=obj.face_normals,
                                                    adj_mat=adj_mat)

        obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
        obj.set_label(label)


        feature_set = obj.export_pyg_json()

        poly_dict.update({'face_feature_set_0' :feature_set })

        with open(save_path,'w') as outfile:
            json.dump(poly_dict, outfile,indent=4)

        return None
    
    def _feature_set_1(self,poly_dict, save_path):

        # Getting label information
        filename = save_path.split(os.sep)[-1]
        label = filename.split('.')[0]
        poly_vert = poly_dict['vertices']

        # Initializing Polyehdron Featureizer
        obj = PolyFeaturizer(vertices=poly_vert, norm = True)
        
        # Creating node features
        face_sides_features = math.face_sides_bin_encoder(obj.face_sides)
        face_areas_features = obj.face_areas
        node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)
        
        # Creating edge features
        dihedral_angles = obj.get_dihedral_angles()
        dihedral_angles_features = math.gaussian_continuous_bin_encoder(values = dihedral_angles, 
                                                                           min_val=np.pi/8, 
                                                                           max_val=np.pi, 
                                                                           sigma= 0.2)
        edge_features = dihedral_angles_features

        pos = obj.face_centers
        adj_mat = obj.faces_adj_mat
        
        target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
                                                    face_normals=obj.face_normals,
                                                    adj_mat=adj_mat)
        # target_variable = obj.get_energy_per_node(pos,adj_mat)

        obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
        obj.set_label(label)


        feature_set = obj.export_pyg_json()

        poly_dict.update({'face_feature_set_1' :feature_set })

        with open(save_path,'w') as outfile:
            json.dump(poly_dict, outfile,indent=4)

        return None

    def _feature_set_2(self,poly_dict, save_path):

        # Getting label information
        filename = save_path.split(os.sep)[-1]
        label = filename.split('.')[0]
        poly_vert = poly_dict['vertices']

        # Initializing Polyehdron Featureizer
        # poly_vert = np.array(poly_vert)
        # poly_vert  = (poly_vert - poly_vert.mean(axis=0))/poly_vert.std(axis=0)
        obj = PolyFeaturizer(vertices=poly_vert, norm = True)
        
        # Creating node features
        face_sides_features = math.face_sides_bin_encoder(obj.face_sides)
        face_areas_features = math.gaussian_continuous_bin_encoder(values = obj.face_areas, 
                                                                           min_val=0, 
                                                                           max_val=20, 
                                                                           sigma= 1)
        node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)
        
        # Creating edge features

        dihedral_angles = obj.get_dihedral_angles()
        dihedral_angles_features = math.gaussian_continuous_bin_encoder(values = dihedral_angles, min_val=np.pi/8, max_val=np.pi, sigma= 0.2)
        edge_features = dihedral_angles_features

        pos = obj.face_centers
        adj_mat = obj.faces_adj_mat
        
        target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
                                                    face_normals=obj.face_normals,
                                                    adj_mat=adj_mat)
        # target_variable = obj.get_energy_per_node(pos,adj_mat)

        obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
        obj.set_label(label)


        feature_set = obj.export_pyg_json()

        poly_dict.update({'face_feature_set_2' :feature_set })

        with open(save_path,'w') as outfile:
            json.dump(poly_dict, outfile,indent=4)

        return None
    
    def _feature_set_3(self,poly_dict, save_path):

        # Getting label information
        filename = save_path.split(os.sep)[-1]
        label = filename.split('.')[0]
        poly_vert = poly_dict['vertices']

        # Initializing Polyehdron Featureizer
        obj = PolyFeaturizer(vertices=poly_vert, norm = True)
        
        # Creating node features
        vert_area_hists = obj.get_verts_areas_encodings(n_bins = 100, min_val=0, max_val=3.0, sigma=0.1)
        
        vert_angle_hists = obj.get_verts_neighbor_angles_encodings(n_bins = 100, min_val=0, max_val=(3/2)*np.pi, sigma=0.1)
        
        node_features = np.concatenate([vert_area_hists,vert_angle_hists],axis=1)
        # Creating edge features
        neighbor_distances = obj.get_verts_neighbor_distance()
        # dihedral_angles_features = encoder.gaussian_continuous_bin_encoder(values = dihedral_angles, min_val=np.pi/8, max_val=np.pi, sigma= 0.2)
        edge_features = neighbor_distances

        pos = obj.vertices
        adj_mat = obj.verts_adj_mat
        
        target_variable = obj.get_energy_per_node(pos,adj_mat)

        obj.get_pyg_verts_input(x = node_features, edge_attr=edge_features,y = target_variable)
        obj.set_label(label)


        feature_set = obj.export_pyg_json()

        poly_dict.update({'face_feature_set_3' :feature_set })

        with open(save_path,'w') as outfile:
            json.dump(poly_dict, outfile,indent=4)

        return None

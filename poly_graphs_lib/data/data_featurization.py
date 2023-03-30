import os
import json
import shutil
from glob import glob
import itertools
import random

import numpy as np
from coxeter.families import PlatonicFamily
from voronoi_statistics.voronoi_structure import VoronoiStructure

from ..utils import test_polys,test_names
from ..poly_featurizer import PolyFeaturizer

from .. import encoder, utils
from ..create_face_edge_features import get_pyg_graph_components, rot_z, collect_data


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class FeatureGeneratorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}datasets"
    
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"
    processed_dir : str = f"{data_dir}{os.sep}processed"

    dirname : str = "nelement_max_2_nsites_max_6_3d"
    raw_json_dir : str = f"{raw_dir}{os.sep}{dirname}"
    raw_test_dir : str = f"{raw_dir}{os.sep}test"

    interim_json_dir : str = f"{interim_dir}{os.sep}{dirname}"
    interim_test_dir : str = f"{interim_dir}{os.sep}test"


    n_cores : int = 20
    n_points : int = 10000

class FeatureGenerator:

    def __init__(self):
        self.config = FeatureGeneratorConfig() 

    def initialize_generation(self):

        self._featurize_dir(dir=self.config.raw_json_dir,save_dir=self.config.interim_json_dir)
        self._featurize_dir(dir=self.config.raw_test_dir,save_dir=self.config.interim_test_dir)


    def _featurize_dir(self,dir,save_dir):
        os.makedirs(save_dir,exist_ok=True)

        filenames = dir + '/*.json'
        poly_files = glob(filenames)

        for poly_file in poly_files:
            self._featurize_poly_file(poly_file,save_dir)

    def _featurize_poly_file(self,poly_file,save_dir):
        try:
            with open(poly_file) as f:
                poly_dict = json.load(f)
                # poly_dict = json.load(f)['vertices']

            filename = poly_file.split(os.sep)[-1]
            label = filename.split('.')[0]
            poly_vert = poly_dict['vertices']

            obj = PolyFeaturizer(vertices=poly_vert)

            face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
            face_areas_features = obj.face_areas
            node_features = np.concatenate([face_areas_features,face_sides_features,],axis=1)
            
            dihedral_angles = obj.get_dihedral_angles()
            edge_features = dihedral_angles

            pos = obj.face_normals
            adj_mat = obj.faces_adj_mat
            
            target_variable = obj.get_three_body_energy(pos,adj_mat)
        
            target_variable = obj.get_energy_per_node(pos,adj_mat)

            obj.get_pyg_faces_input(x = node_features, edge_attr=edge_features,y = target_variable)
            obj.set_label(label)


            feature_set = obj.export_pyg_json()

            poly_dict['face_feature_set'] = feature_set

            save_path = save_dir + os.sep + filename
            with open(save_path,'w') as outfile:
                json.dump(poly_dict, outfile,indent=4)

        except Exception as e:
            print(poly_file)
            print(e)


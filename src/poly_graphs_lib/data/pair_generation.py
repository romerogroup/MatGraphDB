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


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class PairGeneratorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}datasets"
    
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"
    processed_dir : str = f"{data_dir}{os.sep}processed"


    raw_dirname : str = "nelement_max_2_nsites_max_6_3d"
    interim_json_dir : str = f"{interim_dir}{os.sep}{raw_dirname}"
    interim_test_dir : str = f"{interim_dir}{os.sep}test"

    similarity_dirname = 'similarity'
    train_dir : str = f"{processed_dir}{os.sep}{similarity_dirname}{os.sep}train"
    test_dir : str = f"{processed_dir}{os.sep}{similarity_dirname}{os.sep}test"
    
    n_cores : int = 40
    n_points : int = 20000

class PairGenerator:

    def __init__(self):
        self.config = PairGeneratorConfig() 

    def initialize_generation(self,n_pairs=1000):
        # if os.path.exists(self.config.train_dir):
        #     shutil.rmtree(self.config.train_dir)
        # os.makedirs(self.config.train_dir)

        if os.path.exists(self.config.test_dir):
            shutil.rmtree(self.config.test_dir)
        os.makedirs(self.config.test_dir)


        # pairs_list = self._create_pairs(dir = self.config.interim_json_dir)
        # print("Randomly Selecting pairs: ", n_pairs)
        # final_pairs = random.sample(pairs_list, n_pairs)
        # self._featurize_pairs(final_pairs, save_dir=self.config.train_dir)


        pairs_list = self._create_pairs(dir = self.config.interim_test_dir)
        # print(pairs_list)
        self._featurize_pairs(pairs_list, save_dir=self.config.test_dir)

    def _create_pairs(self,dir):
        filenames = dir+ '/*.json'
        poly_files = glob(filenames)
        poly_files_indices = np.arange(len(poly_files))

        pairs_list = list(itertools.combinations(poly_files,2))
        print("Total number of combination pairs : ", len(pairs_list))
        return pairs_list

    def _featurize_pairs(self,pairs_list,save_dir):
        for i,pair in enumerate(pairs_list):

            if i % 20 == 0:
                print(i)

            with open(pair[0]) as f:
                poly_a = json.load(f)

            with open(pair[1]) as f:
                poly_b = json.load(f)

            poly_a_filename = pair[0].split(os.sep)[-1]
            poly_b_filename = pair[1].split(os.sep)[-1]
            verts_a = poly_a['vertices']
            verts_b = poly_b['vertices']

            
            similarity = self._calculate_similarity(verts_a,verts_b)
            # print(similarity)
            poly_a['similarity'] = similarity
            poly_b['similarity'] = similarity

            pair_dir = save_dir + os.sep + f'pair_{i}'
            if os.path.exists(pair_dir):
                shutil.rmtree(pair_dir)
            os.makedirs(pair_dir)
            save_path = pair_dir + os.sep + poly_a_filename
            with open(save_path,'w') as outfile:
                json.dump(poly_a, outfile)

            save_path = pair_dir + os.sep + poly_b_filename
            with open(save_path,'w') as outfile:
                json.dump(poly_b, outfile)

    def _calculate_similarity(self,verts_a,verts_b):
        verts_a = np.array(verts_a)
        verts_b = np.array(verts_b)
        # verts_a = verts_a/verts_a.max()
        # verts_b = verts_b/verts_b.max()
        verts_a = (verts_a - verts_a.mean()) / verts_a.std()
        verts_b = (verts_b - verts_b.mean()) / verts_b.std()
        featurizer = PolyFeaturizer(vertices = verts_a)
        similarity = featurizer.compare_poly(vertices = verts_b, ncores=self.config.n_cores, n_points=self.config.n_points)
        return similarity
        


        
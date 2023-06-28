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

from ..config import PROJECT_DIR

class RegressionGeneratorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}datasets"
    
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"
    processed_dir : str = f"{data_dir}{os.sep}processed"


    raw_dirname : str = "nelement_max_3_nsites_max_10_3d"
    interim_json_dir : str = f"{interim_dir}{os.sep}{raw_dirname}"
    interim_test_dir : str = f"{interim_dir}{os.sep}test"

    dirname = 'three_body_energy'
    train_dir : str = f"{processed_dir}{os.sep}{dirname}{os.sep}train"
    test_dir : str = f"{processed_dir}{os.sep}{dirname}{os.sep}test"
    
    n_cores : int = 40
    n_points : int = 20000

class RegressionGenerator:

    def __init__(self):
        self.config = RegressionGeneratorConfig() 

    def initialize_generation(self):
        if os.path.exists(self.config.train_dir):
            shutil.rmtree(self.config.train_dir)
        os.makedirs(self.config.train_dir)

        if os.path.exists(self.config.test_dir):
            shutil.rmtree(self.config.test_dir)
        os.makedirs(self.config.test_dir)

        filenames = self.config.interim_json_dir + '/*.json'
        poly_files = glob(filenames)
        self._featurize_pairs(poly_files, save_dir=self.config.train_dir)

        filenames = self.config.interim_test_dir+ '/*.json'
        poly_files = glob(filenames)
        self._featurize_pairs(poly_files, save_dir=self.config.test_dir)

    def _featurize_pairs(self,poly_files,save_dir):
        for i,poly_file in enumerate(poly_files):

            if i % 20 == 0:
                print(i)

            with open(poly_file) as f:
                poly = json.load(f)


            poly_filename = poly_file.split(os.sep)[-1]
            
            save_path = save_dir + os.sep + poly_filename
            with open(save_path,'w') as outfile:
                json.dump(poly, outfile)

import os
import shutil
import json
from glob import glob

import numpy as np
from coxeter.families import PlatonicFamily

from matgraphdb.core.voronoi_structure import VoronoiStructure

from ..utils.shapes import test_polys,test_names



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class PolyCollectorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}data"
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"

class PolyCollector:

    def __init__(self,cif_dir,from_scratch=False):
        self.config = PolyCollectorConfig() 


        self.cif_dir=cif_dir
        dir_name=cif_dir.split(os.sep)[-1].split('.')[0]

        self.json_dir : str = f"{self.config.raw_dir}{os.sep}{dir_name}"
        self.test_dir : str = f"{self.config.raw_dir}{os.sep}test"

        if from_scratch:
            if os.path.exists(self.json_dir):
                shutil.rmtree(self.json_dir)
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    def initialize_ingestion(self):
        self._to_raw()
        
    def _to_raw(self):
        if not os.path.exists(self.test_dir):
            print("___processing internal polyhedra___")
            os.makedirs(self.test_dir,exist_ok=True)
            self._internal_to_raw()

        if not os.path.exists(self.json_dir):
            print("___processing external polyhedra___")
            os.makedirs(self.json_dir,exist_ok=True)
            self._external_to_raw()

    def _internal_to_raw(self):
        """Converts interal poly vertices data to raw poly vertices data"""
        for poly_vert,poly_name in zip(test_polys,test_names):
            verts = poly_vert.tolist()

            self._vert_to_json(vertices=verts, poly_name=poly_name,save_dir = self.test_dir)

    def _external_to_raw(self):
        """Converts external data to raw poly vertices data"""

        filenames = self.cif_dir + '/*.cif'

        files = glob(filenames)

        for cif_file in files:
            self.mp_cif_to_json(cif_file)

    def mp_cif_to_json(self,cif_file):
        """Converts a cif file into a VoronoiStructure that multiple Voronoi Polyhedra, 
        then save the Voronoi polyhedra's verts as a json"""

        mp_id = cif_file.split(os.sep)[-1].split('.')[0]

        try:
            voronoi_structure = VoronoiStructure(structure_id = cif_file, 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
            
            voronoi_structure_dict = voronoi_structure.as_dict()

            for i,polyhedra_dict in enumerate(voronoi_structure_dict['voronoi_polyhedra_info']):
                verts=polyhedra_dict['vertices']
                coord_env=polyhedra_dict['coordination_envrionment']
                poly_name=f'{mp_id}_poly_{i}'
                self._vert_to_json( vertices=verts,poly_name=poly_name,save_dir = self.json_dir, coord_env=coord_env)
            
        except Exception as e:
            print(e)
            pass
            # continue

    def _vert_to_json(self, vertices, poly_name, save_dir, coord_env=None):
        poly_data = {'vertices' : vertices, 'coord_env':coord_env}
        filename = f'{save_dir}{os.sep}{poly_name}.json'
        with open(filename,'w') as outfile:
            json.dump(poly_data, outfile, indent=4)

import os
import json
from glob import glob

import numpy as np
from coxeter.families import PlatonicFamily
from voronoi_statistics.voronoi_structure import VoronoiStructure

from ..utils import test_polys,test_names



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class PolyCollectorConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}datasets"
    
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"

    cif_dirname : str = "nelement_max_2_nsites_max_6_3d"

    cif_dir: str = f"{external_dir}{os.sep}{cif_dirname}"
    json_dir : str = f"{raw_dir}{os.sep}{cif_dirname}"
    test_dir : str = f"{raw_dir}{os.sep}test"
        

class PolyCollector:

    def __init__(self):
        self.config = PolyCollectorConfig() 

    def initialize_ingestion(self):
        self._to_raw()
        
    def _to_raw(self):
        if not os.path.exists(self.config.test_dir):
            print("___processing internal polyhedra___")
            os.makedirs(self.config.test_dir,exist_ok=True)
            self._internal_to_raw()

        if not os.path.exists(self.config.json_dir):
            print("___processing external polyhedra___")
            os.makedirs(self.config.json_dir,exist_ok=True)
            self._external_to_raw()

    def _internal_to_raw(self):
        """Converts interal poly vertices data to raw poly vertices data"""
        for poly_vert,poly_name in zip(test_polys,test_names):
            verts = poly_vert.tolist()

            self._vert_to_json(vertices=verts, poly_name=poly_name,save_dir = self.config.test_dir)

    def _external_to_raw(self):
        """Converts external data to raw poly vertices data"""

        filenames = self.config.cif_dir + '/*.cif'

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
                verts =polyhedra_dict['vertices']
                poly_name = f'{mp_id}_poly_{i}'
                self._vert_to_json( vertices=verts, poly_name=poly_name,save_dir = self.config.json_dir)
            
        except Exception as e:
            print(e)
            pass
            # continue

    def _vert_to_json(self, vertices, poly_name,save_dir):
        poly_data = {'vertices' : vertices}
        filename = f'{save_dir}{os.sep}{poly_name}.json'
        with open(filename,'w') as outfile:
            json.dump(poly_data, outfile, indent=4)

if __name__ == "__main__":
    obj = PolyCollector()
    obj.initialize_ingestion()
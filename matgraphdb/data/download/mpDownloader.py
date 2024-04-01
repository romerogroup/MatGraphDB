import os
import json
from typing import List
import multiprocessing as mp
import shutil
from functools import partial

import pyvista as pv
import numpy as np
import pymatgen.core as pmat


import pymatgen.analysis.local_env as pm
from pymatgen.io.cif import CifWriter

from matgraphdb.utils import LOGGER, ROOT

class MPDownloader:

    def __init__(self, apikey:str, from_scratch=True, legacy: bool = False, **kwargs):
        
        self.apikey=apikey
        self.from_scratch=from_scratch
        self.config=kwargs
        

    def initialize_download(self):
        if self.config['legacy']:
            self.legacy_download()
        else:
            self.download()

    def legacy_download(self,dir:str, criteria:dict, properties:List):
        from pymatgen.ext.matproj import MPRester

        mpr = MPRester(self.apikey)

        materials_info = mpr.query(
                criteria=criteria, 
                properties=properties
            )
        n_materials = len(materials_info)
        print("Criteria : ")
        print(json.dumps(criteria, indent=4))
        print('----------------------------------------')
        print("Found {0} materials".format(n_materials))


        for mat_info in materials_info:
            writer = CifWriter(struct = mat_info['structure'])
            cif_file = f"{dir}{os.sep}{mat_info['task_id']}.cif"
            writer.write_file(cif_file)
    
    def download(self, dir:str, criteria:dict,):
        """The method will download cif files for 3d structutre with criteria 
        """
        from mp_api.client import MPRester

        with MPRester(self.apikey) as mpr:

            # Initial screening 
            # summary_docs = mpr.summary._search( nelements= 1, energy_above_hull_min = 0, energy_above_hull_max = 0.02, fields=['material_id'])
            summary_docs = mpr.summary._search( **criteria, fields=['material_id','structure'])
            
            n_materials = len(summary_docs)
            print("Found {0} possible materials".format(n_materials))

            material_ids=[]
            structures=[]
            for doc in summary_docs:
                material_ids.append(str(doc.material_id))
                structures.append(doc.structure)

            # Used to screen 3d dimensional material
            filtered_material_ids=[]
            filtered_structures=[]
            for i,material_id in enumerate(material_ids):
                try:
                    robocrys_docs = mpr.robocrys._search( material_ids = [material_id], fields = ['condensed_structure'])[0]
                    if robocrys_docs.condensed_structure.dimensionality == 3:
                        filtered_material_ids.append(material_id)
                        filtered_structures.append(structures[i])
                except Exception as e:
                    print(e)
                    print(material_id)

                    continue
            n_3d_material = len(filtered_material_ids)

            print("Found {0} possible 3d materials".format(n_3d_material))

            # # Get filtered structures
            # summary_docs = mpr.summary._search( material_ids=filtered_material_ids, fields=['structure'],chunk_size=500)
            
            for structure,material_id in zip(filtered_structures,filtered_material_ids):
                writer = CifWriter(struct = structure)
                cif_file = f"{dir}{os.sep}{material_id}.cif"
                writer.write_file(cif_file)

if __name__=='__main__':

    mp_downloader = MPDownloader(from_scratch=True)
    mp_downloader.initialize_download()
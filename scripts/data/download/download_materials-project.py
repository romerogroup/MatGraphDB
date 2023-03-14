
import os
import json
import multiprocessing as mp
import shutil
from functools import partial

import pyvista as pv
import numpy as np
import pymatgen.core as pmat


import pymatgen.analysis.local_env as pm
from pymatgen.io.cif import CifWriter

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class MPDownloaderConfig:

    legacy_apikey = "A45OsEslmcbF4UiwL"
    apikey = '4MJ9fIoSZVWbcCCVy6FWRK39ychCtL7R'
    cif_download_dir : str = f"{PROJECT_DIR}{os.sep}datasets{os.sep}raw{os.sep}mp_cif"

    legacy_criteria: dict = {
                    "nelements":1 ,
                    "e_above_hull": {"$lt":0.02}
                    }
    legacy_properties: list = ["structure", "efermi","e_above_hull","cif",'task_id']
    
    # These are set based on their endpoints at the followin URL: https://api.materialsproject.org/docs#/Summary/search_summary__get
    criteria : dict = {
        "nelements_max" : 2,
        "nsites_max" : 6,
        "energy_above_hull_min": 0,
        "energy_above_hull_max": 0.02,
    }
    
    
    legacy: bool = False
    def __init__(self,from_scratch=False):
        if from_scratch:
            if os.path.exists(self.cif_download_dir):
                shutil.rmtree(self.cif_download_dir)
            os.makedirs(self.cif_download_dir)

class MPDownloader:

    def __init__(self,from_scratch=True ):

        self.config = MPDownloaderConfig(from_scratch=from_scratch)

    def initialize_download(self):

        # try:
            

            if self.config.legacy:
                self.legacy_download()
            else:
                self.download()
        # except Exception as e:
        #     print('Error: ',e)
    
    def legacy_download(self):
        from pymatgen.ext.matproj import MPRester

        mpr = MPRester(self.config.legacy_apikey)

        materials_info = mpr.query(
                criteria=self.config.legacy_criteria, 
                properties=self.config.legacy_properties
            )
        n_materials = len(materials_info)
        print("Criteria : ")
        print(json.dumps(self.config.legacy_criteria, indent=4))
        print('----------------------------------------')
        print("Found {0} materials".format(n_materials))


        for mat_info in materials_info:
            writer = CifWriter(struct = mat_info['structure'])
            cif_file = f"{self.config.cif_download_dir}{os.sep}{mat_info['task_id']}.cif"
            writer.write_file(cif_file)
    
    def download(self):
        """The method will download cif files for 3d structutre with criteria 
        """
        from mp_api.client import MPRester

        with MPRester(self.config.apikey) as mpr:

            # Initial screening 
            # summary_docs = mpr.summary._search( nelements= 1, energy_above_hull_min = 0, energy_above_hull_max = 0.02, fields=['material_id'])
            summary_docs = mpr.summary._search( **self.config.criteria, fields=['material_id','structure'])
            
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
                cif_file = f"{self.config.cif_download_dir}{os.sep}{material_id}.cif"
                writer.write_file(cif_file)

if __name__=='__main__':

    mp_downloader = MPDownloader(from_scratch=True)
    mp_downloader.initialize_download()
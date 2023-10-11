import os
from glob import glob 

from poly_graphs_lib.utils import PROJECT_DIR


material_file_path = os.path.join(PROJECT_DIR,'data','raw','mp_database','*.json')
material_files =  glob(material_file_path)

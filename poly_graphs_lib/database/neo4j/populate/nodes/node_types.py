import os
from glob import glob

import numpy as np
from pymatgen.core.periodic_table import Element

from poly_graphs_lib.utils.periodic_table import atomic_symbols
from poly_graphs_lib.cfg.coordination_geometries_files import mp_coord_encoding
from poly_graphs_lib.database.json import DB_DIR

ELEMENTS = atomic_symbols[1:]

MAGNETIC_STATES=['NM', 'FM', 'FiM', 'AFM', 'Unknown']

CRYSTAL_SYSTEMS = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']

MATERIAL_FILES =  glob(DB_DIR + os.sep + '*.json')

CHEMENV_NAMES=mp_coord_encoding.keys()

tmp=[]
for element_name in ELEMENTS:
    for chemenv_name in CHEMENV_NAMES:
        class_name= element_name + '_' + chemenv_name
        tmp.append(class_name)

CHEMENV_ELEMENT_NAMES=tmp


SPG_NAMES=[f'spg_{i}' for i in np.arange(1,231)]


wyckoff_letters=['a', 'b', 'c', 'd', 'e', 'f']
spg_wyckoffs=[]
for wyckoff_letter in wyckoff_letters:
    for spg_name in SPG_NAMES:
        spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)
SPG_WYCKOFFS=spg_wyckoffs
    




# Used to get magnetic states
# from poly_graphs_lib.database.json import material_files
# magnetic_states=[]
# for i,mat_file in enumerate(material_files):
#     if i%100==0:
#         print(i)
#     with open(mat_file) as f:
#         db = json.load(f)
#     magnetic_state=db["ordering"]
#     if magnetic_state not in magnetic_states:
#         magnetic_states.append(magnetic_state)
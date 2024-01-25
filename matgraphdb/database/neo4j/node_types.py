import os
from glob import glob

import numpy as np
from pymatgen.core.periodic_table import Element

from matgraphdb.utils.periodic_table import atomic_symbols
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import DB_DIR

ELEMENTS = atomic_symbols[1:]
ELEMENTS_MAP = {element:i for i,element in enumerate(ELEMENTS)}

MAGNETIC_STATES=['NM', 'FM', 'FiM', 'AFM', 'Unknown']

CRYSTAL_SYSTEMS = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']

MATERIAL_FILES =  glob(DB_DIR + os.sep + '*.json')

CHEMENV_NAMES=mp_coord_encoding.keys()
CHEMENV_NAMES_MAP={name:i for i,name in enumerate(CHEMENV_NAMES)}

tmp=[]
for element_name in ELEMENTS:
    for chemenv_name in CHEMENV_NAMES:
        class_name= element_name + '_' + chemenv_name
        tmp.append(class_name)

CHEMENV_ELEMENT_NAMES=tmp
CHEMENV_ELEMENT_NAMES_MAP={name:i for i,name in enumerate(CHEMENV_ELEMENT_NAMES)}

SPG_NAMES=[f'spg_{i}' for i in np.arange(1,231)]


# wyckoff_letters=['a', 'b', 'c', 'd', 'e', 'f']
# spg_wyckoffs=[]
# for wyckoff_letter in wyckoff_letters:
#     for spg_name in SPG_NAMES:
#         spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)
# SPG_WYCKOFFS=spg_wyckoffs
    
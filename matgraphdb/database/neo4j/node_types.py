import os
from glob import glob

import numpy as np
from pymatgen.core.periodic_table import Element

from matgraphdb.utils.periodic_table import atomic_symbols
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import DB_DIR

MATERIAL_FILES =  glob(DB_DIR + os.sep + '*.json')


ELEMENTS = atomic_symbols[1:]
ELEMENTS_ID_MAP = {element:i for i,element in enumerate(ELEMENTS)}

MAGNETIC_STATES=['NM', 'FM', 'FiM', 'AFM', 'Unknown']
MAGNETIC_STATES_ID_MAP={name:i for i,name in enumerate(MAGNETIC_STATES)}

CRYSTAL_SYSTEMS = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
CRYSTAL_SYSTEMS_ID_MAP={name:i for i,name in enumerate(CRYSTAL_SYSTEMS)}

CHEMENV_NAMES=mp_coord_encoding.keys()
CHEMENV_NAMES_ID_MAP={name:i for i,name in enumerate(CHEMENV_NAMES)}

tmp=[]
for element_name in ELEMENTS:
    for chemenv_name in CHEMENV_NAMES:
        class_name= element_name + '_' + chemenv_name
        tmp.append(class_name)

CHEMENV_ELEMENT_NAMES=tmp
CHEMENV_ELEMENT_NAMES_ID_MAP={name:i for i,name in enumerate(CHEMENV_ELEMENT_NAMES)}

SPG_NAMES=[f'spg_{i}' for i in np.arange(1,231)]
SPG_MAP={name:i for i,name in enumerate(SPG_NAMES)}


OXIDATION_STATES = np.arange(-9,10)
OXIDATION_STATES_ID_MAP={name:i for i,name in enumerate(OXIDATION_STATES)}

# wyckoff_letters=['a', 'b', 'c', 'd', 'e', 'f']
# spg_wyckoffs=[]
# for wyckoff_letter in wyckoff_letters:
#     for spg_name in SPG_NAMES:
#         spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)
# SPG_WYCKOFFS=spg_wyckoffs
    
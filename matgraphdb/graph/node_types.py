import os
from glob import glob
import json

import numpy as np
from pymatgen.core.periodic_table import Element
import pymatgen.core as pmat
import pandas as pd

from matgraphdb.utils.periodic_table import atomic_symbols
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import DB_DIR
from matgraphdb.utils import LOGGER, ENCODING_DIR


PROPERTIES = [
    ("material_id", "string"),
    ("nsites", "int"),
    ("elements", "string[]"),
    ("nelements", "int"),
    ("composition", "string"),
    ("composition_reduced", "string"),
    ("formula_pretty", "string"),
    ("volume", "float"),
    ("density", "float"),
    ("density_atomic", "float"),
    ("symmetry", "string"),
    ("energy_per_atom", "float"),
    ("formation_energy_per_atom", "float"),
    ("energy_above_hull", "float"),
    ("is_stable", "boolean"),
    ("band_gap", "float"),
    ("cbm", "float"),
    ("vbm", "float"),
    ("efermi", "string"),
    ("is_gap_direct", "boolean"),
    ("is_metal", "boolean"),
    ("is_magnetic", "boolean"),
    ("ordering", "string"),
    ("total_magnetization", "float"),
    ("total_magnetization_normalized_vol", "float"),
    ("num_magnetic_sites", "int"),
    ("num_unique_magnetic_sites", "int"),
    ("k_voigt", "float"),
    ("k_reuss", "float"),
    ("k_vrh", "float"),
    ("g_voigt", "float"),
    ("g_reuss", "float"),
    ("g_vrh", "float"),
    ("universal_anisotropy", "float"),
    ("homogeneous_poisson", "float"),
    ("e_total", "float"),
    ("e_ionic", "float"),
    ("e_electronic", "float"),
    ("wyckoffs", "string[]"),
]


MATERIAL_FILES = glob(DB_DIR + os.sep + '*.json')

MATERIAL_PROPERTIES = []
MATERIAL_IDS = []
LATTICE_IDS = []
LATTICE_PROPERTIES = []
SITES_IDS = []
SITES_PROPERTIES = []
SITE_IDS = []
SITE_PROPERTIES = []
for i, material_file in enumerate(MATERIAL_FILES[:]):

    with open(material_file) as f:
        db = json.load(f)
        structure = pmat.Structure.from_dict(db['structure'])

    mpid_name = material_file.split(os.sep)[-1].split('.')[0]
    mpid_name = mpid_name.replace('-', '_')

    MATERIAL_IDS.append(mpid_name)
    LATTICE_IDS.append(mpid_name)
    SITES_IDS.append(mpid_name)

    lattice_properties_dict = {'a:float': structure.lattice.a,
                               'b:float': structure.lattice.b,
                               'c:float': structure.lattice.c,
                               'alpha:float': structure.lattice.alpha,
                               'beta:float': structure.lattice.beta,
                               'gamma:float': structure.lattice.gamma}
    LATTICE_PROPERTIES.append(lattice_properties_dict)

    for j, site in enumerate(structure.sites):
        site_properties_dict = {'coordinate:float[]': site.coords.tolist(),
                                'species:string': site.specie.name}
        SITE_IDS.append(mpid_name + '_' + str(j))
        SITE_PROPERTIES.append(site_properties_dict)

    material_property_dict = {}
    for property in PROPERTIES:

        if property[0] == 'symmetry':
            try:
                symmetry_dict = db[property[0]]
                for sym_property_name, sym_property_value in symmetry_dict.items():

                    if sym_property_name == 'crystal_system':
                        property_type = 'string'
                        property_name = 'crystal_system'
                        property_value = sym_property_value.lower()
                    elif sym_property_name == 'number':
                        property_type = 'int'
                        property_name = 'space_group'
                        property_value = sym_property_value
                    elif sym_property_name == 'point_group':
                        property_type = 'string'
                        property_name = 'point_group'
                        property_value = sym_property_value
                    elif sym_property_name == 'symbol':
                        property_type = 'string'
                        property_name = 'hall_symbol'
                        property_value = sym_property_value
                    else:
                        property_name = None
                        property_type = None
                        property_value = None

                    if property_name is not None:
                        node_key = property_name + ':' + property_type
                        material_property_dict.update(
                            {node_key: property_value})
            except Exception as e:
                material_property_dict.update({'crystal_system:string': None,
                                               'space_group:int': None,
                                               'point_group:string': None,
                                               'hall_symbol:string': None})

        else:
            property_name = property[0]
            property_type = property[1]
            node_key = property_name + ':' + property_type
            property_value = db[property_name]

            material_property_dict.update({node_key: property_value})

    # # Check if encodings are present
    # if os.path.exists(ENCODING_DIR):
    #     encoding_files=glob(os.path.join(ENCODING_DIR,'*.csv'))
    #     for encoding_file in encoding_files:
    #         encoding_name=encoding_file.split(os.sep)[-1].split('.')[0]

    #         df=pd.read_csv(encoding_file,index_col=0)

    #         # Convert the dataframe values to a list of strings where the strings are the rows of the dataframe separated by a semicolon
    #         df = df.apply(lambda x: ';'.join(map(str, x)), axis=1)
    #         print(df.head())
    #         # Where the encoding contains nan value replace with None:
    #         df[f'{encoding_name}:float[]']=df[f'{encoding_name}:float[]'].apply(lambda x: None if 'nan' in x else x)

    #         # Remove rows with that contain None values
    #         df=df.dropna(subset=[f'{encoding_name}:float[]'])

    #         material_property_dict.update({f'{encoding_name}:float[]': df.tolist()})
    #     del df

    MATERIAL_PROPERTIES.append(material_property_dict)


ELEMENTS = atomic_symbols[1:]
ELEMENTS_ID_MAP = {element: i for i, element in enumerate(ELEMENTS)}
ELEMENT_PROPERTIES = []
for i, element in enumerate(ELEMENTS[:]):
    # pymatgen object. Given element string, will have useful properties
    pmat_element = Element(element)

    # Handling None and nan value cases
    if str(pmat_element.Z) != 'nan':
        atomic_number = pmat_element.Z
    else:
        atomic_number = None
    if str(pmat_element.X) != 'nan':
        x = pmat_element.X
    else:
        x = None
    if str(pmat_element.atomic_radius) != 'nan' and str(pmat_element.atomic_radius) != 'None':
        atomic_radius = float(pmat_element.atomic_radius)
    else:
        atomic_radius = None
    if str(pmat_element.group) != 'nan':
        group = pmat_element.group
    else:
        group = None
    if str(pmat_element.row) != 'nan':
        row = pmat_element.row
    else:
        row = None
    if str(pmat_element.atomic_mass) != 'nan':
        atomic_mass = float(pmat_element.atomic_mass)
    else:
        atomic_mass = None

    ELEMENT_PROPERTIES.append({"element_name:string": element,
                               "atomic_number:float": atomic_number,
                               "X:float": x,
                               "atomic_radius:float": atomic_radius,
                               "group:int": group,
                               "row:int": row,
                               "atomic_mass:float": atomic_mass})

MAGNETIC_STATES = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
MAGNETIC_STATES_ID_MAP = {name: i for i, name in enumerate(MAGNETIC_STATES)}

CRYSTAL_SYSTEMS = ['triclinic', 'monoclinic', 'orthorhombic',
                   'tetragonal', 'trigonal', 'hexagonal', 'cubic']
CRYSTAL_SYSTEMS_ID_MAP = {name: i for i, name in enumerate(CRYSTAL_SYSTEMS)}

CHEMENV_NAMES = mp_coord_encoding.keys()
CHEMENV_NAMES_ID_MAP = {name: i for i, name in enumerate(CHEMENV_NAMES)}

tmp = []
for element_name in ELEMENTS:
    for chemenv_name in CHEMENV_NAMES:
        class_name = element_name + '_' + chemenv_name
        tmp.append(class_name)

CHEMENV_ELEMENT_NAMES = tmp
CHEMENV_ELEMENT_NAMES_ID_MAP = {name: i for i,
                                name in enumerate(CHEMENV_ELEMENT_NAMES)}

SPG_NAMES = [f'spg_{i}' for i in np.arange(1, 231)]
SPG_MAP = {name: i for i, name in enumerate(SPG_NAMES)}


OXIDATION_STATES = np.arange(-9, 10)
OXIDATION_STATES_NAMES = [f'ox_{i}' for i in OXIDATION_STATES]
OXIDATION_STATES_ID_MAP = {name: i for i, name in enumerate(OXIDATION_STATES)}

wyckoff_letters = ['a', 'b', 'c', 'd', 'e', 'f']
spg_wyckoffs = []
for wyckoff_letter in wyckoff_letters:
    for spg_name in SPG_NAMES:
        spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)
SPG_WYCKOFFS = spg_wyckoffs

import numpy as np
from pymatgen.core.periodic_table import Element

import pandas as pd

from pymatgen.core.units import FloatWithUnit
from matgraphdb.utils.periodic_table import atomic_symbols, pymatgen_properties
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import MATERIAL_PARQUET_FILE



class NodeTypes:
    def __init__(self):
        pass

    def get_element_nodes(self):
        elements = atomic_symbols[1:]
        elements_properties = []
        
        for i, element in enumerate(elements[:]):
            tmp_dict=pymatgen_properties.copy()
            pmat_element=Element(element)
            for key in tmp_dict.keys():
                try:
                    value=getattr(pmat_element,key)
                except:
                    value=None
                if isinstance(value,FloatWithUnit):
                    value=value.real
                if isinstance(value,dict):
                    value=[(key2,value2) for key2,value2 in value.items()]


                tmp_dict[key]=value
            elements_properties.append(tmp_dict)
            
        df = pd.DataFrame(elements_properties)
        df['name'] = df['symbol']
        return df
    
    def get_crystal_system_nodes(self):
        crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic',
                           'tetragonal', 'trigonal', 'hexagonal', 'cubic']
        crystal_systems_properties = []
        for i, crystal_system in enumerate(crystal_systems[:]):
            crystal_systems_properties.append({"crystal_system": crystal_system})

        df = pd.DataFrame(crystal_systems_properties)
        df['name'] = df['crystal_system']
        return df

    def get_magnetic_states_nodes(self):
        magnetic_states = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
        magnetic_states_properties = []
        for i, magnetic_state in enumerate(magnetic_states[:]):
            magnetic_states_properties.append({"magnetic_state": magnetic_state})

        df = pd.DataFrame(magnetic_states_properties)
        df['name'] = df['magnetic_state']
        return df
    
    def get_oxidation_states_nodes(self):
        oxidation_states = np.arange(-9, 10)
        oxidation_states_names = [f'ox_{i}' for i in oxidation_states]
        oxidation_states_properties = []
        for i, oxidation_state in enumerate(oxidation_states_names):
            oxidation_number = oxidation_state.split('_')[1]
            oxidation_states_properties.append({"oxidation_state": oxidation_state})

        df = pd.DataFrame(oxidation_states_properties)
        df['name'] = df['oxidation_state']
        return df
    
    def get_space_group_nodes(self):
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        space_groups_properties = []
        for i, space_group in enumerate(space_groups[:]):
            spg_num=space_group.split('_')[1]
            space_groups_properties.append({"spg": spg_num})

        df = pd.DataFrame(space_groups_properties)
        df['name'] = df['spg']
        return df
    
    def get_chemenv_nodes(self):
        chemenv_names = list(mp_coord_encoding.keys())
        chemenv_names_properties = []
        for i, chemenv_name in enumerate(chemenv_names):
            coordination = int(chemenv_name.split(':')[1])
            chemenv_names_properties.append({"chemenv_name": chemenv_name, 
                                             "coordination": coordination})
            
        df = pd.DataFrame(chemenv_names_properties)
        df['name'] = df['chemenv']
        return df
    
    def get_chemenv_element_nodes(self):
        chemenv_names = list(mp_coord_encoding.keys())
        elements = atomic_symbols[1:]
        chemenv_element_names = []
        for element_name in elements:
            for chemenv_name in chemenv_names:
                
                class_name = element_name + '_' + chemenv_name
                chemenv_element_names.append(class_name)

        chemenv_element_names_properties = []
        for i, chemenv_element_name in enumerate(chemenv_element_names):
            element_name=chemenv_element_name.split('_')[0]


            tmp_dict=pymatgen_properties.copy()
            pmat_element=Element(element_name)
            for key in tmp_dict.keys():
                try:
                    value=getattr(pmat_element,key)
                except:
                    value=None
                if isinstance(value,FloatWithUnit):
                    value=value.real
                    
                tmp_dict[key]=value

            coordination = int(chemenv_element_name.split(':')[1])
            tmp_dict['chemenv_element_name'] = chemenv_element_name
            tmp_dict['coordination'] = coordination
            chemenv_element_names_properties.append(tmp_dict)


        df = pd.DataFrame(chemenv_element_names_properties)
        df['name'] = df['chemenv_element_name']
        return df
    
    def get_wyckoff_positions_nodes(self):
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        wyckoff_letters = ['a', 'b', 'c', 'd', 'e', 'f']
        spg_wyckoffs = []
        for wyckoff_letter in wyckoff_letters:
            for spg_name in space_groups:
                spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)

        spg_wyckoff_properties = []
        for i, spg_wyckoff in enumerate(spg_wyckoffs):
            spg_wyckoff_properties.append({"spg_wyckoff": spg_wyckoff})

        df = pd.DataFrame(spg_wyckoff_properties)
        df['name'] = df['spg_wyckoff']
        return df
    
    def get_material_nodes(self):
        df = pd.read_parquet(MATERIAL_PARQUET_FILE)
        df['name'] = df['material_id']
        return df




if __name__=='__main__':
    nodes=NodeTypes()

    df=nodes.get_element_nodes()

    print(df.head())
    # print(df['name'])



    
    # names,properties,id_map=nodes.get_chemenv_element_nodes()
    # print(properties[:10])


    # names,properties,id_map=nodes.get_chemenv_nodes()
    # print(properties[:10])

    # names,properties,id_map=nodes.get_material_nodes()
    # print(properties[:10])
    # print(names[:10])

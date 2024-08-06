import os
import shutil
import copy
import warnings
from glob import glob
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pymatgen.core.periodic_table import Element
from pymatgen.core.units import FloatWithUnit

from matgraphdb.utils.periodic_table import atomic_symbols, pymatgen_properties
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import MATERIAL_PARQUET_FILE
from matgraphdb.utils import  GRAPH_DIR, LOGGER

class NodeTypes(Enum):
    ELEMENT='ELEMENT'
    CHEMENV='CHEMENV'
    CRYSTAL_SYSTEM='CRYSTAL_SYSTEM'
    MAGNETIC_STATE='MAGNETIC_STATE'
    SPACE_GROUP='SPACE_GROUP'
    OXIDATION_STATE='OXIDATION_STATE'
    MATERIAL='MATERIAL'
    SPG_WYCKOFF='SPG_WYCKOFF'
    CHEMENV_ELEMENT='CHEMENV_ELEMENT'
    LATTICE='LATTICE'
    SITE='SITE'

class Nodes:

    def __init__(self, 
                node_dir, 
                file_type='parquet', 
                output_format='pandas',
                skip_init=False,
                from_scratch=False):
        if file_type not in ['parquet', 'csv']:
            raise ValueError("file_type must be either 'parquet' or 'csv'")
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.file_type=file_type
        self.output_format=output_format
        self.node_dir=node_dir

        if from_scratch:
            shutil.rmtree(self.node_dir)

        os.makedirs(self.node_dir,exist_ok=True)

        if not skip_init:
            self.initialize_nodes()

    def get_element_nodes(self, columns=None):
        warnings.filterwarnings("ignore", category=UserWarning)
        node_type=NodeTypes.ELEMENT.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')

        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

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
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_crystal_system_nodes(self, columns=None):
        node_type=NodeTypes.CRYSTAL_SYSTEM.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df
            
        crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic',
                           'tetragonal', 'trigonal', 'hexagonal', 'cubic']
        crystal_systems_properties = []
        for i, crystal_system in enumerate(crystal_systems[:]):
            crystal_systems_properties.append({"crystal_system": crystal_system})

        df = pd.DataFrame(crystal_systems_properties)
        df['name'] = df['crystal_system']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df

    def get_magnetic_states_nodes(self, columns=None):
        node_type=NodeTypes.MAGNETIC_STATE.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df
            
        magnetic_states = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
        magnetic_states_properties = []
        for i, magnetic_state in enumerate(magnetic_states[:]):
            magnetic_states_properties.append({"magnetic_state": magnetic_state})

        df = pd.DataFrame(magnetic_states_properties)
        df['name'] = df['magnetic_state']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_oxidation_states_nodes(self, columns=None):
        node_type=NodeTypes.OXIDATION_STATE.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

        # Old method
        # oxidation_states = np.arange(-9, 10)
        # oxidation_states_names = [f'ox_{i}' for i in oxidation_states]
        # oxidation_states_properties = []
        # for i, oxidation_state in enumerate(oxidation_states_names):
        #     oxidation_number = oxidation_state.split('_')[1]
        #     oxidation_states_properties.append({"oxidation_state": oxidation_state})
        possible_oxidation_state_names = []
        possible_oxidation_state_valences = []
        material_df = self.get_material_nodes(columns=['oxidation_states-possible_valences'])
        for irow, row in material_df.iterrows():
            possible_valences=row['oxidation_states-possible_valences']
            if possible_valences is None:
                continue
            for possible_valence in possible_valences:
                oxidation_state_name=f'ox_{possible_valence}'
                if oxidation_state_name not in possible_oxidation_state_names:
                    possible_oxidation_state_names.append(oxidation_state_name)
                    possible_oxidation_state_valences.append(possible_valence)

        data={
            'oxidation_state': possible_oxidation_state_names,
            'valence': possible_oxidation_state_valences
        }
        df = pd.DataFrame(data)
        df['name'] = df['oxidation_state']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_space_group_nodes(self, columns=None):
        node_type=NodeTypes.SPACE_GROUP.value


        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df
        
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        space_groups_properties = []
        for i, space_group in enumerate(space_groups[:]):
            spg_num=space_group.split('_')[1]
            space_groups_properties.append({"spg": spg_num})

        df = pd.DataFrame(space_groups_properties)
        df['name'] = df['spg']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_chemenv_nodes(self, columns=None):
        node_type=NodeTypes.CHEMENV.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

        chemenv_names = list(mp_coord_encoding.keys())
        chemenv_names_properties = []
        for i, chemenv_name in enumerate(chemenv_names):
            coordination = int(chemenv_name.split(':')[1])
            chemenv_names_properties.append({"chemenv_name": chemenv_name, 
                                             "coordination": coordination})
            
        df = pd.DataFrame(chemenv_names_properties)
        df['name'] = df['chemenv_name'].str.replace(':','_')
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_chemenv_element_nodes(self, columns=None):
        warnings.filterwarnings("ignore", category=UserWarning)
        node_type=NodeTypes.CHEMENV_ELEMENT.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df
        
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
        df['name'] = df['chemenv_element_name'].str.replace(':','_')
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_wyckoff_positions_nodes(self, columns=None):
        node_type=NodeTypes.SPG_WYCKOFF.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df
        
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
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_material_nodes(self, columns=None):
        node_type=NodeTypes.MATERIAL.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

        df = pd.read_parquet(MATERIAL_PARQUET_FILE)
        df['name'] = df['material_id']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_material_lattice_nodes(self, columns=None):
        node_type=NodeTypes.LATTICE.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

        df = self.get_material_nodes(columns=['material_id', 'lattice', 'a', 'b', 'c', 
                                                                 'alpha', 'beta', 'gamma', 
                                                                 'symmetry-crystal_system','volume'])

        df['name'] = df['material_id']
        df['type'] = node_type
        if columns:
            df = df[columns]
        self.save_nodes(df, filepath)
        return df
    
    def get_material_site_nodes(self, columns=None):
        node_type=NodeTypes.SITE.value

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath):
            df=self.load_nodes(filepath=filepath, columns=columns)
            return df

        df = self.get_material_nodes(columns=['material_id', 'lattice', 'frac_coords', 'species'])
        all_species=[]
        all_coords=[]
        all_lattices=[]
        all_ids=[]
        
        for irow, row in df.iterrows():
            if irow%10000==0:
                print(f"Processing row {irow}")
            if row['species'] is None:
                continue
            for frac_coord, specie in zip(row['frac_coords'], row['species']):
                all_species.append(specie)
                all_coords.append(frac_coord)
                all_lattices.append(row['lattice'])
                all_ids.append(row['material_id'])

        df = pd.DataFrame({'species': all_species, 
                           'frac_coords': all_coords, 
                           'lattice': all_lattices, 
                           'material_id': all_ids})
        
        df['name'] = df['material_id']
        df['type'] = node_type
        if columns:
            df = df[columns]

        self.save_nodes(df, filepath)
        return df

    def initialize_nodes(self):
        self.get_element_nodes()
        self.get_chemenv_nodes()
        self.get_crystal_system_nodes()
        self.get_magnetic_states_nodes()
        self.get_space_group_nodes()
        self.get_oxidation_states_nodes()
        self.get_material_nodes()
        self.get_wyckoff_positions_nodes()

    def get_property_names(self, node_type):
        node_types = [type.value for type in NodeTypes]
        if node_type not in node_types:
            raise ValueError(f"Node type must be one of the following: {node_types}")
        
        if node_type == NodeTypes.MATERIAL.value:
            df = self.get_material_property_names()
        elif node_type == NodeTypes.ELEMENT.value:
            df = self.get_element_property_names()
        elif node_type == NodeTypes.CHEMENV.value:
            df = self.get_chemenv_property_names()
        elif node_type == NodeTypes.CRYSTAL_SYSTEM.value:
            df = self.get_crystal_system_property_names()
        elif node_type == NodeTypes.MAGNETIC_STATE.value:
            df = self.get_magnetic_states_property_names()
        elif node_type == NodeTypes.SPACE_GROUP.value:
            df = self.get_space_group_property_names()
        elif node_type == NodeTypes.OXIDATION_STATE.value:
            df = self.get_oxidation_states_property_names()

        properties=[]
        if self.output_format == 'pyarrow':
            for property_name, filed in zip(df.column_names,df.schema):
                dtype=filed.type
                properties.append((property_name,dtype))
            
        elif self.output_format == 'pandas':
            for property_name, type in zip(df.columns,df.dtypes):
                properties.append((property_name,type))

        return properties
    
    def load_nodes(self, filepath, columns=None):
        if self.output_format=='pandas':
            if self.file_type=='parquet':
                df = pd.read_parquet(filepath, columns=columns)
            elif self.file_type=='csv':
                df = pd.read_csv(filepath, index_col=0, columns=columns)
            return df
        elif self.output_format=='pyarrow':
            if self.file_type=='parquet':
                df = pq.read_table(filepath, columns=columns)
                return df
            
    def save_nodes(self, df, filepath):
        if self.file_type=='parquet':
            df.to_parquet(filepath, engine='pyarrow')
        elif self.file_type=='csv':
            df.to_csv(filepath, index=True)

class RelationshipTypes(Enum):

    MATERIAL_SPG=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.SPACE_GROUP.value}'
    MATERIAL_CRYSTAL_SYSTEM=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CRYSTAL_SYSTEM.value}'
    MATERIAL_LATTICE=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.LATTICE.value}'
    MATERIAL_SITE=f'{NodeTypes.MATERIAL.value}-CAN-{NodeTypes.SITE.value}'

    MATERIAL_CHEMENV=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CHEMENV.value}'
    MATERIAL_CHEMENV_ELEMENT=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CHEMENV_ELEMENT.value}'
    MATERIAL_ELEMENT=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.ELEMENT.value}'

    ELEMENT_OXIDATION_STATE=f'{NodeTypes.ELEMENT.value}-CAN_OCCUR-{NodeTypes.OXIDATION_STATE.value}'
    ELEMENT_CHEMENV=f'{NodeTypes.ELEMENT.value}-CAN_OCCUR-{NodeTypes.CHEMENV.value}'

    ELEMENT_GEOMETRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-GEOMETRIC_CONNECTS-{NodeTypes.ELEMENT.value}'
    ELEMENT_ELECTRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-ELECTRIC_CONNECTS-{NodeTypes.ELEMENT.value}'
    ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-GEOMETRIC_ELECTRIC_CONNECTS-{NodeTypes.ELEMENT.value}'

    CHEMENV_GEOMETRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-GEOMETRIC_CONNECTS-{NodeTypes.CHEMENV.value}'
    CHEMENV_ELECTRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-ELECTRIC_CONNECTS-{NodeTypes.CHEMENV.value}'
    CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-GEOMETRIC_ELECTRIC_CONNECTS-{NodeTypes.CHEMENV.value}'

class Relationships:

    def __init__(self, 
                relationship_dir,
                node_dir,
                file_type='parquet', 
                output_format='pandas',
                skip_init=False,
                from_scratch=False
                ):
        
        self.relationship_dir=relationship_dir
        self.node_dir=node_dir

        self.file_type=file_type
        self.output_format=output_format

        self.nodes=None

        if from_scratch:
            shutil.rmtree(self.relationship_dir)

        os.makedirs(self.relationship_dir,exist_ok=True)

        self.nodes=Nodes(node_dir=self.node_dir,
                         file_type=self.file_type,
                         output_format=self.output_format)
        
        if not skip_init:
            self.initialize_relationships()

    def get_material_spg_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_SPG.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=['symmetry-number'])
        node_b_df = self.nodes.get_space_group_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_b = {int(name): index for index, name in node_b_df['name'].items()}

        # Creating dataframe
        df = node_a_df.copy()
        
        # Removing NaN values
        df = df.dropna()

        # Making current index a column and reindexing
        df = df.reset_index().rename(columns={'index': node_a_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_b_type+'-END_ID'] = df['symmetry-number'].map(name_to_index_mapping_b).astype(int)

        df.drop(columns=['symmetry-number'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]
        self.save_relationships(df, filepath)
        return df
    
    def get_material_crystal_system_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=['symmetry-crystal_system'])
        node_b_df = self.nodes.get_crystal_system_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

        # Creating dataframe
        df = node_a_df.copy()

        # converting to lower case
        df['symmetry-crystal_system'] = df['symmetry-crystal_system'].str.lower()
        
        # Removing NaN values
        df = df.dropna()

        # Making current index a column and reindexing
        df = df.reset_index().rename(columns={'index': node_a_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_b_type+'-END_ID'] = df['symmetry-crystal_system'].map(name_to_index_mapping_b)

        df.drop(columns=['symmetry-crystal_system'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_material_lattice_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_LATTICE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=['name'])
        node_b_df = self.nodes.get_material_lattice_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

        # Creating dataframe
        df = node_a_df.copy()
        
        # Removing NaN values
        df = df.dropna()

        # Making current index a column and reindexing
        df = df.reset_index().rename(columns={'index': node_a_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_b_type+'-END_ID'] = df['name'].map(name_to_index_mapping_b).astype(int)

        df.drop(columns=['name'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_material_site_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_SITE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=['name'])
        node_b_df = self.nodes.get_material_site_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}

        # Creating dataframe
        df = node_b_df.copy()
        
        # Removing NaN values
        df = df.dropna()

        # Making current index a column and reindexing
        df = df.reset_index().rename(columns={'index': node_b_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_a_type+'-END_ID'] = df['name'].map(name_to_index_mapping_a)

        df.drop(columns=['name'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_element_oxidation_state_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_OXIDATION_STATE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_element_nodes(columns=['name','common_oxidation_states'])
        node_b_df = self.nodes.get_oxidation_states_nodes(columns=['name'])

        node_material_df = self.nodes.get_material_nodes(
                                                         columns=[
                                                            'name',
                                                            'species',
                                                            'oxidation_states-possible_valences',
                                                                ])
        
        # Mapping name to index
        name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}
        
        # Connecting Oxidation States to Elements derived from material_nodes
        oxidation_state_names=[]
        element_names=[]
        for irow, row in node_material_df.iterrows():
            possible_valences=row['oxidation_states-possible_valences']
            elements=row['species']
            if possible_valences is None or elements is None:
                continue
            for possible_valence,element in zip(possible_valences,elements):
                oxidation_state_name=f'ox_{possible_valence}'
                
                oxidation_state_names.append(oxidation_state_name)
                element_names.append(element)

        # Connecting Oxidation States to Elements derived from element_nodes
        # oxidation_state_names=[]
        # element_names=[]
        # for irow, row in node_a_df.iterrows():
        #     oxidation_states=row['common_oxidation_states']
        #     for oxidation_state in oxidation_states:
        #         oxidation_state_name=f'ox_{oxidation_state}'

        #         oxidation_state_names.append(oxidation_state_name)
        #         element_names.append(row['name'])

        data={
            f'{node_a_type}-START_ID': element_names,
            f'{node_b_type}-END_ID': oxidation_state_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)

        # This code removes duplicate relationships
        df=self.remove_duplicate_relationships(df)

        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_material_element_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=[
                                                            'name',
                                                            'species'])
        node_b_df = self.nodes.get_element_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        element_names=[]
        for irow, row in node_a_df.iterrows():
            elements=row['species']
            material_name=row['name']
            if elements is None:
                continue
            n_species=len(elements)

            material_names.extend([material_name]*n_species)
            element_names.extend(elements)

        data={
            f'{node_a_type}-START_ID': material_names,
            f'{node_b_type}-END_ID': element_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)
        

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        df=self.remove_duplicate_relationships(df)

        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df

    def get_material_chemenv_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=[
                                                            'name',
                                                            'coordination_environments_multi_weight'])
        node_b_df = self.nodes.get_chemenv_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        chemenv_names=[]
        for irow, row in node_a_df.iterrows():
            bond_connections=row['coordination_environments_multi_weight']
            material_name=row['name']
            if bond_connections is None:
                continue
            
            for coord_env in bond_connections:
                try:
                    chemenv_name=coord_env[0]['ce_symbol'].replace(':','_')
                except:
                    continue

                material_names.append(material_name)
                chemenv_names.append(chemenv_name)

        data={
            f'{node_a_type}-START_ID': material_names,
            f'{node_b_type}-END_ID': chemenv_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)
        

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        df=self.remove_duplicate_relationships(df)

        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
        
    def get_element_chemenv_relationships(self, columns=None):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        node_a_df = self.nodes.get_element_nodes( columns=['name'])
        node_b_df = self.nodes.get_chemenv_nodes( columns=['name'])

        node_material_df = self.nodes.get_material_nodes( 
                                                         columns=[
                                                            'name',
                                                            'species',
                                                            'coordination_environments_multi_weight',
                                                                ])
        
        # Mapping name to index
        name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        chemenv_names=[]
        element_names=[]
        for irow, row in node_material_df.iterrows():
            bond_connections=row['coordination_environments_multi_weight']
            material_name=row['name']
            elements=row['species']
            if bond_connections is None:
                continue
            
            for i,coord_env in enumerate(bond_connections):
                try:
                    chemenv_name=coord_env[0]['ce_symbol'].replace(':','_')
                except:
                    continue
                element_name=elements[i]

                material_names.append(material_name)
                chemenv_names.append(chemenv_name)
                element_names.append(element_name)

        data={
            f'{node_a_type}-START_ID': element_names,
            f'{node_b_type}-END_ID': chemenv_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        df=self.remove_duplicate_relationships(df)

        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_element_geometric_electric_element_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        element_df = self.nodes.get_element_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'species',
                                                        'geometric_electric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        site_element_names=[]
        nieghbor_element_names=[]

        for irow, row in node_material_df.iterrows():
            bond_connections=row['geometric_electric_consistent_bond_connections']
            elements=row['species']

            if bond_connections is None:
                continue

            for i,site_connections in enumerate(bond_connections):
                site_element_name=elements[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_element_name = elements[i_neighbor_element]

                    site_element_names.append(site_element_name)
                    nieghbor_element_names.append(nieghbor_element_name)



        data={
            f'{node_a_type}-START_ID': site_element_names,
            f'{node_b_type}-END_ID': nieghbor_element_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

        # Removing NaN values
        df = df.dropna()

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_element_geometric_element_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        element_df = self.nodes.get_element_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'species',
                                                        'geometric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        site_element_names=[]
        nieghbor_element_names=[]

        for irow, row in node_material_df.iterrows():
            bond_connections=row['geometric_consistent_bond_connections']
            elements=row['species']

            if bond_connections is None:
                continue

            for i,site_connections in enumerate(bond_connections):
                site_element_name=elements[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_element_name = elements[i_neighbor_element]

                    site_element_names.append(site_element_name)
                    nieghbor_element_names.append(nieghbor_element_name)

        data={
            f'{node_a_type}-START_ID': site_element_names,
            f'{node_b_type}-END_ID': nieghbor_element_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

        # Removing NaN values
        df = df.dropna()

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_element_electric_element_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        element_df = self.nodes.get_element_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'species',
                                                        'electric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        material_names=[]
        site_element_names=[]
        nieghbor_element_names=[]

        for irow, row in node_material_df.iterrows():
            bond_connections=row['electric_consistent_bond_connections']
            elements=row['species']

            if bond_connections is None:
                continue

            for i,site_connections in enumerate(bond_connections):
                site_element_name=elements[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_element_name = elements[i_neighbor_element]

                    site_element_names.append(site_element_name)
                    nieghbor_element_names.append(nieghbor_element_name)

        data={
            f'{node_a_type}-START_ID': site_element_names,
            f'{node_b_type}-END_ID': nieghbor_element_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

        # Removing NaN values
        df = df.dropna()

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df

    def get_chemenv_geometric_electric_chemenv_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        chemenv_df = self.nodes.get_chemenv_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'coordination_environments_multi_weight',
                                                        'geometric_electric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        site_chemenv_names=[]
        nieghbor_chemenv_names=[]
        for irow, row in node_material_df.iterrows():
            bond_connections=row['geometric_electric_consistent_bond_connections']
            chemenv_info=row['coordination_environments_multi_weight']

            if bond_connections is None or chemenv_info is None:
                continue

            chemenv_names=[]
            for coord_env in chemenv_info:
                try:
                    chemenv_name=coord_env[0]['ce_symbol'].replace(':','_')
                    chemenv_names.append(chemenv_name)
                except:
                    continue

            for i,site_connections in enumerate(bond_connections):
                site_chemenv_name=chemenv_names[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_chemenv_name = chemenv_names[i_neighbor_element]

                    site_chemenv_names.append(site_chemenv_name)
                    nieghbor_chemenv_names.append(nieghbor_chemenv_name)

        data={
            f'{node_a_type}-START_ID': site_chemenv_names,
            f'{node_b_type}-END_ID': nieghbor_chemenv_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_chemenv_geometric_chemenv_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        chemenv_df = self.nodes.get_chemenv_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'coordination_environments_multi_weight',
                                                        'geometric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        site_chemenv_names=[]
        nieghbor_chemenv_names=[]
        for irow, row in node_material_df.iterrows():
            bond_connections=row['geometric_consistent_bond_connections']
            chemenv_info=row['coordination_environments_multi_weight']

            if bond_connections is None or chemenv_info is None:
                continue

            chemenv_names=[]
            for coord_env in chemenv_info:
                try:
                    chemenv_name=coord_env[0]['ce_symbol'].replace(':','_')
                    chemenv_names.append(chemenv_name)
                except:
                    continue

            for i,site_connections in enumerate(bond_connections):
                site_chemenv_name=chemenv_names[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_chemenv_name = chemenv_names[i_neighbor_element]

                    site_chemenv_names.append(site_chemenv_name)
                    nieghbor_chemenv_names.append(nieghbor_chemenv_name)

        data={
            f'{node_a_type}-START_ID': site_chemenv_names,
            f'{node_b_type}-END_ID': nieghbor_chemenv_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df
    
    def get_chemenv_electric_chemenv_relationships(self, columns=None, remove_duplicates=True):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            df=self.load_relationships(filepath=filepath, columns=columns)
            return df
        
        # Loading nodes
        chemenv_df = self.nodes.get_chemenv_nodes( columns=['name'])
        node_material_df = self.nodes.get_material_nodes( 
                                                        columns=[
                                                        'name',
                                                        'coordination_environments_multi_weight',
                                                        'electric_consistent_bond_connections',
                                                            ])
        
        # Mapping name to index
        name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}
        
        # Connecting Materil to Chemenv derived from material_nodes
        site_chemenv_names=[]
        nieghbor_chemenv_names=[]
        for irow, row in node_material_df.iterrows():
            bond_connections=row['electric_consistent_bond_connections']
            chemenv_info=row['coordination_environments_multi_weight']

            if bond_connections is None or chemenv_info is None:
                continue

            chemenv_names=[]
            for coord_env in chemenv_info:
                try:
                    chemenv_name=coord_env[0]['ce_symbol'].replace(':','_')
                    chemenv_names.append(chemenv_name)
                except:
                    continue

            for i,site_connections in enumerate(bond_connections):
                site_chemenv_name=chemenv_names[i]
                for i_neighbor_element in site_connections:
                    i_neighbor_element=int(i_neighbor_element)
                    nieghbor_chemenv_name = chemenv_names[i_neighbor_element]

                    site_chemenv_names.append(site_chemenv_name)
                    nieghbor_chemenv_names.append(nieghbor_chemenv_name)

        data={
            f'{node_a_type}-START_ID': site_chemenv_names,
            f'{node_b_type}-END_ID': nieghbor_chemenv_names, 
        }
        df = pd.DataFrame(data)

        # Converts the element names to index and oxidation state names to index
        df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
        df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)

        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        self.save_relationships(df, filepath)
        return df

    def initialize_relationships(self):
        self.get_material_spg_relationships()
        self.get_material_crystal_system_relationships()
        self.get_material_lattice_relationships()
        self.get_material_site_relationships()
        self.get_material_chemenv_relationships() 
        self.get_material_element_relationships()
        self.get_element_oxidation_state_relationships()
        self.get_element_chemenv_relationships()
        self.get_element_geometric_electric_element_relationships()
        self.get_element_geometric_element_relationships()
        self.get_element_electric_element_relationships()
        self.get_chemenv_geometric_electric_chemenv_relationships()
        self.get_chemenv_geometric_chemenv_relationships()
        self.get_chemenv_electric_chemenv_relationships()

    def get_property_names(self, relationship_type):
        relationship_types = [type.value for type in RelationshipTypes]
        if relationship_type not in relationship_types:
            raise ValueError(f"Relationship type must be one of the following: {relationship_types}")
        
        if relationship_type == RelationshipTypes.MATERIAL_SPG.value:
            df = self.get_material_spg_relationships()
        elif relationship_type == RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM.value:
            df = self.get_material_crystal_system_relationships()
        elif relationship_type == RelationshipTypes.MATERIAL_LATTICE.value:
            df = self.get_material_lattice_relationships()
        elif relationship_type == RelationshipTypes.MATERIAL_SITE.value:
            df = self.get_material_site_relationships()
        elif relationship_type == RelationshipTypes.MATERIAL_CHEMENV.value:
            df = self.get_material_chemenv_relationships()
        elif relationship_type == RelationshipTypes.MATERIAL_ELEMENT.value:
            df = self.get_material_element_relationships()
        elif relationship_type == RelationshipTypes.ELEMENT_OXIDATION_STATE.value:
            df = self.get_element_oxidation_state_relationships()
        elif relationship_type == RelationshipTypes.ELEMENT_CHEMENV.value:
            df = self.get_element_chemenv_relationships()        
        elif relationship_type == RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT.value:
            df = self.get_element_geometric_electric_element_relationships()
        elif relationship_type == RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT.value:
            df = self.get_element_geometric_element_relationships()
        elif relationship_type == RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT.value:
            df = self.get_element_electric_element_relationships()
        elif relationship_type == RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV.value:
            df = self.get_chemenv_geometric_electric_chemenv_relationships()
        elif relationship_type == RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV.value:
            df = self.get_chemenv_geometric_chemenv_relationships()
        elif relationship_type == RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV.value:
            df = self.get_chemenv_electric_chemenv_relationships()
        

        properties=[]
        if self.output_format == 'pyarrow':
            for property_name, filed in zip(df.column_names,df.schema):
                dtype=filed.type
                properties.append((property_name,dtype))
            
        elif self.output_format == 'pandas':
            for property_name, type in zip(df.columns,df.dtypes):
                properties.append((property_name,type))

        return properties
    
    def load_relationships(self, filepath, columns=None):
        if self.output_format=='pandas':
            if self.file_type=='parquet':
                df = pd.read_parquet(filepath, columns=columns)
                return df
            elif self.file_type=='csv':
                df = pd.read_csv(filepath, index_col=0, columns=columns)
                return df
        elif self.output_format=='pyarrow':
            if self.file_type=='parquet':
                df = pq.read_table(filepath, columns=columns)
                return df
            
    def save_relationships(self, df, filepath):
        if self.file_type=='parquet':
            df.to_parquet(filepath, index=False, engine='pyarrow')
        elif self.file_type=='csv':
            df.to_csv(filepath, index=False)

    @staticmethod
    def remove_duplicate_relationships(df):
        """Expects only two columns with the that represent the id of the nodes

        Parameters
        ----------
        df : pandas.DataFrame
        """
        column_names=list(df.columns)

        df['id_tuple'] = df.apply(lambda x: tuple(sorted([x[column_names[0]], x[column_names[1]]])), axis=1)
        # Group by the sorted tuple and count occurrences
        grouped = df.groupby('id_tuple')
        weights = grouped.size().reset_index(name='weight')
        
        # Drop duplicates based on the id_tuple
        df_weighted = df.drop_duplicates(subset='id_tuple')

        # Merge with weights
        df_weighted = pd.merge(df_weighted, weights, on='id_tuple', how='left')

        # Drop the id_tuple column
        df_weighted = df_weighted.drop(columns='id_tuple')
        return df_weighted


class MaterialGraph:
    def __init__(self,
                graph_dir=os.path.join(GRAPH_DIR,'main'),
                from_scratch=False,
                skip_init=False,
                file_type='parquet',
                output_format='pandas'):
        """
        Initializes the GraphGenerator object.

        Args:
            main_graph_dir (str,optional): The directory where the main graph is stored. Defaults to MAIN_GRAPH_DIR.
            from_scratch (bool,optional): If True, deletes the graph database and recreates it from scratch.
            skip_main_init (bool,optional): If True, skips the initialization of the main nodes and relationships.

        """
        if file_type not in ['parquet','csv']:
            raise ValueError("file_type must be either 'parquet' or 'csv'")

        self.file_type=file_type
        self.output_format=output_format

        self.graph_dir=graph_dir
        self.node_dir=os.path.join(self.graph_dir,'nodes')
        self.relationship_dir=os.path.join(self.graph_dir,'relationships')
        self.sub_graphs_dir=os.path.join(self.graph_dir,'sub_graphs')

        if from_scratch and os.path.exists(self.graph_dir):
            shutil.rmtree(self.graph_dir)

        os.makedirs(self.node_dir,exist_ok=True)
        os.makedirs(self.relationship_dir,exist_ok=True)
        os.makedirs(self.sub_graphs_dir,exist_ok=True)

        self.relationships=Relationships(relationship_dir=self.relationship_dir,
                                    node_dir=self.node_dir,
                                    output_format=self.output_format, 
                                    file_type=self.file_type,
                                    skip_init=skip_init)
        self.nodes=self.relationships.nodes

    def get_node_filepaths(self):
        node_files=[os.path.join(self.node_dir,node_file) for node_file in os.listdir(self.node_dir)]
        return node_files
    
    def get_relationship_filepaths(self):
        relationship_files=[os.path.join(self.relationship_dir,relationship_file) for relationship_file in os.listdir(self.relationship_dir)]
        return relationship_files
    
    def list_nodes(self):
        node_names=[node_file.split('.')[0] for node_file in os.listdir(self.node_dir)]
        return node_names

    def list_relationships(self):
        relationship_names=[relationship_file.split('.')[0] for relationship_file in os.listdir(self.relationship_dir)]
        return relationship_names
    
    def screen_graph(self, sub_graph_name, from_scratch=False, **kwargs):
        # Define subgraph directory paths
        sub_graph_dir=os.path.join(self.sub_graphs_dir,sub_graph_name)
        if from_scratch and os.path.exists(sub_graph_dir):
            shutil.rmtree(sub_graph_dir)

        sub_node_dir=os.path.join(sub_graph_dir,'nodes')
        sub_relationship_dir=os.path.join(sub_graph_dir,'relationships')

        os.makedirs(sub_node_dir,exist_ok=True)
        os.makedirs(sub_relationship_dir,exist_ok=True)

        node_files=self.get_node_filepaths()

        for file_paths in node_files:
            filename=os.path.basename(file_paths)
            if filename == f"{NodeTypes.MATERIAL.value}.{self.file_type}":
                continue
            shutil.copy(os.path.join(self.node_dir,filename),os.path.join(sub_node_dir,filename))

        material_df=self.nodes.get_material_nodes()
        sub_material_node_file=os.path.join(sub_node_dir,f"{NodeTypes.MATERIAL.value}.{self.file_type}")

        materials_df=self.screen_material_nodes(df=material_df,**kwargs)
        self.nodes.save_nodes(df=materials_df,filepath=sub_material_node_file)

        sub_relationships=Relationships(node_dir=sub_node_dir,
                                        relationship_dir=sub_relationship_dir,
                                        file_type=self.file_type,
                                        output_format=self.output_format)
        
    @staticmethod
    def screen_material_nodes(self, df,
                        include:bool=True,
                        material_ids:List[str]=None, 
                        elements:List[str]=None,
                        compositions:List[str]=None,
                        space_groups:List[int]=None,
                        point_groups:List[str]=None,
                        magnetic_states:List[str]=None,
                        crystal_systems:List[str]=None,
                        nsites:Tuple[int,int]=None,
                        nelements:Tuple[int,int]=None,
                        energy_per_atom:Tuple[float,float]=None,
                        formation_energy_per_atom:Tuple[float,float]=None,
                        energy_above_hull:Tuple[float,float]=None,
                        band_gap:Tuple[float,float]=None,
                        cbm:Tuple[float,float]=None,
                        vbm:Tuple[float,float]=None,
                        efermi:Tuple[float,float]=None,
                        k_voigt:Tuple[float,float]=None,
                        k_reuss:Tuple[float,float]=None,
                        k_vrh:Tuple[float,float]=None,
                        g_voigt:Tuple[float,float]=None,
                        g_reuss:Tuple[float,float]=None,
                        g_vrh:Tuple[float,float]=None,
                        universal_anisotropy:Tuple[float,float]=None,
                        homogeneous_poisson:Tuple[float,float]=None,
                        is_stable:bool=None,
                        is_gap_direct:bool=None,
                        is_metal:bool=None,
                        is_magnetic:bool=None,):
        rows_to_keep=[]
        # Iterate through the rows of the dataframe
        for irow, row in df.iterrows():
            keep_material=False
            if material_ids:
                material_id=row['name:string']
                keep_material=self.is_in_list(material_id,material_ids)
            if elements:
                material_elements=row['elements:string[]'].split(';')
                for material_element in material_elements:
                    keep_material=is_in_list(val=material_element,string_list=elements, negation=include)
            if magnetic_states:
                material_magnetic_state=row['magnetic_states:string']
                keep_material=is_in_list(val=material_magnetic_state,string_list=magnetic_states, negation=include)
            if crystal_systems:
                material_crystal_system=row['crystal_system:string']
                keep_material=is_in_list(val=material_crystal_system,string_list=crystal_systems, negation=include)
            if compositions:
                material_composition=row['composition:string']
                keep_material=is_in_list(val=material_composition,string_list=compositions, negation=include)
            if space_groups:
                material_space_group=row['space_group:int']
                keep_material=is_in_list(val=material_space_group,string_list=space_groups, negation=include)
            if point_groups:
                material_point_group=row['point_group:string']
                keep_material=is_in_list(val=material_point_group,string_list=point_groups, negation=include)
            if nelements:
                min_nelements=nelements[0]
                max_nelements=nelements[1]
                material_nelements=row['nelements:int']
                keep_material=is_in_range(val=material_nelements,min_val=min_nelements,max_val=max_nelements, negation=include)
            if nsites:
                min_nsites=nsites[0]
                max_nsites=nsites[1]
                material_nsites=row['nsites:int']
                keep_material=is_in_range(val=material_nsites,min_val=min_nsites,max_val=max_nsites, negation=include)
            if energy_per_atom:
                min_energy_per_atom=energy_per_atom[0]
                max_energy_per_atom=energy_per_atom[1]
                material_energy_per_atom=row['energy_per_atom:float']
                keep_material=is_in_range(val=material_energy_per_atom,min_val=min_energy_per_atom,max_val=max_energy_per_atom, negation=include)
            if formation_energy_per_atom:
                min_formation_energy_per_atom=formation_energy_per_atom[0]
                max_formation_energy_per_atom=formation_energy_per_atom[1]
                material_formation_energy_per_atom=row['formation_energy_per_atom:float']
                keep_material=is_in_range(val=material_formation_energy_per_atom,min_val=min_formation_energy_per_atom,max_val=max_formation_energy_per_atom, negation=include)
            if energy_above_hull:
                min_energy_above_hull=energy_above_hull[0]
                max_energy_above_hull=energy_above_hull[1]
                material_energy_above_hull=row['energy_above_hull:float']
                keep_material=is_in_range(val=material_energy_above_hull,min_val=min_energy_above_hull,max_val=max_energy_above_hull, negation=include)
            if band_gap:
                min_band_gap=band_gap[0]
                max_band_gap=band_gap[1]
                material_band_gap=row['band_gap:float']
                keep_material=is_in_range(val=material_band_gap,min_val=min_band_gap,max_val=max_band_gap, negation=include)
            if cbm:
                min_cbm=cbm[0]
                max_cbm=cbm[1]
                material_cbm=row['cbm:float']
                keep_material=is_in_range(val=material_cbm,min_val=min_cbm,max_val=max_cbm, negation=include)
            if vbm:
                min_vbm=vbm[0]
                max_vbm=vbm[1]
                material_vbm=row['vbm:float']
                keep_material=is_in_range(val=material_vbm,min_val=min_vbm,max_val=max_vbm, negation=include)
            if efermi:
                min_efermi=efermi[0]
                max_efermi=efermi[1]
                material_efermi=row['efermi:float']
                keep_material=is_in_range(val=material_efermi,min_val=min_efermi,max_val=max_efermi, negation=include)
            if k_voigt:
                min_k_voigt=k_voigt[0]
                max_k_voigt=k_voigt[1]
                material_k_voigt=row['k_voigt:float']
                keep_material=is_in_range(val=material_k_voigt,min_val=min_k_voigt,max_val=max_k_voigt, negation=include)
            if k_reuss:
                min_k_reuss=k_reuss[0]
                max_k_reuss=k_reuss[1]
                material_k_reuss=row['k_reuss:float']
                keep_material=is_in_range(val=material_k_reuss,min_val=min_k_reuss,max_val=max_k_reuss, negation=include)
            if k_vrh:
                min_k_vrh=k_vrh[0]
                max_k_vrh=k_vrh[1]   
                material_k_vrh=row['k_vrh:float']
                keep_material=is_in_range(val=material_k_vrh,min_val=min_k_vrh,max_val=max_k_vrh, negation=include)
            if g_voigt:
                min_g_voigt=g_voigt[0]
                max_g_voigt=g_voigt[1]
                material_g_voigt=row['g_voigt:float']
                keep_material=is_in_range(val=material_g_voigt,min_val=min_g_voigt,max_val=max_g_voigt, negation=include)
            if g_reuss:
                min_g_reuss=g_reuss[0]
                max_g_reuss=g_reuss[1]
                material_g_reuss=row['g_reuss:float']
                keep_material=is_in_range(val=material_g_reuss,min_val=min_g_reuss,max_val=max_g_reuss, negation=include)
            if g_vrh:
                min_g_vrh=g_vrh[0]
                max_g_vrh=g_vrh[1]
                material_g_vrh=row['g_vrh:float']
                keep_material=is_in_range(val=material_g_vrh,min_val=min_g_vrh,max_val=max_g_vrh, negation=include)
            if universal_anisotropy:
                min_universal_anisotropy=universal_anisotropy[0]
                max_universal_anisotropy=universal_anisotropy[1]
                material_universal_anisotropy=row['universal_anisotropy:float']
                keep_material=is_in_range(val=material_universal_anisotropy,min_val=min_universal_anisotropy,max_val=max_universal_anisotropy, negation=include)
            if homogeneous_poisson:
                min_homogeneous_poisson=homogeneous_poisson[0]
                max_homogeneous_poisson=homogeneous_poisson[1]
                material_homogeneous_poisson=row['homogeneous_poisson:float']
                keep_material=is_in_range(val=material_homogeneous_poisson,min_val=min_homogeneous_poisson,max_val=max_homogeneous_poisson, negation=include)
            if is_stable:
                if not (is_stable ^ include):
                    keep_material=True
            if is_gap_direct:
                if not (is_gap_direct ^ include):
                    keep_material=True
            if is_metal:
                if not (is_metal ^ include):
                    keep_material=True
            if is_magnetic:
                if not (is_magnetic ^ include):
                    keep_material=True

            if keep_material:
                rows_to_keep.append(irow)
        
        filtered_df=df.iloc[rows_to_keep]
        return filtered_df

def is_in_range(val:Union[float, int],min_val:Union[float, int],max_val:Union[float, int], negation:bool=True):
    """
    Screens a list of floats to keep only those that are within a given range.

    Args:
        floats (Union[float, int]): A list of floats to be screened.
        min_val (float): The minimum value to keep.
        max_val (float): The maximum value to keep.
        negation (bool, optional): If True, returns True if the value is within the range. 
                                If False, returns True if the value is outside the range.
                                Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is within the given range.
    """
    if negation:
        return min_val <= val <= max_val
    else:
        return not (min_val <= val <= max_val)

def is_in_list(val, string_list: List, negation: bool = True) -> bool:
    """
    Checks if a value is (or is not, based on the inverse_check flag) in a given list.

    Args:
        val: The value to be checked.
        string_list (List): The list to check against.
        negation (bool, optional): If True, returns True if the value is in the list.
                                        If False, returns True if the value is not in the list.
                                        Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is (or is not) in the list based on 'inverse_check'.
    """
    return (val in string_list) if negation else (val not in string_list)



if __name__=='__main__':

    # node_dir=os.path.join('data','production','materials_project','graph_database','test','nodes')
    # nodes=Nodes(node_dir=node_dir,output_format='pandas')
    # df=nodes.get_material_nodes()
    # df=nodes.get_material_lattice_nodes()
    # df=nodes.get_material_site_nodes()

    # df=nodes.get_chemenv_element_nodes()
    # df=nodes.get_chemenv_nodes()

    # print(df.head())
    # df=nodes.get_element_nodes()
    # df=nodes.get_crystal_system_nodes()
    # df = nodes.get_magnetic_states_nodes()
    # df = nodes.get_oxidation_states_nodes()
    # df = nodes.get_space_group_nodes()
    # df = nodes.get_wyckoff_positions_nodes()

    # columns=df.column_names
    # schema=[field.type for field in df.schema]
    # for column, type in zip(columns,schema):
    #     print(f'{column} | {type}')

    # print(df['name'].to_dict())

    ################################################################################################
    # Nodes with columns
    ################################################################################################
    # node_dir=os.path.join('data','production','materials_project','graph_database','test','nodes')
    # nodes=Nodes(node_dir=node_dir,output_format='pandas')
    # df=nodes.get_element_nodes()
    
    # print(df.head())
    # # for irow, row in df.iterrows():
    # #     print(row)

    # df=nodes.get_oxidation_states_nodes()
    
    # # print(df.head())
    # # for irow, row in df.iterrows():
    # #     print(row)

    # df=nodes.get_chemenv_nodes(columns=['name'])
    # for irow, row in df.iterrows():
    #     print(row)


    # df=nodes.get_material_nodes(columns=['coordination_environments_multi_weight','species'])
    

    # df=nodes.get_material_nodes(columns=['geometric_electric_consistent_bond_connections','geometric_electric_consistent_bond_orders'])

    # # print(df.head(5))

    # print(df.iloc[1]['geometric_electric_consistent_bond_orders'])
    ################################################################################################
    # Relationships
    ################################################################################################
    # relationships=Relationships(relationship_dir=os.path.join('data','production','materials_project','graph_database','test','relationships'),
                                
    #                             node_dir=os.path.join('data','production','materials_project','graph_database','test','nodes'))
    # df=relationships.get_material_spg_relationships()
    # df=relationships.get_material_crystal_system_relationships()
    # df=relationships.get_material_lattice_relationships()
    # df=relationships.get_material_site_relationships()
    # df=relationships.get_material_chemenv_relationships() 
    # df=relationships.get_material_element_relationships()

    # df=relationships.get_element_oxidation_state_relationships()
    # df = relationships.get_element_chemenv_relationships()
    

    # df = relationships.get_element_geometric_electric_element_relationships()
    # # df = relationships.get_element_geometric_element_relationships()
    # # df = relationships.get_element_electric_element_relationships()
    # print(df.head())

    # df = relationships.get_chemenv_geometric_electric_chemenv_relationships()
    # # df=relationships.get_chemenv_geometric_chemenv_relationships()
    # # print(df.head())

    # # df=relationships.get_chemenv_electric_chemenv_relationships()
    # print(df.head())

     ################################################################################################
    # Relationships
    ################################################################################################


    material_graph=MaterialGraph(graph_dir=os.path.join(GRAPH_DIR,'main'))

    print(material_graph.list_relationships())
    print(material_graph.list_nodes())

 
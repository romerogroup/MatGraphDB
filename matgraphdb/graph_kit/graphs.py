import os
import shutil
import warnings
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils.periodic_table import get_group_period_edge_index
from matgraphdb.utils import MATERIAL_PARQUET_FILE
from matgraphdb.utils import GRAPH_DIR,PKG_DIR, get_logger
from matgraphdb.graph_kit.metadata import get_node_schema,get_relationship_schema
from matgraphdb.graph_kit.metadata import NodeTypes, RelationshipTypes
from matgraphdb.graph_kit.utils import is_in_range, is_in_list

logger=get_logger(__name__, console_out=False, log_level='debug')

#TODO: Have it so users can pass Nodes and Relationships into GraphManager
class Nodes:

    def __init__(self, 
                node_dir, 
                output_format='pandas',
                skip_init=False,
                from_scratch=False):
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.output_format=output_format
        self.node_dir=node_dir
        self.file_type='parquet'

        self.node_types=NodeTypes

        if from_scratch:
            logger.info(f"Starting from scratch. Deleting node directory {self.node_dir}")
            shutil.rmtree(self.node_dir)

        os.makedirs(self.node_dir,exist_ok=True)

        if not skip_init:
            logger.info(f"Initializing nodes")
            self.initialize_nodes()

    def get_element_nodes(self, columns=None, 
                        base_element_csv='imputed_periodic_table_values.csv',
                        from_scratch=False,
                        **kwargs):
        csv_files = glob(os.path.join(PKG_DIR,'utils',"*.csv"))
        csv_filenames = [os.path.basename(file) for file in csv_files]
        if base_element_csv not in csv_filenames:
            raise ValueError(f"base_element_csv must be one of the following: {csv_filenames}")
        warnings.filterwarnings("ignore", category=UserWarning)
        node_type=NodeTypes.ELEMENT.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df

        logger.info(f"No node file found. Attemping to create {node_type} nodes")

        df=pd.read_csv(os.path.join(PKG_DIR,'utils',base_element_csv),index_col=0)#, encoding='utf8')
        
        df['oxidation_states']=df['oxidation_states'].apply(lambda x: x.replace(']', '').replace('[', ''))
        df['oxidation_states']=df['oxidation_states'].apply(lambda x: ','.join(x.split()) )
        df['oxidation_states']=df['oxidation_states'].apply(lambda x: eval('['+x+']') )

        df['experimental_oxidation_states']=df['experimental_oxidation_states'].apply(lambda x: eval(x) )
        df['ionization_energies']=df['ionization_energies'].apply(lambda x: eval(x) )
        # for irow, row in df.iterrows():
        #     row['oxidation_states']=eval(row['oxidation_states'])
        # df['oxidation_states']=df['oxidation_states']
        
        # elements = atomic_symbols[1:]
        # elements_properties = []
        # for i, element in enumerate(elements[:]):
        #     tmp_dict=pymatgen_properties.copy()
        #     pmat_element=Element(element)
        #     for key in tmp_dict.keys():
        #         try:
        #             value=getattr(pmat_element,key)
        #         except:
        #             value=None
        #         if isinstance(value,FloatWithUnit):
        #             value=value.real
        #         if isinstance(value,dict):
        #             value=[(key2,value2) for key2,value2 in value.items()]
        #         tmp_dict[key]=value
        #     elements_properties.append(tmp_dict)

        # df = pd.DataFrame(elements_properties)
        df['name'] = df['symbol']
        df['type'] = node_type

        if columns:
            df = df[columns]

        schema = get_node_schema(NodeTypes.ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_crystal_system_nodes(self, columns=None, 
                                from_scratch=False,
                                **kwargs):
        node_type=NodeTypes.CRYSTAL_SYSTEM.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")

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

        schema=get_node_schema(NodeTypes.CRYSTAL_SYSTEM)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None
        
        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df

    def get_magnetic_states_nodes(self, columns=None,
                                from_scratch=False,
                                **kwargs):
        node_type=NodeTypes.MAGNETIC_STATE.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")
            
        magnetic_states = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
        magnetic_states_properties = []
        for i, magnetic_state in enumerate(magnetic_states[:]):
            magnetic_states_properties.append({"magnetic_state": magnetic_state})

        df = pd.DataFrame(magnetic_states_properties)
        df['name'] = df['magnetic_state']
        df['type'] = node_type
        if columns:
            df = df[columns]

        schema=get_node_schema(NodeTypes.MAGNETIC_STATE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None
        
        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_oxidation_states_nodes(self, columns=None, 
                                   from_scratch=False, 
                                   **kwargs):
        node_type=NodeTypes.OXIDATION_STATE.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        

        logger.info(f"No node file found. Attemping to create {node_type} nodes")
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

        logger.info(f"Saving {node_type} nodes to {filepath}")

        schema=get_node_schema(NodeTypes.OXIDATION_STATE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")
        self.save_nodes(parquet_table, filepath)
 
        return df
    
    def get_space_group_nodes(self, columns=None,
                            from_scratch=False, 
                            **kwargs):
        node_type=NodeTypes.SPACE_GROUP.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")
        
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        space_groups_properties = []
        for i, space_group in enumerate(space_groups[:]):
            spg_num=space_group.split('_')[1]
            space_groups_properties.append({"spg": int(spg_num)})

        df = pd.DataFrame(space_groups_properties)
        df['name'] = df['spg'].astype(str)
        df['type'] = node_type
        if columns:
            df = df[columns]

        schema=get_node_schema(NodeTypes.SPACE_GROUP)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_chemenv_nodes(self, columns=None, 
                        from_scratch=False, 
                        **kwargs):
        node_type=NodeTypes.CHEMENV.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")

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

        schema=get_node_schema(NodeTypes.CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_wyckoff_positions_nodes(self, columns=None, 
                                from_scratch=False, 
                                **kwargs):
        node_type=NodeTypes.SPG_WYCKOFF.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")
        
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

        schema=get_node_schema(NodeTypes.SPG_WYCKOFF)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_material_nodes(self, columns=None, 
                           from_scratch=False, 
                           **kwargs):
        node_type=NodeTypes.MATERIAL.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")
        try:
            df = pd.read_parquet(MATERIAL_PARQUET_FILE)
        except Exception as e:
            logger.error(f"Error reading {node_type} parquet file: {e}")
            return None
        
        df['name'] = df['material_id']
        df['type'] = node_type
        if columns:
            df = df[columns]
        
        
        schema = get_node_schema(NodeTypes.MATERIAL)

        logger.info(f"Saving {node_type} nodes to {filepath}")

        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        pq.write_table(parquet_table, filepath)

        logger.info(f"Finished saving {node_type} nodes to {filepath}")
        return df
    
    def get_material_lattice_nodes(self, columns=None, 
                                from_scratch=False,
                                **kwargs):
        node_type=NodeTypes.LATTICE.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")

        df = self.get_material_nodes(columns=['material_id', 'lattice', 'a', 'b', 'c', 
                                                                 'alpha', 'beta', 'gamma', 
                                                                 'crystal_system','volume'])

        df['name'] = df['material_id']
        df['type'] = node_type
        if columns:
            df = df[columns]

        schema=get_node_schema(NodeTypes.LATTICE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df
    
    def get_material_site_nodes(self, columns=None, 
                            from_scratch=False,
                            **kwargs):
        node_type=NodeTypes.SITE.value

        logger.info(f"Getting {node_type} nodes")

        filepath=os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath) and not from_scratch:
            logger.info(f"Trying to load {node_type} nodes from {filepath}")
            df=self.load_nodes(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No node file found. Attemping to create {node_type} nodes")

        df = self.get_material_nodes(columns=['material_id', 'lattice', 'frac_coords', 'species'])
        all_species=[]
        all_coords=[]
        all_lattices=[]
        all_ids=[]
        
        for irow, row in df.iterrows():
            if irow%10000==0:
                logger.info(f"Processing row {irow}")
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

        schema=get_node_schema(NodeTypes.SITE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {node_type} nodes to {filepath}")

        self.save_nodes(parquet_table, filepath)
        return df

    def initialize_nodes(self):
        self.get_material_nodes()
        self.get_element_nodes()
        self.get_chemenv_nodes()
        self.get_crystal_system_nodes()
        self.get_magnetic_states_nodes()
        self.get_space_group_nodes()
        self.get_oxidation_states_nodes()
        
        self.get_wyckoff_positions_nodes()

        self.get_material_lattice_nodes()
        self.get_material_site_nodes()

    def get_property_names(self, node_type):
        logger.info(f"Getting property names for {node_type} nodes")
        filepath=self.get_node_filepaths(node_type=node_type)
        properties = Nodes.get_column_names(filepath)
        for property in properties:
            logger.info(f"Property: {property}")
        return properties
    
    def load_nodes(self, filepath, columns=None, include_cols=True, **kwargs):
        if not include_cols:
            metadata = pq.read_metadata(filepath)
            all_columns = []
            for filed_schema in metadata.schema:
                
                # Only want top column names
                max_defintion_level=filed_schema.max_definition_level
                if max_defintion_level!=1:
                    continue

                all_columns.append(filed_schema.name)

            columns = [col for col in all_columns if col not in columns]

        if self.output_format=='pandas':
            df = pd.read_parquet(filepath, columns=columns)
        elif self.output_format=='pyarrow':
            df = pq.read_table(filepath, columns=columns)
        return df
            
    def save_nodes(self, df, filepath):
        pq.write_table(df, filepath)

    def get_node_filepaths(self, node_type=None):
        node_types = [type.value for type in NodeTypes]
        if node_type not in node_types:
            raise ValueError(f"Node type must be one of the following: {node_types}")
        
        # Construct the file path directly
        filepath = os.path.join(self.node_dir, f'{node_type}.{self.file_type}')

        return filepath

    def to_neo4j(self, node_path, save_path):
        logger.info(f"Converting node to Neo4j : {node_path}")
        node_type=os.path.basename(node_path).split('.')[0]

        logger.debug(f"Node type: {node_type}")

        metadata = pq.read_metadata(node_path)
        column_types = {}
        neo4j_column_name_mapping={}
        for filed_schema in metadata.schema:
            # Only want top column names
            type=filed_schema.physical_type
            
            field_path=filed_schema.path.split('.')
            name=field_path[0]

            is_list=False
            if len(field_path)>1:
                is_list=field_path[1] == 'list'

            column_types[name] = {}
            column_types[name]['type']=type
            column_types[name]['is_list']=is_list
            
            if type=='BYTE_ARRAY':
               neo4j_type ='string'
            if type=='BOOLEAN':
                neo4j_type='boolean'
            if type=='DOUBLE':
                neo4j_type='float'
            if type=='INT64':
                neo4j_type='int'

            if is_list:
                neo4j_type+='[]'

            column_types[name]['neo4j_type'] = f'{name}:{neo4j_type}'
            column_types[name]['neo4j_name'] = f'{name}:{neo4j_type}'

            neo4j_column_name_mapping[name]=f'{name}:{neo4j_type}'

        neo4j_column_name_mapping['type']=':LABEL'

        df=self.load_nodes(filepath=node_path)
        df.rename(columns=neo4j_column_name_mapping, inplace=True)
        df.index.name = f'{node_type}:ID({node_type}-ID)'

        os.makedirs(save_path,exist_ok=True)

        save_file=os.path.join(save_path,f'{node_type}.csv')
        logger.info(f"Saving {node_type} nodes to {save_file}")


        df.to_csv(save_file, index=True)

        logger.info(f"Finished converting node to Neo4j : {node_type}")

    @staticmethod
    def get_column_names(filepath):
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)
        return all_columns


class Relationships:

    def __init__(self, 
                relationship_dir,
                node_dir,
                output_format='pandas',
                skip_init=False,
                from_scratch=False
                ):
        
        self.relationship_dir=relationship_dir
        self.node_dir=node_dir

        self.file_type='parquet'
        self.output_format=output_format

        self.relationship_types=RelationshipTypes

        if from_scratch:
            logger.info(f"Starting from scratch. Deleting relationship directory {self.relationship_dir}")
            shutil.rmtree(self.relationship_dir)

        os.makedirs(self.relationship_dir,exist_ok=True)

        self.nodes=Nodes(node_dir=self.node_dir,
                         output_format=self.output_format)
        
        if not skip_init:
            logger.info(f"Initializing relationships")
            self.initialize_relationships()

    def get_material_spg_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_SPG.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes(columns=['space_group'])
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
        df[node_b_type+'-END_ID'] = df['space_group'].map(name_to_index_mapping_b).astype(int)

        df.drop(columns=['space_group'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_SPG)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_material_crystal_system_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
        # Loading nodes
        node_a_df = self.nodes.get_material_nodes( columns=['crystal_system'])
        node_b_df = self.nodes.get_crystal_system_nodes( columns=['name'])

        # Mapping name to index
        name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

        # Creating dataframe
        df = node_a_df.copy()

        # converting to lower case
        df['crystal_system'] = df['crystal_system'].str.lower()
        
        # Removing NaN values
        df = df.dropna()

        # Making current index a column and reindexing
        df = df.reset_index().rename(columns={'index': node_a_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_b_type+'-END_ID'] = df['crystal_system'].map(name_to_index_mapping_b)

        df.drop(columns=['crystal_system'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_material_lattice_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_LATTICE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_LATTICE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_material_site_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_SITE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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
        df = df.reset_index().rename(columns={'index': node_a_type+'-START_ID'})

        # Adding node b ID with the mapping
        df[node_b_type+'-END_ID'] = df['name'].map(name_to_index_mapping_a)

        df.drop(columns=['name'], inplace=True)
        df['TYPE'] = relationship_type

        df['weight'] = 1.0

        if columns:
            df = df[columns]

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_SITE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    # def get_material_similarity_relationships(self,material_columns, relationship_name, columns=None, remove_duplicates=True, similarity_cutoff=0.8, **kwargs):
    #     material_df = self.nodes.get_material_nodes(columns=['space_group'])
    #     similarity_scores=[]
    #     for irow in material_df:
    #         material_i_encoding= material_df.loc[irow][  material_columns  ]
    #         for jrow in material_df:
    #             material_j_encoding= material_df.loc[jrow][ material_columns  ]
    #             similarity_score=cosine_similarity(material_i_encoding,material_j_encoding)
    #             if similarity_score > similarity_cutoff:
    #                   similarity_scores.append((irow ,jrow, similarity_score))

    #    df = pd.dataframe(similarity_scores), columns=['START_ID', 'END_ID']
    #    self.save_relationships(df, filepath=relationship_name.parquet)

    def get_element_oxidation_state_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_OXIDATION_STATE.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
        # Loading nodes
        node_a_df = self.nodes.get_element_nodes(columns=['name','experimental_oxidation_states'])
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

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_OXIDATION_STATE)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_material_element_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)

        return df

    def get_material_chemenv_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.MATERIAL_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.MATERIAL_CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
        
    def get_element_chemenv_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_element_geometric_electric_element_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_element_geometric_element_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_element_electric_element_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df

    def get_chemenv_geometric_electric_chemenv_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_chemenv_geometric_chemenv_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df
    
    def get_chemenv_electric_chemenv_relationships(self, columns=None, remove_duplicates=True, **kwargs):

        # Defining node types and relationship type
        relationship_type=RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.info(f"Getting {relationship_type} relationships")

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
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

        schema=get_relationship_schema(RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
        return df

    def get_element_group_period_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        relationship_type=RelationshipTypes.ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT.value
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        # Loading relationship if it exists
        filepath=os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        if os.path.exists(filepath):
            logger.info(f"Trying to load {relationship_type} relationships from {filepath}")
            df=self.load_relationships(filepath=filepath, columns=columns, **kwargs)
            return df
        
        logger.info(f"No relationship file found. Attemping to create {relationship_type} relationships")
        
        # Loading nodes
        element_df = self.nodes.get_element_nodes( columns=['name','atomic_number','extended_group','period','symbol'])
        
        # Mapping name to index
        name_to_index_mapping = {name: index for index, name in element_df['name'].items()}
        
        edge_index=get_group_period_edge_index(element_df)

        df = pd.DataFrame(edge_index, columns=[f'{node_a_type}-START_ID', f'{node_b_type}-END_ID'])
        
        # Removing NaN values
        df = df.dropna().astype(int)

        # This code removes duplicate relationships
        if remove_duplicates:
            df=self.remove_duplicate_relationships(df)
        
        df['TYPE'] = relationship_type
        if columns:
            df = df[columns]

        schema=get_relationship_schema(RelationshipTypes.ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT)
        try:
            parquet_table=pa.Table.from_pandas(df,schema=schema)
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")
            return None

        logger.info(f"Saving {relationship_type} relationships to {filepath}")
        self.save_relationships(parquet_table, filepath)
    
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
        filepath=self.get_relationship_filepaths(relationship_type=relationship_type)
        properties = Relationships.get_column_names(filepath)
        return properties
    
    def load_relationships(self, filepath, columns=None, include_cols=True):
        if not include_cols:
            metadata = pq.read_metadata(filepath)
            all_columns = []
            for filed_schema in metadata.schema:
                
                # Only want top column names
                max_defintion_level=filed_schema.max_definition_level
                if max_defintion_level!=1:
                    continue

                all_columns.append(filed_schema.name)

            columns = [col for col in all_columns if col not in columns]
        if self.output_format=='pandas':
            df = pd.read_parquet(filepath, columns=columns)
        elif self.output_format=='pyarrow':
            df = pq.read_table(filepath, columns=columns)
        return df
            
    def save_relationships(self, df, filepath):
        pq.write_table(df, filepath)

    def get_relationship_filepaths(self, relationship_type=None):
        relationship_types = [type.value for type in RelationshipTypes]
        if relationship_type not in relationship_types:
            raise ValueError(f"Relationship type must be one of the following: {relationship_types}")
        # Construct the file path directly
        filepath = os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        return filepath

    def to_neo4j(self, relationship_path, save_path):
        logger.info(f"Converting relationship to Neo4j : {relationship_path}")

        relationship_type=os.path.basename(relationship_path).split('.')[0]
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.debug(f"Relationship type: {relationship_type}")

        metadata = pq.read_metadata(relationship_path)
        column_types = {}
        neo4j_column_name_mapping={}
        for filed_schema in metadata.schema:
            # Only want top column names
            type=filed_schema.physical_type
            
            field_path=filed_schema.path.split('.')
            name=field_path[0]

            is_list=False
            if len(field_path)>1:
                is_list=field_path[1] == 'list'

            column_types[name] = {}
            column_types[name]['type']=type
            column_types[name]['is_list']=is_list
            
            if type=='BYTE_ARRAY':
               neo4j_type ='string'
            if type=='BOOLEAN':
                neo4j_type='boolean'
            if type=='DOUBLE':
                neo4j_type='float'
            if type=='INT64':
                neo4j_type='int'

            if is_list:
                neo4j_type+='[]'

            column_types[name]['neo4j_type'] = f'{name}:{neo4j_type}'
            column_types[name]['neo4j_name'] = f'{name}:{neo4j_type}'

            neo4j_column_name_mapping[name]=f'{name}:{neo4j_type}'

        neo4j_column_name_mapping['TYPE']=':LABEL'

        neo4j_column_name_mapping[f'{node_a_type}-START_ID']=f':START_ID({node_a_type}-ID)'
        neo4j_column_name_mapping[f'{node_b_type}-END_ID']=f'END_ID({node_a_type}-ID)'

        df=self.load_relationships(filepath=relationship_path)


        df.rename(columns=neo4j_column_name_mapping, inplace=True)

        os.makedirs(save_path,exist_ok=True)

       
        save_file=os.path.join(save_path,f'{relationship_type}.csv')

        logger.debug(f"Saving {relationship_type} relationship_path to {save_file}")

        df.to_csv(save_file, index=False)

        logger.info(f"Finished converting relationship to Neo4j : {relationship_type}")

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

    @staticmethod
    def get_column_names(filepath):
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)
        return all_columns

class GraphManager:
    def __init__(self,
                graph_dir=os.path.join(GRAPH_DIR,'main'),
                from_scratch=False,
                skip_init=False,
                output_format='pandas'):
        """
        Initializes the GraphGenerator object.

        Args:
            main_graph_dir (str,optional): The directory where the main graph is stored. Defaults to MAIN_GRAPH_DIR.
            from_scratch (bool,optional): If True, deletes the graph database and recreates it from scratch.
            skip_main_init (bool,optional): If True, skips the initialization of the main nodes and relationships.

        """
        self.file_type='parquet'
        self.output_format=output_format

        self.graph_dir=graph_dir
        self.node_dir=os.path.join(self.graph_dir,'nodes')
        self.relationship_dir=os.path.join(self.graph_dir,'relationships')
        self.sub_graphs_dir=os.path.join(self.graph_dir,'sub_graphs')

        if from_scratch and os.path.exists(self.graph_dir):
            logger.info(f"Starting from scratch. Deleting graph directory {self.graph_dir}")
            shutil.rmtree(self.graph_dir)

        os.makedirs(self.node_dir,exist_ok=True)
        os.makedirs(self.relationship_dir,exist_ok=True)
        os.makedirs(self.sub_graphs_dir,exist_ok=True)

        self.relationships=Relationships(relationship_dir=self.relationship_dir,
                                    node_dir=self.node_dir,
                                    output_format=self.output_format, 
                                    skip_init=skip_init)
        self.nodes=self.relationships.nodes

    def get_node_filepaths(self):
        node_files=glob(os.path.join(self.node_dir,'*.parquet'))
        for node_file in node_files:
            logger.debug(f"Node file: {node_file}")
        return node_files
    
    def get_relationship_filepaths(self):
        relationship_files=glob(os.path.join(self.relationship_dir,'*.parquet'))
        for relationship_file in relationship_files:
            logger.debug(f"Relationship file: {relationship_file}")
        return relationship_files
    
    def list_nodes(self):
        node_files=glob(os.path.join(self.node_dir,'*.parquet'))
        node_names=[os.path.basename(node_file).split('.')[0] for node_file in node_files]
        logger.debug(f"Node names: {node_names}")
        return node_names

    def list_relationships(self):
        relationship_files=glob(os.path.join(self.relationship_dir,'*.parquet'))
        relationship_names=[os.path.basename(relationship_file).split('.')[0] for relationship_file in relationship_files]
        logger.debug(f"Relationship names: {relationship_names}")
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

    def to_neo4j(self, save_path):
        logger.info(f"Converting graph to Neo4j")

        node_paths=self.get_node_filepaths()
        relationship_paths=self.get_relationship_filepaths()

        for node_path in node_paths:
            self.nodes.to_neo4j_task(node_path, save_path)

        for relationship_path in relationship_paths:
            self.relationships.to_neo4j_task(relationship_path, save_path)

        logger.info(f"Finished converting graph to Neo4j")

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
    # node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes')
    node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes')
    nodes=Nodes(node_dir=node_dir,skip_init=True)
    # df=nodes.get_material_nodes()
    df=nodes.get_element_nodes(base_element_csv='interim_periodic_table_values.csv', from_scratch=True)
    # df=nodes.get_pre_imputed_element_nodes()
    print(df.head())
    print(df['modulus_bulk'])
    # properties=nodes.get_property_names(node_type='MATERIAL')
    # print(properties)
    # # print(df.head())
    # # # for irow, row in df.iterrows():
    # # #     print(row)

    # # df=nodes.get_oxidation_states_nodes()
    
    # # # print(df.head())
    # # # for irow, row in df.iterrows():
    # # #     print(row)

    # # df=nodes.get_chemenv_nodes(columns=['name'])
    # # for irow, row in df.iterrows():
    # #     print(row)


    # df=nodes.get_material_nodes(columns=['name'], include_cols=False)
    # df=nodes.get_material_nodes()
    # print(df.head(5))
    # print('name' not in list(df.columns))
    # # df=df.dropna()
    # s=df['elasticity-debye_temperature']
    # clean_series = s.loc[s.notnull()]
    # print(clean_series)


    ################################################################################################
    # Relationships
    # ################################################################################################
    # relationships=Relationships(relationship_dir=os.path.join('data','production','materials_project','graph_database','main','relationships'),
    #                             node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes'))
    # df=relationships.get_material_spg_relationships()
    # df=relationships.get_material_crystal_system_relationships()
    # df=relationships.get_material_lattice_relationships()
    # df=relationships.get_material_site_relationships()
    # df=relationships.get_material_chemenv_relationships() 
    # df=relationships.get_material_element_relationships()

    # df = relationships.get_element_group_period_relationships()

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
    # Material Graph
    ################################################################################################

    # material_graph=MaterialGraph(graph_dir=os.path.join(GRAPH_DIR,'main'))

    # print(material_graph.list_relationships())
    # print(material_graph.list_nodes())


    ###############################################################################################
    # Nodes with columns
    ################################################################################################
    # node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes')
    # node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes')

    # neo4j_dir=os.path.join('data','production','materials_project','graph_database','neo4j_csv','nodes')
    # nodes=Nodes(node_dir=node_dir,skip_init=True)


    # node_path=os.path.join(node_dir,'ELEMENT.parquet')

    # nodes.to_neo4j(node_path=node_path, save_path=neo4j_dir)



    # relationship_dir=os.path.join('data','production','materials_project','graph_database','main','relationships')
    # relationships=Relationships(relationship_dir=relationship_dir,
    #                             node_dir=os.path.join('data','production','materials_project','graph_database','main','nodes'), 
    #                             skip_init=True)
    
    # neo4j_dir=os.path.join('data','production','materials_project','graph_database','neo4j_csv','relationships')
    # relationships.to_neo4j(relationship_path=os.path.join(relationship_dir,'MATERIAL-HAS-CRYSTAL_SYSTEM.parquet'), 
    #                        save_path=neo4j_dir)

 

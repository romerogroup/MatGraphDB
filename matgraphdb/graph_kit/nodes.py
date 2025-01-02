from glob import glob
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from matgraphdb import config
# from matgraphdb.utils.chem_utils.coord_geometry import mp_coord_encoding
from matgraphdb.graph_kit.metadata import get_node_schema
from matgraphdb.graph_kit.metadata import NodeTypes

logger = logging.getLogger(__name__)

class Nodes:
    """
    A base class to manage node operations, including creating, loading, and saving nodes as Parquet files, 
    with options to format data as either Pandas or PyArrow DataFrames. Subclasses should implement custom 
    logic for node creation and schema generation.
    """
    
    def __init__(self, node_type, node_dir, output_format='pandas'):
        """
        Initializes a Nodes object with the given node type, directory, and output format.

        Parameters:
        -----------
        node_type : str
            The type of node to manage.
        node_dir : str
            Directory where node files will be stored.
        output_format : str, optional
            Format for loading data, either 'pandas' (default) or 'pyarrow'. Must be one of these two options.
        
        Raises:
        -------
        ValueError
            If output_format is not 'pandas' or 'pyarrow'.
        """
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.node_type = node_type
        self.node_dir = node_dir
        os.makedirs(self.node_dir,exist_ok=True)

        self.output_format = output_format
        self.file_type = 'parquet'
        self.filepath =  os.path.join(self.node_dir, f'{self.node_type}.{self.file_type}')
        self.schema = self.create_schema() 

        self.get_dataframe()
        
    def get_dataframe(self, columns=None, include_cols=True, from_scratch=False, **kwargs):
        """
        Loads or creates a node dataframe. If the node file exists and from_scratch is False, 
        the existing file will be loaded. Otherwise, it will call the create_nodes() method to 
        create the nodes and save them.

        Parameters:
        -----------
        columns : list of str, optional
            A list of column names to load. If None, all columns will be loaded.
        include_cols : bool, optional
            If True (default), the specified columns will be included. If False, 
            they will be excluded from the loaded dataframe.
        from_scratch : bool, optional
            If True, forces the creation of a new node dataframe even if a file exists. Default is False.
        **kwargs : dict
            Additional arguments passed to the create_nodes() method.

        Returns:
        --------
        pandas.DataFrame or pyarrow.Table
            The loaded or newly created node data.

        Raises:
        -------
        ValueError
            If the 'name' field is missing from the created nodes dataframe.
        """

        if os.path.exists(self.filepath) and not from_scratch:
            logger.info(f"Trying to load {self.node_type} nodes from {self.filepath}")
            df = self.load_dataframe(filepath=self.filepath, columns=columns, include_cols=include_cols, **kwargs)
            return df
        
        logger.info(f"No node file found. Attempting to create {self.node_type} nodes")
        df = self.create_nodes(**kwargs)  # Subclasses will define this

        # Ensure the 'name' field is present
        if 'name' not in df.columns:
            raise ValueError(f"The 'name' field must be defined for {self.node_type} nodes. Define this in the create_nodes.")
        df['name'] = df['name']  # Ensure 'name' is set
        df['type'] = self.node_type  # Ensure 'type' is set
        if columns:
            df = df[columns]

        if not self.schema:
            logger.error(f"No schema set for {self.node_type} nodes")
            return None

        self.save_dataframe(df, self.filepath)
        return df
    
    def get_property_names(self):
        """
        Retrieves and logs the column names (properties) of the node data from the Parquet file.

        Returns:
        --------
        list of str
            A list of column names in the node file.
        """
        properties = Nodes.get_column_names(self.filepath)
        for property in properties:
            logger.info(f"Property: {property}")
        return properties

    def create_nodes(self, **kwargs):
        """
        Abstract method for creating nodes. Must be implemented by subclasses to define the logic 
        for creating nodes specific to the node type.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        if self.__class__.__name__ != 'Nodes':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass
    
    def create_schema(self, **kwargs):
        """
        Abstract method for creating a Parquet schema. Must be implemented by subclasses to define 
        the schema for the node data.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        if self.__class__.__name__ != 'Nodes':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def load_dataframe(self, filepath, columns=None, include_cols=True, **kwargs):
        """
        Loads node data from a Parquet file, optionally filtering by columns.

        Parameters:
        -----------
        filepath : str
            Path to the Parquet file.
        columns : list of str, optional
            A list of column names to load. If None, all columns will be loaded.
        include_cols : bool, optional
            If True (default), the specified columns will be included. If False, 
            they will be excluded from the loaded dataframe.
        **kwargs : dict
            Additional arguments for reading the Parquet file.

        Returns:
        --------
        pandas.DataFrame or pyarrow.Table
            The loaded node data.
        """
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

        try:
            if self.output_format=='pandas':
                df = pd.read_parquet(filepath, columns=columns)
            elif self.output_format=='pyarrow':
                df = pq.read_table(filepath, columns=columns)

            return df
        except Exception as e:
            logger.error(f"Error loading {self.node_type} nodes from {filepath}: {e}")
            return None

    def save_dataframe(self, df, filepath):
        """
        Saves the given dataframe to a Parquet file at the specified filepath.

        Parameters:
        -----------
        df : pandas.DataFrame or pyarrow.Table
            The node data to save.
        filepath : str
            The path where the Parquet file should be saved.

        Raises:
        -------
        Exception
            If there is an error during the save process.
        """
        try:
            parquet_table = pa.Table.from_pandas(df, self.schema)
            pq.write_table(parquet_table, filepath)
            logger.info(f"Finished saving {self.node_type} nodes to {filepath}")
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")

    def to_neo4j(self, save_dir):
        """
        Converts the node data to a CSV file for importing into Neo4j. Saves the file in the given directory.

        Parameters:
        -----------
        save_dir : str
            Directory where the CSV file will be saved.
        """
        logger.info(f"Converting node to Neo4j : {self.filepath}")
        node_type=os.path.basename(self.filepath).split('.')[0]

        logger.debug(f"Node type: {node_type}")

        metadata = pq.read_metadata(self.filepath)
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

        df=self.load_nodes(filepath=self.filepath)
        df.rename(columns=neo4j_column_name_mapping, inplace=True)
        df.index.name = f'{node_type}:ID({node_type}-ID)'

        os.makedirs(save_dir,exist_ok=True)

        save_file=os.path.join(save_dir,f'{node_type}.csv')
        logger.info(f"Saving {node_type} nodes to {save_file}")


        df.to_csv(save_file, index=True)

        logger.info(f"Finished converting node to Neo4j : {node_type}")
    
    @staticmethod
    def get_column_names(filepath):
        """
        Extracts and returns the top-level column names from a Parquet file.

        This method reads the metadata of a Parquet file and extracts the names of the top-level columns.
        It filters out nested columns or columns with a `max_definition_level` other than 1, ensuring that
        only primary, non-nested columns are included in the output.

        Args:
            filepath (str): The file path to the Parquet (.parquet) file.

        Returns:
            list of str: A list containing the names of the top-level columns in the Parquet file.

        Example:
            columns = Nodes.get_column_names('data/example.parquet')
            print(columns)
            # Output: ['column1', 'column2', 'column3']
        
        """
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)
        return all_columns


class MaterialNodes(Nodes):
    """
    A specialized class for handling Material nodes within the node management system.

    This class inherits from the `Nodes` base class and is designed to manage nodes of type 'Material'.
    It defines the schema for Material nodes and provides functionality to create these nodes from 
    an external Parquet file.

    Attributes:
        node_dir (str): The directory where the Material node data is stored.
        output_format (str): The format for returning node data, defaulting to 'pandas'. Options might include 
                             'pandas' for a DataFrame or other formats supported by the system.

    Methods:
        create_schema(): 
            Defines and returns the schema for Material nodes. The schema is fetched based on the Material node type.
        
        create_nodes(**kwargs): 
            Reads the material node data from a Parquet file, processes it, and returns the nodes as a pandas DataFrame.
            In case of an error during the reading process, it logs the error and returns `None`.

    Example:
        material_nodes = MaterialNodes(node_dir='path/to/material_nodes')
    """
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.MATERIAL.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Material nodes
        return get_node_schema(NodeTypes.MATERIAL)

    def create_nodes(self, **kwargs):
        # The logic for creating material nodes
        try:
            df = pd.read_parquet(MATERIAL_PARQUET_FILE)
            df['name'] = df.index + 1  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error reading material parquet file: {e}")
            return None
        return df

class ElementNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        """
        Initializes the `ElementNodes` class.

        Args:
            node_dir (str): The directory where the node files are located.
            output_format (str): The format in which the nodes will be outputted. Default is 'pandas'.
        
        Inherits the initialization from the parent `Nodes` class, setting the node type 
        as 'Element' and configuring the output format for the node data.
        """
        super().__init__(node_type=NodeTypes.ELEMENT.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        """
        Defines and returns the schema for the `ElementNodes` class.

        Returns:
            dict: A dictionary representing the schema for element nodes, 
            which includes all necessary fields for storing element information 
            such as oxidation states, ionization energies, and other properties.

        This method uses the `get_node_schema` function to generate the schema 
        specific to the 'Element' node type.
        """
        # Define and return the schema for Element nodes
        return get_node_schema(NodeTypes.ELEMENT)
    
    def create_nodes(self, base_element_csv='imputed_periodic_table_values.csv', **kwargs):
        """
        Reads the element data from a CSV file and processes it for node creation.

        Args:
            base_element_csv (str): The filename of the CSV containing element data. 
                                    Defaults to 'imputed_periodic_table_values.csv'.
            **kwargs: Additional arguments for flexibility, if needed.

        Returns:
            pandas.DataFrame: A DataFrame containing the processed element data, 
            ready to be used as nodes in the application.
        """
        # Ensure the CSV file exists
        csv_files = glob(os.path.join(PKG_DIR, 'utils', "*.csv"))
        csv_filenames = [os.path.basename(file) for file in csv_files]
        if base_element_csv not in csv_filenames:
            raise ValueError(f"base_element_csv must be one of the following: {csv_filenames}")

        # Suppress warnings during node creation
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            df = pd.read_csv(os.path.join(PKG_DIR, 'utils', base_element_csv), index_col=0)
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: x.replace(']', '').replace('[', ''))
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: ','.join(x.split()) )
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: eval('['+x+']') )
            df['experimental_oxidation_states']=df['experimental_oxidation_states'].apply(lambda x: eval(x) )
            df['ionization_energies']=df['ionization_energies'].apply(lambda x: eval(x) )

            df['name'] = df['symbol']  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error reading element CSV file: {e}")
            return None

        return df
    

class CrystalSystemNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.CRYSTAL_SYSTEM.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for CrystalSystem nodes
        return get_node_schema(NodeTypes.CRYSTAL_SYSTEM)
    
    def create_nodes(self, **kwargs):
        """
        Creates Crystal System nodes if no file exists, otherwise loads them from a file.
        """
        try:
            crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
            crystal_systems_properties = [{"crystal_system": cs} for cs in crystal_systems]

            df = pd.DataFrame(crystal_systems_properties)
            df['name'] = df['crystal_system']  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error creating crystal system nodes: {e}")
            return None

        return df
    
class MagneticStatesNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.MAGNETIC_STATE.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Magnetic State nodes
        return get_node_schema(NodeTypes.MAGNETIC_STATE)

    def create_nodes(self, **kwargs):
        """
        Creates Magnetic State nodes if no file exists, otherwise loads them from a file.
        """
        # Define magnetic states
        try:
            magnetic_states = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
            magnetic_states_properties = [{"magnetic_state": ms} for ms in magnetic_states]

            df = pd.DataFrame(magnetic_states_properties)
            df['name'] = df['magnetic_state']  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error creating magnetic state nodes: {e}")
            return None
        return df
    
class OxidationStatesNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.OXIDATION_STATE.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Oxidation State nodes
        return get_node_schema(NodeTypes.OXIDATION_STATE)

    def create_nodes(self,  **kwargs):
        """
        Creates Oxidation State nodes if no file exists, otherwise loads them from a file.
        """
    
        # Retrieve material nodes with possible valences
        try:
            # material_df = self.get_material_nodes(columns=['oxidation_states-possible_valences'])
            # possible_oxidation_state_names = []
            # possible_oxidation_state_valences = []

            # # Iterate through the material DataFrame to collect possible valences
            # for _, row in material_df.iterrows():
            #     possible_valences = row['oxidation_states-possible_valences']
            #     if possible_valences is None:
            #         continue
            #     for possible_valence in possible_valences:
            #         oxidation_state_name = f'ox_{possible_valence}'
            #         if oxidation_state_name not in possible_oxidation_state_names:
            #             possible_oxidation_state_names.append(oxidation_state_name)
            #             possible_oxidation_state_valences.append(possible_valence)

            # # Create DataFrame with the collected oxidation state names and valences
            # data = {
            #     'oxidation_state': possible_oxidation_state_names,
            #     'valence': possible_oxidation_state_valences
            # }

            oxidation_states = np.arange(-9, 10)
            oxidation_states_names = [f'ox_{i}' for i in oxidation_states]
            data={
                'oxidation_state': oxidation_states_names,
                'value': oxidation_states
            }
            df = pd.DataFrame(data)
            df['name'] = df['oxidation_state']  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error creating oxidation state nodes: {e}")
            return None
        return df

class SpaceGroupNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.SPACE_GROUP.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Space Group nodes
        return get_node_schema(NodeTypes.SPACE_GROUP)

    def create_nodes(self, **kwargs):
        """
        Creates Space Group nodes if no file exists, otherwise loads them from a file.
        """

        # Generate space group numbers from 1 to 230
        try:
            space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
            space_groups_properties = [{"spg": int(space_group.split('_')[1])} for space_group in space_groups]

            # Create DataFrame with the space group properties
            df = pd.DataFrame(space_groups_properties)
            df['name'] = df['spg'].astype(str)  # Assign 'name' field as string version of 'spg'
        except Exception as e:
            logger.error(f"Error creating space group nodes: {e}")
            return None

        return df
    
class ChemEnvNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.CHEMENV.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for ChemEnv nodes
        return get_node_schema(NodeTypes.CHEMENV)

    def create_nodes(self,  **kwargs):
        """
        Creates ChemEnv nodes if no file exists, otherwise loads them from a file.
        """
        # Get the chemical environment names from a dictionary (mp_coord_encoding)
        try:
            chemenv_names = list(mp_coord_encoding.keys())
            chemenv_names_properties = []
            
            # Create a list of dictionaries with 'chemenv_name' and 'coordination'
            for chemenv_name in chemenv_names:
                coordination = int(chemenv_name.split(':')[1])
                chemenv_names_properties.append({
                    "chemenv_name": chemenv_name, 
                    "coordination": coordination
                })

            # Create DataFrame with the chemical environment names and coordination numbers
            df = pd.DataFrame(chemenv_names_properties)
            df['name'] = df['chemenv_name'].str.replace(':', '_')  # Replace ':' with '_' for 'name' field
        except Exception as e:
            logger.error(f"Error creating chemical environment nodes: {e}")
            return None

        return df
    
class WyckoffPositionsNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.SPG_WYCKOFF.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Wyckoff Positions nodes
        return get_node_schema(NodeTypes.SPG_WYCKOFF)

    def create_nodes(self, **kwargs):
        """
        Creates Wyckoff Position nodes if no file exists, otherwise loads them from a file.
        """
        logger.info(f"No node file found. Attempting to create {self.node_type} nodes")

        # Generate space group names from 1 to 230
        try:
            space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
            # Define Wyckoff letters
            wyckoff_letters = ['a', 'b', 'c', 'd', 'e', 'f']

            # Create a list of space group-Wyckoff position combinations
            spg_wyckoffs = [f"{spg}_{wyckoff_letter}" for wyckoff_letter in wyckoff_letters for spg in space_groups]

            # Create a list of dictionaries with 'spg_wyckoff'
            spg_wyckoff_properties = [{"spg_wyckoff": spg_wyckoff} for spg_wyckoff in spg_wyckoffs]

            # Create DataFrame with Wyckoff positions
            df = pd.DataFrame(spg_wyckoff_properties)
            df['name'] = df['spg_wyckoff']  # Assign 'name' field
        except Exception as e:
            logger.error(f"Error creating Wyckoff position nodes: {e}")
            return None

        return df

class MaterialLatticeNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.LATTICE.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Lattice nodes
        return get_node_schema(NodeTypes.LATTICE)

    def create_nodes(self,**kwargs):
        """
        Creates Lattice nodes if no file exists, otherwise loads them from a file.
        """

        # Retrieve material nodes with lattice properties
        try:
            df = pd.read_parquet(MATERIAL_PARQUET_FILE,columns=['material_id', 'lattice', 'a', 'b', 'c', 
                                                'alpha', 'beta', 'gamma', 'crystal_system', 'volume'])

            # Set the 'name' field as 'material_id'
            df['name'] = df['material_id']

        except Exception as e:
            logger.error(f"Error creating lattice nodes: {e}")
            return None

        return df

class MaterialSiteNodes(Nodes):
    def __init__(self, node_dir, output_format='pandas'):
        super().__init__(node_type=NodeTypes.SITE.value, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for Site nodes
        return get_node_schema(NodeTypes.SITE)

    def create_nodes(self, **kwargs):
        """
        Creates Site nodes if no file exists, otherwise loads them from a file.
        """

        # Retrieve material nodes with relevant site properties
        try:
            df = pd.read_parquet(MATERIAL_PARQUET_FILE,columns=['material_id', 'lattice', 'frac_coords', 'species'])
            
            all_species = []
            all_coords = []
            all_lattices = []
            all_ids = []
            
            # Iterate through each row of the DataFrame
            for irow, row in df.iterrows():
                if irow % 10000 == 0:
                    logger.info(f"Processing row {irow}")
                if row['species'] is None:
                    continue
                
                # Collect species, fractional coordinates, lattices, and material IDs
                for frac_coord, specie in zip(row['frac_coords'], row['species']):
                    all_species.append(specie)
                    all_coords.append(frac_coord)
                    all_lattices.append(row['lattice'])
                    all_ids.append(row['material_id'])

            # Create DataFrame for Site nodes
            df = pd.DataFrame({
                'species': all_species,
                'frac_coords': all_coords,
                'lattice': all_lattices,
                'material_id': all_ids
            })

            df['name'] = df['material_id']  # Assign 'name' field as 'material_id'
        except Exception as e:
            logger.error(f"Error creating site nodes: {e}")
            return None
        return df


class NodeManager:
    def __init__(self, node_dir, output_format='pandas'):
        """
        Initialize the NodesManager with the directory where nodes are stored.
        """
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.node_dir = node_dir
        os.makedirs(self.node_dir, exist_ok=True)
        self.file_type = 'parquet'
        self.get_existing_nodes()

    def get_existing_nodes(self):
        self.nodes =  set(self.list_nodes())
        return self.nodes

    def list_nodes(self):
        """
        List all node files available in the node directory.
        """
        node_files = [f for f in os.listdir(self.node_dir) if f.endswith(f'.{self.file_type}')]
        node_types = [os.path.splitext(f)[0] for f in node_files]  # Extract file names without extension
        logger.info(f"Found the following node types: {node_types}")
        return node_types

    def get_node(self, node_type, output_format='pandas'):
        """
        Load a node dataframe by its type (which corresponds to the filename without extension).
        """
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        filepath = os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if not os.path.exists(filepath):
            logger.error(f"No node file found for type: {node_type}")
            return None
        
        nodes=Nodes(node_type=node_type, node_dir=self.node_dir,output_format=output_format)

        return nodes
    
    def get_node_dataframe(self, node_type, columns=None, include_cols=True, output_format='pandas', **kwargs):
        """
        Return the node dataframe if it has already been loaded; otherwise, load it from file.
        """
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        return self.get_node(node_type, output_format=output_format).get_dataframe(columns=columns, include_cols=include_cols)

    def add_node(self, node_class):
        """
        Add a new node by providing a custom node class (must inherit from the base Node class).
        The node class must implement its own creation logic.
        """
        if not issubclass(node_class, Nodes):
            raise TypeError("The provided class must inherit from the Nodes class.")
        
        node = node_class(node_dir=self.node_dir)  # Initialize the node class
        node.get_dataframe()  # Get or create the node dataframe

        self.get_existing_nodes()

    def delete_node(self, node_type):
        """
        Delete a node type. This method will remove the parquet file and the node from the self.nodes set.
        """
        filepath = os.path.join(self.node_dir, f'{node_type}.{self.file_type}')
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                self.nodes.discard(node_type)  # Remove from the set of nodes
                logger.info(f"Deleted node of type {node_type} and removed it from the node set.")
            except Exception as e:
                logger.error(f"Error deleting node of type {node_type}: {e}")
        else:
            logger.warning(f"No node file found for type {node_type} to delete.")

    def convert_all_to_neo4j(self, save_dir):
        """
        Convert all Parquet node files in the node directory to Neo4j CSV format.
        """
        os.makedirs(save_dir, exist_ok=True)
        for node_type in self.nodes:
            logger.info(f"Converting {node_type} to Neo4j CSV format.")
            try:
                node = self.get_node(node_type)  # Load the node
                if node is not None:
                    node.to_neo4j(save_dir)  # Convert to Neo4j format
                    logger.info(f"Successfully converted {node_type} to Neo4j CSV.")
                else:
                    logger.warning(f"Skipping {node_type} as it could not be loaded.")
            except Exception as e:
                logger.error(f"Error converting {node_type} to Neo4j CSV: {e}")




if __name__ == "__main__":
    node_dir = os.path.join('data','raw','nodes')
    node=ElementNodes(node_dir=node_dir)
    print(node.get_property_names())

    nodes=SpaceGroupNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=MagneticStatesNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=MaterialNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=CrystalSystemNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=OxidationStatesNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=ChemEnvNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=WyckoffPositionsNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=MaterialLatticeNodes(node_dir=node_dir)
    print(nodes.get_property_names())

    nodes=MaterialSiteNodes(node_dir=node_dir)
    print(nodes.get_property_names())




    # node=Nodes(node_type='ELEMENT', node_dir=node_dir)

    # print(node.get_property_names())



    # df = node.get_nodes()
    # print(df.head())


    # node=MaterialNodes(node_dir=node_dir)
    # df = node.get_nodes()
    # print(df.head())

    manager=NodeManager(node_dir=node_dir)

    print(manager.nodes)

    df=manager.get_node_dataframe('ELEMENT')
    print(df.head())

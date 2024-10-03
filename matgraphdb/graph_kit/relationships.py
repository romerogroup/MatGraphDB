from glob import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from matgraphdb.graph_kit.metadata import get_relationship_schema
from matgraphdb.graph_kit.metadata import RelationshipTypes
from matgraphdb.graph_kit.nodes import NodeManager

logger = logging.getLogger(__name__)

class Relationships:
    """
    A class for managing and creating relationships between nodes in a graph database. This class handles 
    relationship creation, loading, validation, and exporting relationships into different formats, such as 
    Parquet and Neo4j CSV. It utilizes a NodeManager instance to handle node-related operations and can be 
    extended to implement custom relationship creation logic.

    Attributes
    ----------
    relationship_type : str
        A string representing the relationship type in the format 'start_node-relationship-end_node'.
    relationship_dir : str
        The directory where the relationships are stored.
    node_manager : NodeManager
        An instance of NodeManager responsible for handling node operations.
    output_format : str
        The format used for reading and writing data ('pandas' or 'pyarrow'). Default is 'pandas'.
    file_type : str
        The file format used for saving relationships. Currently, 'parquet' is used.
    filepath : str
        The full path of the file where the relationships are stored.
    schema : object
        The schema definition for the relationships, created by the `create_schema` method.
    """
    def __init__(self, relationship_type, relationship_dir, node_dir, output_format='pandas'):
        """
        Initializes the Relationships class with the given relationship type, directories, and output format.

        Parameters
        ----------
        relationship_type : str
            The type of relationship, formatted as 'start_node-relationship-end_node'.
        relationship_dir : str
            Directory where the relationship files will be stored.
        node_dir : str
            Directory where node information is stored, managed by the NodeManager.
        output_format : str, optional
            Format for reading and writing data. Options are 'pandas' or 'pyarrow' (default is 'pandas').

        Raises
        ------
        ValueError
            If `output_format` is not 'pandas' or 'pyarrow'.
        """
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.relationship_type = relationship_type
        self.relationship_dir = relationship_dir
        self.node_manager =  NodeManager(node_dir=node_dir)  # Store the NodeManager instance
        os.makedirs(self.relationship_dir, exist_ok=True)

        self.output_format = output_format
        self.file_type = 'parquet'
        self.filepath = os.path.join(self.relationship_dir, f'{self.relationship_type}.{self.file_type}')
        self.schema = self.create_schema()

        self.get_dataframe()

    def get_dataframe(self, columns=None, include_cols=True, from_scratch=False, remove_duplicates=True, **kwargs):
        """
        Loads or creates the relationship data based on the specified parameters.

        Parameters
        ----------
        columns : list, optional
            A list of columns to include or exclude from the DataFrame.
        include_cols : bool, optional
            Whether to include or exclude the specified columns (default is True).
        from_scratch : bool, optional
            If True, forces the creation of the relationship data from scratch (default is False).
        remove_duplicates : bool, optional
            Whether to remove duplicate relationships (default is True).

        Returns
        -------
        pd.DataFrame or pyarrow.Table
            The loaded or newly created relationship DataFrame.

        Raises
        ------
        ValueError
            If required fields ('start_node_id' and 'end_node_id') are missing in the relationship DataFrame.
        """

        start_node_type,connection_name,end_node_type=self.relationship_type.split('-')
        start_node_name=f'{start_node_type}-START_ID'
        end_node_name=f'{end_node_type}-END_ID'

       
        if os.path.exists(self.filepath) and not from_scratch:
            logger.info(f"Trying to load {self.relationship_type} relationships from {self.filepath}")
            df = self.load_dataframe(filepath=self.filepath, columns=columns, include_cols=include_cols, **kwargs)
            return df

        logger.info(f"No relationship file found. Attempting to create {self.relationship_type} relationships")
        df = self.create_relationships(**kwargs)  # Subclasses will define this

        # Ensure the 'start_node_id' and 'end_node_id' fields are present
        if start_node_name not in df.columns or end_node_name not in df.columns:
            raise ValueError(f"'{start_node_name}' and '{end_node_name}' fields must be defined for {self.relationship_type} relationships.")
        
        # If 'weight' is not in the dataframe, add it or if remove_duplicates is True, remove duplicates
        if 'weight' not in df.columns:
            if remove_duplicates:
                df=self.remove_duplicate_relationships(df)

        df['TYPE'] = self.relationship_type

        if columns:
            df = df[columns]

        if not self.schema:
            logger.error(f"No schema set for {self.relationship_type} relationships")
            return None

        self.save_dataframe(df, self.filepath)
        return df
    
    def get_property_names(self):
        """
        Retrieves and logs the names of properties (columns) in the relationship file.

        Returns
        -------
        list
            A list of property names in the relationship file.
        """
        properties = Relationships.get_column_names(self.filepath)
        for property in properties:
            logger.info(f"Property: {property}")
        return properties

    def create_relationships(self, **kwargs):
        """
        Abstract method for creating relationships. Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If this method is called from the base class instead of a subclass.
        """
        if self.__class__.__name__ != 'Relationships':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def create_schema(self, **kwargs):
        """
        Abstract method for creating the schema for relationships. Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If this method is called from the base class instead of a subclass.
        """
        if self.__class__.__name__ != 'Relationships':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def load_dataframe(self, filepath, columns=None, include_cols=True, **kwargs):
        """
        Loads a DataFrame from a parquet file.

        Parameters
        ----------
        filepath : str
            The path to the parquet file.
        columns : list, optional
            List of columns to include or exclude when loading the file.
        include_cols : bool, optional
            Whether to include or exclude the specified columns (default is True).

        Returns
        -------
        pd.DataFrame or pyarrow.Table
            The loaded DataFrame or table, depending on the output format.

        Raises
        ------
        Exception
            If an error occurs while loading the file.
        """
        try:
            if self.output_format == 'pandas':
                df = pd.read_parquet(filepath, columns=columns)
            elif self.output_format == 'pyarrow':
                df = pq.read_table(filepath, columns=columns)
            return df
        except Exception as e:
            logger.error(f"Error loading {self.relationship_type} relationships from {filepath}: {e}")
            return None

    def save_dataframe(self, df, filepath):
        """
        Saves a DataFrame to a parquet file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to save.
        filepath : str
            The path where the DataFrame will be saved.

        Raises
        ------
        Exception
            If an error occurs while saving the DataFrame to a parquet file.
        """
        try:
            parquet_table = pa.Table.from_pandas(df, self.schema)
            pq.write_table(parquet_table, filepath)
            logger.info(f"Finished saving {self.relationship_type} relationships to {filepath}")
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")

    def to_neo4j(self, save_dir):
        """
        Converts the relationship data to Neo4j-compatible CSV format.

        Parameters
        ----------
        save_dir : str
            The directory where the Neo4j CSV file will be saved.
        """
        logger.info(f"Converting relationship to Neo4j : {self.filepath}")

        relationship_type=os.path.basename(self.filepath).split('.')[0]
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.debug(f"Relationship type: {relationship_type}")

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

        neo4j_column_name_mapping['TYPE']=':LABEL'

        neo4j_column_name_mapping[f'{node_a_type}-START_ID']=f':START_ID({node_a_type}-ID)'
        neo4j_column_name_mapping[f'{node_b_type}-END_ID']=f'END_ID({node_a_type}-ID)'

        df=self.load_relationships(filepath=self.filepath)


        df.rename(columns=neo4j_column_name_mapping, inplace=True)

        os.makedirs(save_dir,exist_ok=True)

       
        save_file=os.path.join(save_dir,f'{relationship_type}.csv')

        logger.debug(f"Saving {relationship_type} relationship_path to {save_file}")

        df.to_csv(save_file, index=False)

        logger.info(f"Finished converting relationship to Neo4j : {relationship_type}")

    def validate_nodes(self):
        """
        Validate that the nodes used in the relationships exist in the node manager.
        """
        start_nodes = set(self.get_dataframe()['start_node'].unique())
        end_nodes = set(self.get_dataframe()['end_node'].unique())
        
        existing_nodes = self.node_manager.get_existing_nodes()
        missing_start_nodes = start_nodes - existing_nodes
        missing_end_nodes = end_nodes - existing_nodes

        if missing_start_nodes:
            logger.warning(f"Missing start nodes: {missing_start_nodes}")
        if missing_end_nodes:
            logger.warning(f"Missing end nodes: {missing_end_nodes}")

        return not missing_start_nodes and not missing_end_nodes
    
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
    

# Example subclass for specific relationship types
class MaterialSPGRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_SPG.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_SPG)

    def create_relationships(self, **kwargs):
        # The logic for creating material relationships
        try:
            relationship_type=RelationshipTypes.MATERIAL_SPG.value
            start_node_type,connection_name,end_node_type=relationship_type.split('-')

            # Example: Create relationships between materials from nodes in node_manager
            start_nodes_df = self.node_manager.get_node_dataframe(start_node_type, columns=['space_group'])
            end_nodes_df = self.node_manager.get_node_dataframe(end_node_type, columns=['name'])
            
            # Mapping name to index
            name_to_index_mapping_b = {int(name): index for index, name in end_nodes_df['name'].items()}

            # Creating dataframe
            df = start_nodes_df.copy()
            
            # Removing NaN values
            df = df.dropna()

            # Making current index a column and reindexing
            df = df.reset_index().rename(columns={'index': start_node_type+'-START_ID'})

            # Adding node b ID with the mapping
            df[end_node_type+'-END_ID'] = df['space_group'].map(name_to_index_mapping_b).astype(int)

            df.drop(columns=['space_group'], inplace=True)

            df['weight'] = 1.0
 
        except Exception as e:
            logger.error(f"Error creating material relationships: {e}")
            return None
        return df
    
class MaterialCrystalSystemRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material-crystal system relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.MATERIAL_CRYSTAL_SYSTEM.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for material and crystal system
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['crystal_system'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])

            # Mapping name to index for the crystal system nodes
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Creating relationships dataframe by copying node_a dataframe
            df = node_a_df.copy()

            # Converting the 'crystal_system' column to lowercase to standardize
            df['crystal_system'] = df['crystal_system'].str.lower()

            # Removing rows with missing 'crystal_system' values
            df = df.dropna()

            # Resetting the index and renaming it to follow the START_ID convention
            df = df.reset_index().rename(columns={'index': node_a_type + '-START_ID'})

            # Adding the END_ID for the crystal system nodes by mapping the 'crystal_system' to the index
            df[node_b_type + '-END_ID'] = df['crystal_system'].map(name_to_index_mapping_b)

            # Dropping the 'crystal_system' column as it's no longer needed
            df.drop(columns=['crystal_system'], inplace=True)

            df['weight'] = 1.0
        except Exception as e:
            logger.error(f"Error creating material relationships: {e}")
            return None


        return df

class MaterialLatticeRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_LATTICE.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material-lattice relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_LATTICE)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.MATERIAL_LATTICE.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for material and lattice
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])

            # Mapping name to index for the lattice nodes
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Creating relationships dataframe by copying node_a dataframe
            df = node_a_df.copy()

            # Removing rows with missing 'name' values
            df = df.dropna()

            # Resetting the index and renaming it to follow the START_ID convention
            df = df.reset_index().rename(columns={'index': node_a_type + '-START_ID'})

            # Adding the END_ID for the lattice nodes by mapping the 'name' to the index
            df[node_b_type + '-END_ID'] = df['name'].map(name_to_index_mapping_b).astype(int)

            # Dropping the 'name' column as it's no longer needed
            df.drop(columns=['name'], inplace=True)

            # Setting the relationship type and a default weight of 1.0
            df['weight'] = 1.0

        except Exception as e:
            logger.error(f"Error creating material lattice relationships: {e}")
            return None

        return df

class MaterialSiteRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_SITE.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material-site relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_SITE)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.MATERIAL_SITE.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for material and site
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])

            # Mapping name to index for the material nodes
            name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}

            # Creating relationships dataframe by copying node_b dataframe
            df = node_b_df.copy()

            # Removing rows with missing 'name' values
            df = df.dropna()

            # Resetting the index and renaming it to follow the START_ID convention
            df = df.reset_index().rename(columns={'index': node_a_type + '-START_ID'})

            # Adding the END_ID for the site nodes by mapping the 'name' to the material nodes index
            df[node_b_type + '-END_ID'] = df['name'].map(name_to_index_mapping_a)

            # Dropping the 'name' column as it's no longer needed
            df.drop(columns=['name'], inplace=True)

            df['weight'] = 1.0

        except Exception as e:
            logger.error(f"Error creating material site relationships: {e}")
            return None

        return df

class ElementOxidationStateRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_OXIDATION_STATE.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-oxidation state relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_OXIDATION_STATE)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_OXIDATION_STATE.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements and oxidation states
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name', 'experimental_oxidation_states'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe("MATERIAL", columns=['name', 'species', 'oxidation_states-possible_valences'])

            # Mapping names to indices for element and oxidation state nodes
            name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Connecting oxidation states to elements derived from material nodes
            oxidation_state_names = []
            element_names = []
            for _, row in node_material_df.iterrows():
                possible_valences = row['oxidation_states-possible_valences']
                elements = row['species']
                if possible_valences is None or elements is None:
                    continue
                for possible_valence, element in zip(possible_valences, elements):
                    oxidation_state_name = f'ox_{possible_valence}'
                    oxidation_state_names.append(oxidation_state_name)
                    element_names.append(element)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': element_names,
                f'{node_b_type}-END_ID': oxidation_state_names
            }
            df = pd.DataFrame(data)

            # Convert element names to indices and oxidation state names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)


        except Exception as e:
            logger.error(f"Error creating element oxidation state relationships: {e}")
            return None

        return df
    
class MaterialElementRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_ELEMENT.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material-element relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_ELEMENT)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.MATERIAL_ELEMENT.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for materials and elements
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name', 'species'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])

            # Mapping names to indices for materials and element nodes
            name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Connecting materials to elements derived from material nodes
            material_names = []
            element_names = []
            for _, row in node_a_df.iterrows():
                elements = row['species']
                material_name = row['name']
                if elements is None:
                    continue

                # Append the material name for each element in the species list
                material_names.extend([material_name] * len(elements))
                element_names.extend(elements)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': material_names,
                f'{node_b_type}-END_ID': element_names
            }
            df = pd.DataFrame(data)

            # Convert material names to indices and element names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)


        except Exception as e:
            logger.error(f"Error creating material-element relationships: {e}")
            return None

        return df

class MaterialChemEnvRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.MATERIAL_CHEMENV.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for material-chemenv relationships
        return get_relationship_schema(RelationshipTypes.MATERIAL_CHEMENV)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.MATERIAL_CHEMENV.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for materials and chemenv
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name', 'coordination_environments_multi_weight'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])

            # Mapping names to indices for materials and chemenv nodes
            name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Connecting materials to chemenv derived from material nodes
            material_names = []
            chemenv_names = []
            for _, row in node_a_df.iterrows():
                bond_connections = row['coordination_environments_multi_weight']
                material_name = row['name']
                if bond_connections is None:
                    continue
                
                # Extract chemenv name from bond connections
                for coord_env in bond_connections:
                    try:
                        chemenv_name = coord_env[0]['ce_symbol'].replace(':', '_')
                    except:
                        continue

                    material_names.append(material_name)
                    chemenv_names.append(chemenv_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': material_names,
                f'{node_b_type}-END_ID': chemenv_names
            }
            df = pd.DataFrame(data)

            # Convert material and chemenv names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)


        except Exception as e:
            logger.error(f"Error creating material-chemenv relationships: {e}")
            return None

        return df

class ElementChemEnvRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_CHEMENV.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-chemenv relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_CHEMENV)

    def create_relationships(self, columns=None, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_CHEMENV.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements and chemenv
            node_a_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_b_df = self.node_manager.get_node_dataframe(node_b_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'species', 'coordination_environments_multi_weight'])

            # Mapping names to indices for element and chemenv nodes
            name_to_index_mapping_a = {name: index for index, name in node_a_df['name'].items()}
            name_to_index_mapping_b = {name: index for index, name in node_b_df['name'].items()}

            # Connecting materials to chemenv derived from material nodes
            element_names = []
            chemenv_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['coordination_environments_multi_weight']
                elements = row['species']
                if bond_connections is None:
                    continue

                # Extract chemenv name and corresponding element
                for i, coord_env in enumerate(bond_connections):
                    try:
                        chemenv_name = coord_env[0]['ce_symbol'].replace(':', '_')
                    except:
                        continue
                    element_name = elements[i]

                    chemenv_names.append(chemenv_name)
                    element_names.append(element_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': element_names,
                f'{node_b_type}-END_ID': chemenv_names
            }
            df = pd.DataFrame(data)

            # Convert element names and chemenv names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_a)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_b)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)

        except Exception as e:
            logger.error(f"Error creating element-chemenv relationships: {e}")
            return None

        return df

class ElementGeometricElectricElementRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-geometric electric element relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements and material data
            element_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'species', 'geometric_electric_consistent_bond_connections'])

            # Mapping names to indices for element nodes
            name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}

            # Connecting materials to elements based on bond connections
            site_element_names = []
            neighbor_element_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['geometric_electric_consistent_bond_connections']
                elements = row['species']

                if bond_connections is None:
                    continue

                for i, site_connections in enumerate(bond_connections):
                    site_element_name = elements[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_element_name = elements[i_neighbor_element]

                        site_element_names.append(site_element_name)
                        neighbor_element_names.append(neighbor_element_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_element_names,
                f'{node_b_type}-END_ID': neighbor_element_names
            }
            df = pd.DataFrame(data)

            # Convert element names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

            # Removing NaN values
            df = df.dropna()


        except Exception as e:
            logger.error(f"Error creating element-geometric electric element relationships: {e}")
            return None

        return df

class ElementGeometricElementRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-geometric element relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_GEOMETRIC_CONNECTS_ELEMENT.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements and material data
            element_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'species', 'geometric_consistent_bond_connections'])

            # Mapping names to indices for element nodes
            name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}

            # Connecting materials to elements based on bond connections
            site_element_names = []
            neighbor_element_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['geometric_consistent_bond_connections']
                elements = row['species']

                if bond_connections is None:
                    continue

                for i, site_connections in enumerate(bond_connections):
                    site_element_name = elements[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_element_name = elements[i_neighbor_element]

                        site_element_names.append(site_element_name)
                        neighbor_element_names.append(neighbor_element_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_element_names,
                f'{node_b_type}-END_ID': neighbor_element_names
            }
            df = pd.DataFrame(data)

            # Convert element names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

            # Removing NaN values
            df = df.dropna()

        except Exception as e:
            logger.error(f"Error creating element-geometric element relationships: {e}")
            return None

        return df


class ElementElectricElementRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-electric element relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_ELECTRIC_CONNECTS_ELEMENT.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements and material data
            element_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'species', 'electric_consistent_bond_connections'])

            # Mapping names to indices for element nodes
            name_to_index_mapping_element = {name: index for index, name in element_df['name'].items()}

            # Connecting materials to elements based on electric bond connections
            site_element_names = []
            neighbor_element_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['electric_consistent_bond_connections']
                elements = row['species']

                if bond_connections is None:
                    continue

                for i, site_connections in enumerate(bond_connections):
                    site_element_name = elements[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_element_name = elements[i_neighbor_element]

                        site_element_names.append(site_element_name)
                        neighbor_element_names.append(neighbor_element_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_element_names,
                f'{node_b_type}-END_ID': neighbor_element_names
            }
            df = pd.DataFrame(data)

            # Convert element names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_element)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_element)

            # Removing NaN values
            df = df.dropna()

        except Exception as e:
            logger.error(f"Error creating element-electric element relationships: {e}")
            return None

        return df

class ChemEnvGeometricElectricChemEnvRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for chemenv-geometric electric chemenv relationships
        return get_relationship_schema(RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for chemenv and material data
            chemenv_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'coordination_environments_multi_weight', 'geometric_electric_consistent_bond_connections'])

            # Mapping names to indices for chemenv nodes
            name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}

            # Connecting materials to chemenv based on bond connections
            site_chemenv_names = []
            neighbor_chemenv_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['geometric_electric_consistent_bond_connections']
                chemenv_info = row['coordination_environments_multi_weight']

                if bond_connections is None or chemenv_info is None:
                    continue

                # Extract chemenv names from the coordination environments
                chemenv_names = []
                for coord_env in chemenv_info:
                    try:
                        chemenv_name = coord_env[0]['ce_symbol'].replace(':', '_')
                        chemenv_names.append(chemenv_name)
                    except:
                        continue

                # Creating connections between site chemenv and neighbor chemenv
                for i, site_connections in enumerate(bond_connections):
                    site_chemenv_name = chemenv_names[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_chemenv_name = chemenv_names[i_neighbor_element]

                        site_chemenv_names.append(site_chemenv_name)
                        neighbor_chemenv_names.append(neighbor_chemenv_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_chemenv_names,
                f'{node_b_type}-END_ID': neighbor_chemenv_names
            }
            df = pd.DataFrame(data)

            # Convert chemenv names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)


        except Exception as e:
            logger.error(f"Error creating chemenv-geometric electric chemenv relationships: {e}")
            return None

        return df


class ChemEnvGeometricChemEnvRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for chemenv-geometric chemenv relationships
        return get_relationship_schema(RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.CHEMENV_GEOMETRIC_CONNECTS_CHEMENV.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for chemenv and material data
            chemenv_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'coordination_environments_multi_weight', 'geometric_consistent_bond_connections'])

            # Mapping names to indices for chemenv nodes
            name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}

            # Connecting materials to chemenv based on bond connections
            site_chemenv_names = []
            neighbor_chemenv_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['geometric_consistent_bond_connections']
                chemenv_info = row['coordination_environments_multi_weight']

                if bond_connections is None or chemenv_info is None:
                    continue

                # Extract chemenv names from the coordination environments
                chemenv_names = []
                for coord_env in chemenv_info:
                    try:
                        chemenv_name = coord_env[0]['ce_symbol'].replace(':', '_')
                        chemenv_names.append(chemenv_name)
                    except:
                        continue

                # Creating connections between site chemenv and neighbor chemenv
                for i, site_connections in enumerate(bond_connections):
                    site_chemenv_name = chemenv_names[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_chemenv_name = chemenv_names[i_neighbor_element]

                        site_chemenv_names.append(site_chemenv_name)
                        neighbor_chemenv_names.append(neighbor_chemenv_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_chemenv_names,
                f'{node_b_type}-END_ID': neighbor_chemenv_names
            }
            df = pd.DataFrame(data)

            # Convert chemenv names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)

        except Exception as e:
            logger.error(f"Error creating chemenv-geometric chemenv relationships: {e}")
            return None

        return df

class ChemEnvElectricChemEnvRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for chemenv-electric chemenv relationships
        return get_relationship_schema(RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.CHEMENV_ELECTRIC_CONNECTS_CHEMENV.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for chemenv and material data
            chemenv_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name'])
            node_material_df = self.node_manager.get_node_dataframe('material', columns=['name', 'coordination_environments_multi_weight', 'electric_consistent_bond_connections'])

            # Mapping names to indices for chemenv nodes
            name_to_index_mapping_chemenv = {name: index for index, name in chemenv_df['name'].items()}

            # Connecting materials to chemenv based on electric bond connections
            site_chemenv_names = []
            neighbor_chemenv_names = []
            for _, row in node_material_df.iterrows():
                bond_connections = row['electric_consistent_bond_connections']
                chemenv_info = row['coordination_environments_multi_weight']

                if bond_connections is None or chemenv_info is None:
                    continue

                # Extract chemenv names from the coordination environments
                chemenv_names = []
                for coord_env in chemenv_info:
                    try:
                        chemenv_name = coord_env[0]['ce_symbol'].replace(':', '_')
                        chemenv_names.append(chemenv_name)
                    except:
                        continue

                # Creating connections between site chemenv and neighbor chemenv
                for i, site_connections in enumerate(bond_connections):
                    site_chemenv_name = chemenv_names[i]
                    for i_neighbor_element in site_connections:
                        i_neighbor_element = int(i_neighbor_element)
                        neighbor_chemenv_name = chemenv_names[i_neighbor_element]

                        site_chemenv_names.append(site_chemenv_name)
                        neighbor_chemenv_names.append(neighbor_chemenv_name)

            # Creating the relationships dataframe
            data = {
                f'{node_a_type}-START_ID': site_chemenv_names,
                f'{node_b_type}-END_ID': neighbor_chemenv_names
            }
            df = pd.DataFrame(data)

            # Convert chemenv names to indices
            df[f'{node_a_type}-START_ID'] = df[f'{node_a_type}-START_ID'].map(name_to_index_mapping_chemenv)
            df[f'{node_b_type}-END_ID'] = df[f'{node_b_type}-END_ID'].map(name_to_index_mapping_chemenv)

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)

        except Exception as e:
            logger.error(f"Error creating chemenv-electric chemenv relationships: {e}")
            return None

        return df
    
class ElementGroupPeriodRelationships(Relationships):
    def __init__(self, relationship_dir, node_dir, output_format='pandas'):
        super().__init__(relationship_type=RelationshipTypes.ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT.value, relationship_dir=relationship_dir, node_dir=node_dir, output_format=output_format)

    def create_schema(self):
        # Define and return the schema for element-group-period relationships
        return get_relationship_schema(RelationshipTypes.ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT)

    def create_relationships(self, columns=None, remove_duplicates=True, **kwargs):
        # Defining node types and relationship type
        try:
            relationship_type = RelationshipTypes.ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT.value
            node_a_type, connection_name, node_b_type = relationship_type.split('-')

            logger.info(f"Getting {relationship_type} relationships")

            # Loading nodes for elements
            element_df = self.node_manager.get_node_dataframe(node_a_type, columns=['name', 'atomic_number', 'extended_group', 'period', 'symbol'])

            # Mapping names to indices for elements
            name_to_index_mapping = {name: index for index, name in element_df['name'].items()}

            # Getting group-period edge index
            edge_index = get_group_period_edge_index(element_df)

            # Creating the relationships dataframe
            df = pd.DataFrame(edge_index, columns=[f'{node_a_type}-START_ID', f'{node_b_type}-END_ID'])

            # Removing NaN values and converting to int
            df = df.dropna().astype(int)


        except Exception as e:
            logger.error(f"Error creating element-group-period relationships: {e}")
            return None

        return df


class RelationshipManager:
    def __init__(self, relationship_dir, output_format='pandas'):
        """
        Initialize the RelationshipManager with the directory where relationships are stored.
        """
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.relationship_dir = relationship_dir
        os.makedirs(self.relationship_dir, exist_ok=True)

        self.file_type = 'parquet'
        self.get_existing_relationships()

    def get_existing_relationships(self):
        self.relationships = set(self.list_relationships())
        return self.relationships

    def list_relationships(self):
        """
        List all relationship files available in the relationship directory.
        """
        relationship_files = [f for f in os.listdir(self.relationship_dir) if f.endswith(f'.{self.file_type}')]
        relationship_types = [os.path.splitext(f)[0] for f in relationship_files]  # Extract file names without extension
        logger.info(f"Found the following relationship types: {relationship_types}")
        return relationship_types

    def get_relationship(self, relationship_type, output_format='pandas'):
        """
        Load a relationship dataframe by its type (which corresponds to the filename without extension).
        """
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        filepath = os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        
        if not os.path.exists(filepath):
            logger.error(f"No relationship file found for type: {relationship_type}")
            return None
        
        relationship = Relationships(relationship_type=relationship_type, relationship_dir=self.relationship_dir, output_format=output_format)
        return relationship
    
    def get_relationship_dataframe(self, relationship_type, columns=None, include_cols=True, output_format='pandas', **kwargs):
        """
        Return the relationship dataframe if it has already been loaded; otherwise, load it from file.
        """
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        return self.get_relationship(relationship_type, output_format=output_format).get_dataframe(columns=columns, include_cols=include_cols)

    def add_relationship(self, relationship_class):
        """
        Add a new relationship by providing a custom relationship class (must inherit from the base Relationships class).
        The relationship class must implement its own creation logic.
        """
        if not issubclass(relationship_class, Relationships):
            raise TypeError("The provided class must inherit from the Relationships class.")
        
        relationship = relationship_class(relationship_dir=self.relationship_dir)  # Initialize the relationship class
        relationship.get_dataframe()  # Get or create the relationship dataframe

        self.get_existing_relationships()

    def delete_relationship(self, relationship_type):
        """
        Delete a relationship type. This method will remove the parquet file and the relationship from the self.relationships set.
        """
        filepath = os.path.join(self.relationship_dir, f'{relationship_type}.{self.file_type}')
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                self.relationships.discard(relationship_type)  # Remove from the set of relationships
                logger.info(f"Deleted relationship of type {relationship_type} and removed it from the relationship set.")
            except Exception as e:
                logger.error(f"Error deleting relationship of type {relationship_type}: {e}")
        else:
            logger.warning(f"No relationship file found for type {relationship_type} to delete.")

    def convert_all_to_neo4j(self, save_dir):
        """
        Convert all Parquet relationship files in the relationship directory to Neo4j CSV format.
        """
        os.makedirs(save_dir, exist_ok=True)
        for relationship_type in self.relationships:
            logger.info(f"Converting {relationship_type} to Neo4j CSV format.")
            try:
                relationship = self.get_relationship(relationship_type)  # Load the relationship
                if relationship is not None:
                    relationship.to_neo4j(save_dir)  # Convert to Neo4j format
                    logger.info(f"Successfully converted {relationship_type} to Neo4j CSV.")
                else:
                    logger.warning(f"Skipping {relationship_type} as it could not be loaded.")
            except Exception as e:
                logger.error(f"Error converting {relationship_type} to Neo4j CSV: {e}")


if __name__ == "__main__":

    node_dir = os.path.join('data','raw','nodes')
    relationship_dir = os.path.join('data','raw','relationships')

    # relationships=ElementOxidationStateRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    # print(relationships.get_dataframe().head())
    # print(relationships.get_property_names())

    relationships=ElementGroupPeriodRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=ElementGeometricElectricElementRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=ElementElectricElementRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=ElementGeometricElementRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=ChemEnvGeometricElectricChemEnvRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=ChemEnvGeometricChemEnvRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships = ChemEnvElectricChemEnvRelationships(node_dir=node_dir, relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())



    relationships= MaterialChemEnvRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships = MaterialElementRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships = MaterialLatticeRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=MaterialSiteRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=MaterialCrystalSystemRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())

    relationships=MaterialSPGRelationships(node_dir=node_dir,relationship_dir=relationship_dir)
    print(relationships.get_dataframe().head())
    print(relationships.get_property_names())





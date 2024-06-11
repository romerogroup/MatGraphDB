import json
import re
import itertools
from functools import partial
from multiprocessing import Pool

import pandas as pd
import pymatgen.core as pmat
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import XRDPowderPattern
from matminer.featurizers.composition import ElementFraction

from matgraphdb.graph.node_types import *
from matgraphdb.utils import  GLOBAL_PROP_FILE, RELATIONSHIP_DIR,NODE_DIR, N_CORES, LOGGER, ENCODING_DIR, timeit
from matgraphdb.utils.periodic_table import atomic_symbols_map
# from matgraphdb.utils.math_utils import cosine_similarity
from matgraphdb.utils.general import chunk_list
############################################################
# Below is for is for creating relationships between nodes
############################################################

def extract_id_column_headers(df):
    """
    Extracts the ID column headers from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame from which to extract the ID column headers.

    Returns:
        str: The first ID column header found in the DataFrame.
    """
    id_columns = []
    for column in df.columns:
        if re.search(r':ID\(.+?\)', column):
            match = re.search(r':ID\((.+?)\)', column)
            id_columns.append(match.group(1))
    return id_columns[0]

def create_bonding_task(material_file, node_type, bonding_method='geometric_electric'):
    """
    Create bonding task based on the given material file, node type, and bonding method.

    Args:
        material_file (str): The path to the material file.
        node_type (str): The type of node to create for each site.
        bonding_method (str, optional): The method used for bonding. Defaults to 'geometric_electric'.

    Returns:
        list: A list of tuples representing the material connections.

    Raises:
        Exception: If there is an error processing the file.

    """
    material_connections=[]
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)

    with open(GLOBAL_PROP_FILE) as f:
        db_gp = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]

    try:
        
        if bonding_method == 'geometric_electric':
            coord_connections = db['geometric_electric_consistent_bond_connections']
        elif bonding_method == 'geometric':
            coord_connections = db['geometric_consistent_bond_connections']
        elif bonding_method == 'electric':
            coord_connections = db['electric_consistent_bond_connections']

        if node_type == 'chemenv' or node_type == 'chemenvElement':
            coord_env_names= [coord_env[0]['ce_symbol'] for coord_env in db['coordination_environments_multi_weight']]
        element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_connections in enumerate(coord_connections):
            
            site_element_name = element_names[i_site]

            if node_type == 'chemenv':
                site_coord_env_name=coord_env_names[i_site]
                site_node_id=CHEMENV_NAMES_ID_MAP[site_coord_env_name]
            elif node_type=='element':
                site_node_id=ELEMENTS_ID_MAP[site_element_name]
            elif node_type=='chemenvElement':
                site_coord_env_name=coord_env_names[i_site]
                site_node_id=CHEMENV_ELEMENT_NAMES_ID_MAP[site_element_name+'_'+site_coord_env_name]

            # Iterate over each neighbor of the site
            for i_neighbor in site_connections:

                neighbor_element_name = element_names[i_neighbor]

                if node_type == 'chemenv':
                    neighbor_coord_env_name=coord_env_names[i_neighbor]
                    neighbor_node_id=CHEMENV_NAMES_ID_MAP[neighbor_coord_env_name]
                elif node_type=='element':
                    neighbor_node_id=ELEMENTS_ID_MAP[neighbor_element_name]
                elif node_type=='chemenvElement':
                    neighbor_coord_env_name=coord_env_names[i_neighbor]
                    neighbor_node_id=CHEMENV_ELEMENT_NAMES_ID_MAP[neighbor_element_name+'_'+neighbor_coord_env_name]

                data = (site_node_id,neighbor_node_id)

                material_connections.append(data)
    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
    return material_connections

def create_chemenv_element_task(material_file):
    """
    Create chemical environment element task.

    This function takes a material file as input and extracts the coordination environments,
    connections, and site element names from the material data. It then iterates over each site
    and its coordination environment, and creates a list of tuples containing the chemical
    environment ID and element ID for each site.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples containing the chemical environment ID and element ID for each site.
    """
    
    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    with open(GLOBAL_PROP_FILE) as f:
        db_gp = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]

    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'] for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]

            site_chemenv_id=CHEMENV_NAMES_ID_MAP[site_coord_env_name]
            site_element_id=ELEMENTS_ID_MAP[site_element_name]

            data = (site_chemenv_id,site_element_id)
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_element_task(material_file):
    """
    Create material-element relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples representing the material-element relationships.
              Each tuple contains the material project ID and the element ID.

    Raises:
        Exception: If there is an error processing the material file.
    """

    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0].replace('-','_')

    try:

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_element_name = site_element_names[i_site]
            site_element_id=ELEMENTS_ID_MAP[site_element_name]

            data = (mpid,site_element_id)
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_chemenv_task(material_file):
    """
    Create material-chemenv task.

    This function takes a material file as input and extracts coordination environments,
    connections, and site element names from the material data. It then iterates over each
    site and its coordination environment to create a list of material connections.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of material connections, where each connection is represented as a tuple
        containing the material project ID and the coordination environment ID.

    """
    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0].replace('-','_')

    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'] for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_chemenv_id=CHEMENV_NAMES_ID_MAP[site_coord_env_name]

            data = (mpid,site_chemenv_id)
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_chemenvElement_task(material_file):
    """
    Create material-chemenvElement relationship task.

    This function takes a material file as input and extracts the coordination environments,
    connections, and site element names from the material data. It then iterates over each site
    and its coordination environment, and creates a relationship between the material and the
    chemenvElement.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples representing the material-chemenvElement relationships.

    Raises:
        Exception: If there is an error processing the material file.

    """
    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0].replace('-','_')

    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'] for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]
            site_chemenv_element_name=site_element_name+'_'+site_coord_env_name

            site_chemenvElement_id=CHEMENV_ELEMENT_NAMES_ID_MAP[site_chemenv_element_name]


            data = (mpid,site_chemenvElement_id)
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_spg_task(material_file):
    """
    Create material-spg relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples representing the material-spg relationships.
              Each tuple contains the material project ID and the spg ID.

    Raises:
        Exception: If there is an error processing the material file.
    """

    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0].replace('-','_')

    try:
        spg_name = db['symmetry']['number']
        spg_name = 'spg_' + str(spg_name)
        spg_id=SPG_MAP[spg_name]

        data = (mpid,spg_id)
        material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_crystal_system_task(material_file):
    """
    Create material-crystal_system relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples representing the material-crystal_system relationships.
              Each tuple contains the material project ID and the crystal_system ID.

    Raises:
        Exception: If there is an error processing the material file.
    """

    material_connections=[]

    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0].replace('-','_')

    try:
        crystal_system_name = db['symmetry']['crystal_system'].lower()
        crystal_system_id=CRYSTAL_SYSTEMS_ID_MAP[crystal_system_name]

        data = (mpid,crystal_system_id)
        material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_oxi_state_element_task(material_file):
    """
    Creates a list of tuples representing the connections between site elements and their oxidation states.

    Args:
        material_file (str): The path to the material file.

    Returns:
        list: A list of tuples representing the connections between site elements and their oxidation states.
    """

    material_connections=[]
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)

        structure=pmat.Structure.from_dict(db['structure'])
        
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]

    try:

        site_element_names = [x['label'] for x in db['structure']['sites']]
        oxi_states = structure.composition.oxi_state_guesses()[0]
        
        # Iterate over each site and its coordination environment
        for i_site, element_name in enumerate(site_element_names):
            oxi_state=oxi_states[0][element_name]

            site_element_id=ELEMENTS_ID_MAP[element_name]
            oxi_id= OXIDATION_STATES_ID_MAP[oxi_state]

            data = (site_element_id,oxi_id)
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_relationships(node_a_csv,node_b_csv, mp_task, connection_name='CONNECTS', filepath=None):
    """
    Create relationships between nodes based on the provided CSV files.

    Args:
        node_a_csv (str): Path to the CSV file containing the first set of nodes.
        node_b_csv (str): Path to the CSV file containing the second set of nodes.
        mp_task (function): A function that defines the task to be performed on each material.
        connection_name (str, optional): The name of the relationship between the nodes. Defaults to 'CONNECTS'.
        filepath (str, optional): Path to save the resulting CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the relationships between the nodes.

    """
    df_a=pd.read_csv(node_a_csv)
    df_b=pd.read_csv(node_b_csv)
    node_a_id_space = extract_id_column_headers(df_a)
    node_b_id_space = extract_id_column_headers(df_b)

    node_dict={
            f':START_ID({node_a_id_space})':[],
            f':END_ID({node_b_id_space})':[],
            f':TYPE':[],
    }

    with Pool(N_CORES) as p:
        materials=p.map(mp_task,MATERIAL_FILES[:])

    properties_names=None
    # Get the properties names of relationships if any
    if len(materials[0][0]) != 2:
        properties_names=[]
        for i,property in enumerate(materials[0]):
            if i>1:
                properties_names.append(property[0])
        for name in properties_names:
            node_dict.update({name:[] for name in properties_names} )

    # Iterate over the materials and create the relationships
    for imat,material in enumerate(materials):

        # Iterate over the connections in the material
        for i,connection in enumerate(material):
            
            if node_a_id_space == 'material-ID':
                node_dict[f':START_ID({node_a_id_space})'].append(imat)
                node_dict[f':END_ID({node_b_id_space})'].append(connection[1])
            elif node_b_id_space == 'material-ID':
                node_dict[f':START_ID({node_a_id_space})'].append(connection[0])
                node_dict[f':END_ID({node_b_id_space})'].append(imat)
            else:
                node_dict[f':START_ID({node_a_id_space})'].append(connection[0])
                node_dict[f':END_ID({node_b_id_space})'].append(connection[1])
            node_dict[f':TYPE'].append(connection_name)

            if properties_names:
                for iproperty,name in enumerate(properties_names):
                    node_dict[name].append(connection[1+iproperty+1])

    df=pd.DataFrame(node_dict)

    # Create a column with sorted tuples of START_ID and END_ID
    df['id_tuple'] = df.apply(lambda x: tuple(sorted([x[f':START_ID({node_a_id_space})'], x[f':END_ID({node_b_id_space})']])), axis=1)

    # Group by the sorted tuple and count occurrences
    grouped = df.groupby('id_tuple')
    weights = grouped.size().reset_index(name='weight:float')
    
    # Drop duplicates based on the id_tuple
    df_weighted = df.drop_duplicates(subset='id_tuple')

    # Merge with weights
    df_weighted = pd.merge(df_weighted, weights, on='id_tuple', how='left')

    # Drop the id_tuple column
    df_weighted = df_weighted.drop(columns='id_tuple')

    if filepath is not None:
        df_weighted.to_csv(filepath, index=False)

    return df_weighted


############################################################
# Below is for similarity between materials
############################################################
def similarity_task(material_combs, features):
    """
    Calculate the similarity between pairs of materials based on their features.

    Args:
        material_combs (list): A list of tuples representing pairs of material IDs.
        features (pandas.DataFrame): A DataFrame containing the features of the materials.

    Returns:
        list: A list of tuples containing the material IDs, relationship type, and similarity score for each pair of materials.
    """

    n_materials_combs = len(material_combs)
    material_combs_values = [None] * n_materials_combs

    for i, material_comb in enumerate(material_combs):
        mat_id_1, mat_id_2 = material_comb

        row_1 = features.iloc[mat_id_1].values
        row_2 = features.iloc[mat_id_2].values
        similarity = cosine_similarity(a=row_1, b=row_2)

        material_comb_values = (mat_id_1, mat_id_2, 'RELATIONSHP', similarity)
        material_combs_values[i] = material_comb_values

    return material_combs_values

def megnet_lookup_task(material_combs, features):
    """
    Perform a lookup task using the MEGNet model to calculate the similarity between pairs of materials.

    Args:
        material_combs (list): A list of tuples representing pairs of material IDs.
        features (pandas.DataFrame): A DataFrame containing the features of the materials.

    Returns:
        list: A list of tuples containing the material IDs, relationship type, and similarity score for each pair of materials.
    """

    n_materials_combs = len(material_combs)
    material_combs_values = [None] * n_materials_combs

    for i, material_comb in enumerate(material_combs):
        mat_id_1, mat_id_2 = material_comb

        row_1 = features.iloc[mat_id_1].values
        row_2 = features.iloc[mat_id_2].values

        similarity = cosine_similarity(a=row_1, b=row_2)

        material_comb_values = (mat_id_1, mat_id_2, 'RELATIONSHP', similarity)
        material_combs_values[i] = material_comb_values

    return material_combs_values

def get_structure_composition_task(material_file):
    """
    Load material data from a file and extract the pymatgen Structure and Composition objects.

    Args:
        material_file (str): The path to the material data file.

    Returns:
        tuple: A tuple containing the Structure and Composition objects extracted from the file.
    """
    with open(material_file) as f:
        db = json.load(f)
        struct = pmat.Structure.from_dict(db['structure'])
        composition = struct.composition
    return struct, composition

@timeit
def create_material_material_relationship(material_file_csv, mp_task, similarity_task, features, chunk_size=1000, filepath=None):
    """
    Create relationships between materials based on similarity.

    Args:
        material_file_csv (str): Path to the CSV file containing material information.
        mp_task (function): Function to retrieve structure and composition information for a material.
        similarity_task (function): Function to calculate similarity between two materials.
        features (pandas.DataFrame): DataFrame containing features for each material.
        chunk_size (int, optional): Number of material combinations to process in each chunk. Defaults to 1000.
        filepath (str, optional): Path to save the resulting CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the material relationships and their similarity scores.
    """
    
    df = pd.read_csv(material_file_csv)[['materialsId:ID(materialsId-ID)', 'name']]
    material_ids = df['materialsId:ID(materialsId-ID)'].values[:]
    node_id_space = 'materialsId-ID'

    material_id_combs = tuple(itertools.combinations_with_replacement(material_ids, r=2))
    material_id_combs_chunks = chunk_list(material_id_combs, chunk_size)

    # Get the structures and compositions for each material
    with Pool(N_CORES) as p:
        structure_composition_tuples = p.map(mp_task, MATERIAL_FILES[:])

    structures = []
    compositions = []
    for structure_composition_tuple in structure_composition_tuples:
        structure, composition = structure_composition_tuple
        structures.append(structure)
        compositions.append(composition)

    features = features
    with Pool(N_CORES) as p:
        material_combs_chunks_values = p.map(partial(similarity_task, features=features), material_id_combs_chunks)

    material_combs_values = []
    for material_combs_chunk_values in material_combs_chunks_values:
        material_combs_values.extend(material_combs_chunk_values)

    df = pd.DataFrame(material_combs_values,
                      columns=[f':START_ID({node_id_space})', f':END_ID({node_id_space})', f':TYPE', 'similarity'])

    if filepath is not None:
        df.to_csv(filepath, index=False)

    return df



def main():
    """
    This function is the entry point of the script and is responsible for creating relationships between different nodes.
    It calls the `create_relationships` function multiple times with different parameters to create various types of relationships.
    """
    save_path=os.path.join(RELATIONSHIP_DIR)
    print('Save_path : ', save_path)
    os.makedirs(save_path,exist_ok=True)
    print('Creating Relationship...')

    # # # ##########################################################################################################################
    # # # # Element - Element Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'elements.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='element', bonding_method='geometric_electric'), 
    #                      connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'element_element_geometric-electric.csv'))
    
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'elements.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='element',bonding_method='geometric'), 
    #                      connection_name='GEOMETRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'element_element_geometric.csv'))
    
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'elements.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='element',bonding_method='electric'), 
    #                      connection_name='ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'element_element_electric.csv'))
    
    # # ##########################################################################################################################
    # # # # Chemenv - Chemenv Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenv',bonding_method='geometric_electric'), 
    #                      connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'chemenv_chemenv_geometric-electric.csv'))
    
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenv',bonding_method='geometric'), 
    #                      connection_name='GEOMETRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'chemenv_chemenv_geometric.csv'))
    

    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenv',bonding_method='electric'), 
    #                      connection_name='ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,f'chemenv_chemenv_electric.csv'))

    # # ##########################################################################################################################
    # # # # ChemenvElement - ChemenvElement Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenvElement',bonding_method='geometric_electric'),
    #                      connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,'chemenvElement_chemenvElement_geometric-electric.csv'))
    
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenvElement',bonding_method='geometric'),
    #                      connection_name='GEOMETRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,'chemenvElement_chemenvElement_geometric.csv'))
    
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=partial(create_bonding_task,node_type='chemenvElement',bonding_method='electric'),
    #                      connection_name='ELECTRIC_CONNECTS',
    #                      filepath=os.path.join(save_path,'chemenvElement_chemenvElement_electric.csv'))

    # # # ##########################################################################################################################
    # # # # # Chemenv - Element Connections

    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=create_chemenv_element_task,
    #                      connection_name='CAN_OCCUR',
    #                      filepath=os.path.join(save_path,f'chemenv_elements.csv'))
    
    ###################################################################################################
    # Material Relationships
    #######################################################################################
    
    # # Material - Element Connections

    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=create_material_element_task,
    #                      connection_name='COMPOSED_OF',
    #                      filepath=os.path.join(save_path,f'materials_elements.csv'))
    

    # # # # Material - Chemenv Connections

    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
    #                      mp_task=create_material_chemenv_task,
    #                      connection_name='COMPOSED_OF',
    #                      filepath=os.path.join(save_path,f'materials_chemenv.csv'))
    
    # # # # # Material - ChemenvElement Connections

    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=create_material_chemenvElement_task,
    #                      connection_name='COMPOSED_OF',
    #                      filepath=os.path.join(save_path,f'materials_chemenvElement.csv'))
    
    # # # # Material - spg Connections

    create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'spg.csv'), 
                         mp_task=create_material_spg_task,
                         connection_name='HAS_SPACE_GROUP_SYMMETRY',
                         filepath=os.path.join(save_path,f'materials_spg.csv'))

    # # # # Material - crystal_system Connections

    create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'crystal_systems.csv'), 
                         mp_task=create_material_crystal_system_task,
                         connection_name='HAS_CRYSTAL_SYSTEM',
                         filepath=os.path.join(save_path,f'materials_crystal_system.csv'))





















    # # ChemenvElement - ChemenvElement Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=create_chemenvElement_chemenvElement_task, 
    #                      filepath=os.path.join(save_path,'chemenvElement_chemenvElement.csv'))

    # # Material - ChemenvElement Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_element_names.csv'), 
    #                      mp_task=create_chemenv_chemenv_task, 
    #                      filepath=os.path.join(save_path,'materials_chemenvElement.csv'))

    # Bonding Relationships
    # oxidation_states - Magnetic States Connections
    # create_bonding_relationships(node_a_csv=os.path.join(NODE_DIR,'oxidation_states.csv'),
    #                         node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                         mp_task=create_oxi_state_element_task, 
    #                         filepath=os.path.join(save_path,'oxiStates_elements.csv'))



    # # Material - Element Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
    #                      mp_task=, 
    #                      filepath=os.path.join(save_path,'materials_elements.csv'))
    
    # # Material - Chemenv Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                      node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
    #                      mp_task=, 
    #                      filepath=os.path.join(save_path,'materials_chemenv.csv'))
    
    # # Material - Crystal System Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                         node_b_csv=os.path.join(NODE_DIR,'crystal_systems.csv'), 
    #                         mp_task=, 
    #                         filepath=os.path.join(save_path,'materials_crystal_systems.csv'))
    
    # # Material - Magnetic States Connections
    # create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                         node_b_csv=os.path.join(NODE_DIR,'magnetic_states.csv'), 
    #                         mp_task=, 
    #                         filepath=os.path.join(save_path,'materials_magnetic_states.csv'))
    

    











    # Below is for similarity between materials
    # Note for 10647 materials this took 1457.1755.0496 seconds to complete. Max memory used 18.5 Gb
    # Reset the index
    # df=pd.read_csv(os.path.join(ENCODING_DIR,'MEGNet-MP-2018.6.1-Eform.csv'),index_col=0)
    # df.reset_index(drop=True, inplace=True)

    # create_material_material_relationship(material_file_csv=os.path.join(NODE_DIR,'materials.csv'),
    #                                       mp_task=get_structure_composition_task,
    #                                       similarity_task=megnet_lookup_task,
    #                                       features=df,
    #                                       chunk_size=500,
    #                                       filepath=os.path.join(save_path,'material-material_MEGNet-MP-2018.6.1-Eform-similarity.csv')
    #                                       )
    # print('Finished creating nodes')


if __name__ == '__main__':
    main()
import json
import os
import re
import itertools
from functools import partial
from multiprocessing import Pool

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.utils import  GLOBAL_PROP_FILE, N_CORES, LOGGER, timeit
from matgraphdb.utils.math_utils import cosine_similarity
from matgraphdb.utils.general import chunk_list
from matgraphdb.data.manager import DBManager

# TODO: Need a better way to handle the creation of relationship between nodes

def get_name_id_map(df):
    """
    Get a dictionary mapping material names to their IDs.

    Args:
        df (pandas.DataFrame): The DataFrame containing the material names and IDs.

    Returns:
        dict: A dictionary mapping material names to their IDs.
    """
    name_id_map = {}
    column_names = df.columns
    id_column_name=column_names[0]
    for index, row in df.iterrows():
        name_id_map[row['name:string']] = row[id_column_name]
    return name_id_map

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

def create_bonding_task(material_file, node_a_name, node_b_name, node_a_id_map, node_b_id_map, bonding_method='geometric_electric'):
    """
    Create bonding task based on the given material file, node type, and bonding method.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.
        bonding_method (str, optional): The method used for bonding. Defaults to 'geometric_electric'.

    Returns:
        list: A list of tuples representing the material connections.

    Raises:
        Exception: If there is an error processing the file.

    """
    
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)

    with open(GLOBAL_PROP_FILE) as f:
        db_gp = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]
    node_type=node_a_name
    id_maps = {'element': node_a_id_map if node_a_name == 'element' else node_b_id_map,
               'chemenv': node_a_id_map if node_a_name == 'chemenv' else node_b_id_map,
               'chemenv_element': node_a_id_map if node_a_name == 'chemenv_element' else node_b_id_map}
    id_map=id_maps[node_type]
    method_map = {
        'geometric_electric': 'geometric_electric_consistent_bond_connections',
        'geometric': 'geometric_consistent_bond_connections',
        'electric': 'electric_consistent_bond_connections'
    }
    material_connections=[]
    try:
        element_names = [x['label'] for x in db['structure']['sites']]
        coord_env_names=None
        if node_a_name=='chemenv' or node_a_name=='chemenv_element' or node_b_name=='chemenv' or node_b_name=='chemenv_element':
            coord_env_names = [coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]
        
        names_map = {'element': element_names,'chemenv': coord_env_names}

        
        coord_connections = db[method_map[bonding_method]]
        for i_site, site_connections in enumerate(coord_connections):
            if node_type=='chemenv_element':
                element_name = names_map['element'][i_site]
                chemenv_name = names_map['chemenv'][i_site]
                site_name = element_name+'_'+chemenv_name
            else:
                site_name = names_map[node_type][i_site]
            site_node_id = id_map[site_name]

            for i_neighbor in site_connections:
                if node_type=='chemenv_element':
                    element_name = names_map['element'][i_neighbor]
                    chemenv_name = names_map['chemenv'][i_neighbor]
                    neighbor_name = element_name+'_'+chemenv_name
                else:
                    neighbor_name = names_map[node_type][i_neighbor]
                neighbor_node_id = id_map[neighbor_name]

                material_connections.append((site_node_id, neighbor_node_id))

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
    return material_connections

def create_chemenv_element_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create chemical environment element task.

    This function takes a material file as input and extracts the coordination environments,
    connections, and site element names from the material data. It then iterates over each site
    and its coordination environment, and creates a list of tuples containing the chemical
    environment ID and element ID for each site.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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
    
    # Map names to ID maps
    chemenv_name_id_map = node_a_id_map if node_a_name == 'chemenv' else node_b_id_map
    element_name_id_map = node_a_id_map if node_a_name == 'element' else node_b_id_map

    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]

            site_chemenv_id=chemenv_name_id_map[site_coord_env_name]
            site_element_id=element_name_id_map[site_element_name]

            if  node_a_name == 'element':
                data = (site_element_id,site_chemenv_id)
            else:
                data = (site_chemenv_id,site_element_id)
                
            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_element_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create material-element relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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
    
    # Map names to ID maps
    materials_id_map = node_a_id_map if node_a_name == 'material' else node_b_id_map
    element_name_id_map = node_a_id_map if node_a_name == 'element' else node_b_id_map

    material_id=materials_id_map[mpid]
    try:

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_element_name = site_element_names[i_site]
            site_element_id=element_name_id_map[site_element_name]

            if  node_a_name == 'element':
                data = (site_element_id,material_id)
            else:
                data = (material_id,site_element_id)

            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_chemenv_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create material-chemenv task.

    This function takes a material file as input and extracts coordination environments,
    connections, and site element names from the material data. It then iterates over each
    site and its coordination environment to create a list of material connections.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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

    # Map names to ID maps
    materials_id_map = node_a_id_map if node_a_name == 'material' else node_b_id_map
    chemenv_name_id_map = node_a_id_map if node_a_name == 'chemenv' else node_b_id_map

    material_id=materials_id_map[mpid]
    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_chemenv_id=chemenv_name_id_map[site_coord_env_name]
            
            if  node_a_name == 'chemenv':
                data = (site_chemenv_id,material_id)
            else:
                data = (material_id,site_chemenv_id)

            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_chemenvElement_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create material-chemenvElement relationship task.

    This function takes a material file as input and extracts the coordination environments,
    connections, and site element names from the material data. It then iterates over each site
    and its coordination environment, and creates a relationship between the material and the
    chemenvElement.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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

    # Map names to ID maps
    materials_id_map = node_a_id_map if node_a_name == 'material' else node_b_id_map
    chemenv_element_name_id_map = node_a_id_map if node_a_name == 'chemenv_element' else node_b_id_map
    
    material_id=materials_id_map[mpid]
    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]
            site_chemenv_element_name=site_element_name+'_'+site_coord_env_name

            site_chemenvElement_id=chemenv_element_name_id_map[site_chemenv_element_name]

            if  node_a_name == 'chemenv_element':
                data = (site_chemenvElement_id,material_id)
            else:
                data = (material_id,site_chemenvElement_id)

            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_spg_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create material-spg relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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

    # Map names to ID maps
    materials_id_map = node_a_id_map if node_a_name == 'material' else node_b_id_map
    spg_id_map = node_a_id_map if node_a_name == 'spg' else node_b_id_map

    material_id=materials_id_map[mpid]
    try:
        spg_name = db['symmetry']['number']
        spg_name = 'spg_' + str(spg_name)
        spg_id=spg_id_map[spg_name]

        if  node_a_name == 'spg_id':
            data = (spg_id,material_id)
        else:
            data = (material_id,spg_id)
 
        material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_material_crystal_system_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Create material-crystal_system relationships based on the given material file.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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

    # Map names to ID maps
    materials_id_map = node_a_id_map if node_a_name == 'material' else node_b_id_map
    crystal_system_id_map = node_a_id_map if node_a_name == 'crystal_system' else node_b_id_map


    material_id=materials_id_map[mpid]
    try:
        crystal_system_name = db['symmetry']['crystal_system'].lower()
        crystal_system_id=crystal_system_id_map[crystal_system_name]

        if  node_a_name == 'crystal_system':
            data = (crystal_system_id,material_id)
        else:
            data = (material_id,crystal_system_id)

        material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections

def create_oxi_state_element_task(material_file,node_a_name,node_b_name,node_a_id_map,node_b_id_map):
    """
    Creates a list of tuples representing the connections between site elements and their oxidation states.

    Args:
        material_file (str): The path to the material file.
        node_a_name (str): The name of the first node in the relationship.
        node_b_name (str): The name of the second node in the relationship.
        node_a_id_map (dict): A dictionary mapping node names to IDs.
        node_b_id_map (dict): A dictionary mapping node names to IDs.

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

    element_id_map = node_a_id_map if node_a_name == 'element' else node_b_id_map
    oxidation_state_id_map = node_a_id_map if node_a_name == 'oxidation_state' else node_b_id_map

    try:

        site_element_names = [x['label'] for x in db['structure']['sites']]
        oxi_states = structure.composition.oxi_state_guesses()[0]
        
        # Iterate over each site and its coordination environment
        for i_site, element_name in enumerate(site_element_names):
            oxi_state=oxi_states[0][element_name]

            site_element_id=element_id_map[element_name]
            oxi_id= oxidation_state_id_map[oxi_state]

            if node_a_name == 'element':
                data = (site_element_id,oxi_id)
            else:
                data = (oxi_id,site_element_id)

            material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections


def create_relationships(node_a_csv, node_b_csv, material_csv,  mp_task, 
                         mp_task_params={}, 
                         connection_name='CONNECTS',
                         relationship_dir= None,
                         ):
    """
    Create relationships between nodes based on the provided CSV files.

    Args:
        node_a_csv (str): Path to the CSV file containing the first set of nodes.
        node_b_csv (str): Path to the CSV file containing the second set of nodes.
        material_csv (str): Path to the CSV file containing the material data.
        mp_task (function): A function that defines the task to be performed on each material.
        relationship_dir (str,optional): The directory where the relationship files are saved. Defaults to None.
        connection_name (str, optional): The name of the relationship between the nodes. Defaults to 'CONNECTS'.
        
    Returns:
        pandas.DataFrame: DataFrame containing the relationships between the nodes.

    """

    node_a_name=node_a_csv.split(os.sep)[-1].split('.')[0]
    node_b_name=node_b_csv.split(os.sep)[-1].split('.')[0]
    connection_name=f"{node_a_name.upper()}-{connection_name.upper()}-{node_b_name.upper()}"
    filepath=None
    if relationship_dir:
        filename=f"{connection_name}.csv"
        filepath=os.path.join(relationship_dir,filename)

        if os.path.exists(filepath):
            LOGGER.info(f"Relationship file {filepath} already exists. Skipping creation.")
            return None

    df_a=pd.read_csv(node_a_csv)
    df_b=pd.read_csv(node_b_csv)
    node_a_id_space = extract_id_column_headers(df_a)
    node_b_id_space = extract_id_column_headers(df_b)

    

    node_a_name_id_map = get_name_id_map(df_a)
    node_b_name_id_map = get_name_id_map(df_b)

    mp_task_params['node_a_name']=node_a_name
    mp_task_params['node_b_name']=node_b_name
    mp_task_params['node_a_id_map']=node_a_name_id_map
    mp_task_params['node_b_id_map']=node_b_name_id_map

    db_manager=DBManager()
    database_dir=db_manager.directory_path
    material_df=pd.read_csv(material_csv,index_col=0)
    material_ids=material_df['material_id:string'].tolist()
    files=[os.path.join(database_dir,f'{material_id}.json') for material_id in material_ids]
    materials=db_manager.process_task(mp_task,files,**mp_task_params)


    node_dict={
            f':START_ID({node_a_id_space})':[],
            f':END_ID({node_b_id_space})':[],
            f':TYPE':[],
    }
    properties_names=None
    # Get the properties names of relationships if any
    # Note : Need better way to handle this
    for material in materials:
        try:
            if len(material[0]) != 2:
                properties_names=[]
                for i,property in enumerate(material[0]):
                    if i>1:
                        properties_names.append(property[0])
                for name in properties_names:
                    node_dict.update({name:[] for name in properties_names} )
            break
        except:
            LOGGER.error(f"Error processing file {material}")
            continue
        

    # Iterate over the materials and create the relationships
    for imat, material in enumerate(materials):

        # Iterate over the connections in the material
        for i,connection in enumerate(material):
            
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

    if filepath:
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

    db_manager=DBManager()
    files=db_manager.database_files()
    # Get the structures and compositions for each material
    with Pool(N_CORES) as p:
        structure_composition_tuples = p.map(mp_task, files)

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
    pass
    # df=pd.read_csv('/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/graph_database/main/nodes/chemenv.csv')
    # map=get_name_id_map(df)
    # print(map)
    # material_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/graph_database/main/nodes/materials.csv'
    # material_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/json_database/mp-170.json'
    # node_a_csv='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/graph_database/main/nodes/chemenv.csv'
    # node_b_csv='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/graph_database/main/nodes/chemenv.csv'
    # df_a=pd.read_csv(node_a_csv)
    # df_b=pd.read_csv(node_b_csv)
    # node_a_id_space = extract_id_column_headers(df_a)
    # node_b_id_space = extract_id_column_headers(df_b)


    # node_a_name=os.path.basename(node_a_csv).split('.')[0]
    # node_b_name=os.path.basename(node_a_csv).split('.')[0]

    # node_a_id_map=get_name_id_map(df_a)
    # node_b_id_map=get_name_id_map(df_b)
    # create_bonding_task(material_file, node_a_name, node_b_name, node_a_id_map, node_b_id_map, bonding_method='geometric_electric')


if __name__ == '__main__':
    main()
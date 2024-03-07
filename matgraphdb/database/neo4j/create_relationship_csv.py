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

from matgraphdb.database.neo4j.node_types import *
from matgraphdb.utils import  GLOBAL_PROP_FILE, RELATIONSHIP_DIR,NODE_DIR, N_CORES, LOGGER, ENCODING_DIR, timeit
from matgraphdb.utils.periodic_table import atomic_symbols_map
from matgraphdb.database.json.utils import chunk_list,cosine_similarity

############################################################
# Below is for is for creating relationships between nodes
############################################################

def extract_id_column_headers(df):
    id_columns = []
    for column in df.columns:
        if re.search(r':ID\(.+?\)', column):
            match = re.search(r':ID\((.+?)\)', column)
            id_columns.append(match.group(1))
    return id_columns[0]

def create_element_element_task(material_file, use_chargemol_bonds=False):

    material_connections=[]
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)

    with open(GLOBAL_PROP_FILE) as f:
        db_gp = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]

    try:

        if use_chargemol_bonds:
            coord_connections = db['chargemol_bonding_connections']
        else:
            coord_connections = db['coordination_multi_connections']

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, site_element_name in enumerate(site_element_names):

            neighbors = coord_connections[i_site]
    
            site_chemElement_id=ELEMENTS_ID_MAP[site_element_name]
            # Iterate over each neighbor of the site
            for i_coord_env_neighbor in neighbors:
                element_neighbor_name = site_element_names[i_coord_env_neighbor]
                neighbor_chemElement_id=ELEMENTS_ID_MAP[element_neighbor_name]

                if use_chargemol_bonds:
                    i_element=atomic_symbols_map[site_element_name]
                    j_element=atomic_symbols_map[element_neighbor_name]
                    bond_order_avg= np.array(db_gp['bond_orders_avg'])[i_element,j_element]
                    bond_order_std= np.array(db_gp['bond_orders_std'])[i_element,j_element]

                    data = (site_chemElement_id,neighbor_chemElement_id,('bond_order_avg:float',bond_order_avg),('bond_orders_std:float',bond_order_std))
                else:
                    data = (site_chemElement_id,neighbor_chemElement_id)
                    
                material_connections.append(data)
    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
    return material_connections

def create_chemenv_chemenv_task(material_file, use_chargemol_bonds=False):
    
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

        if use_chargemol_bonds:
            coord_connections = db['chargemol_bonding_connections']
        else:
            coord_connections = db['coordination_multi_connections']

        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Iterate over each site and its coordination environment
        for i_site, coord_env in enumerate(coord_envs):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]
            neighbors = coord_connections[i_site]

            site_chemenv_id=CHEMENV_NAMES_ID_MAP[site_coord_env_name]
            # Iterate over each neighbor of the site
            for i_coord_env_neighbor in neighbors:

                element_neighbor_name = site_element_names[i_coord_env_neighbor]
                coord_env_neighbor_name=coord_envs[i_coord_env_neighbor]
                neighbor_chemenv_id=CHEMENV_NAMES_ID_MAP[coord_env_neighbor_name]

                if use_chargemol_bonds:
                    i_element=atomic_symbols_map[site_element_name]
                    j_element=atomic_symbols_map[element_neighbor_name]
                    bond_order_avg= np.array(db_gp['bond_orders_avg'])[i_element,j_element]
                    bond_order_std= np.array(db_gp['bond_orders_std'])[i_element,j_element]
                    data = (site_chemenv_id,neighbor_chemenv_id,('bond_orders_avg:float',bond_order_avg),('bond_orders_std:float',bond_order_std))
                else:
                    data = (site_chemenv_id,neighbor_chemenv_id)

                material_connections.append(data)

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return material_connections


def create_chemenv_element_task(material_file):
    
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

def create_chemenvElement_chemenvElement_task(material_file):
    
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
        coord_connections = db['chargemol_bonding_connections']

        site_element_names = [x['label'] for x in db['structure']['sites']]

        # Iterate over each site and its coordination environment
        for i_site, coord_env in enumerate(coord_envs):

            site_coord_env_name=coord_envs[i_site]
            site_element_name = site_element_names[i_site]
            neighbors = coord_connections[i_site]

            chemenv_element_name=site_element_name+'_'+site_coord_env_name

            site_chemenvElement_id=CHEMENV_ELEMENT_NAMES_ID_MAP[chemenv_element_name]
            # Iterate over each neighbor of the site
            for i_coord_env_neighbor in neighbors:
                element_neighbor_name = site_element_names[i_coord_env_neighbor]
                coord_env_neighbor_name=coord_envs[i_coord_env_neighbor]
                chemenv_element_neighbor_name=element_neighbor_name+'_'+coord_env_neighbor_name

                neighbor_chemenvElement_id=CHEMENV_ELEMENT_NAMES_ID_MAP[chemenv_element_neighbor_name]

                i_element=atomic_symbols_map[site_element_name]
                j_element=atomic_symbols_map[element_neighbor_name]
                bond_order_avg= np.array(db_gp['bond_orders_avg'])[i_element,j_element]
                bond_order_std= np.array(db_gp['bond_orders_std'])[i_element,j_element]

                data = (site_chemenvElement_id,neighbor_chemenvElement_id,('bond_order_avg:float',bond_order_avg),('bond_order_std:float',bond_order_std) )
                material_connections.append(data)
    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
    return material_connections

def create_oxi_state_element_task(material_file):

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

    # Get the properties names of relationships if any
    if len(materials[0][0]!= 2):
        properties_names=[]
        for i,property in enumerate(materials[0]):
            if i>1:
                properties_names.append(property[0])
    for name in properties_names:
        node_dict.update({name:[] for name in properties_names} )

    # Iterate over the materials and create the relationships
    for material in materials:

        # Iterate over the connections in the material
        for i,connection in enumerate(material):
            
            node_dict[f':START_ID({node_a_id_space})'].append(connection[0])
            node_dict[f':END_ID({node_b_id_space})'].append(connection[1])
            node_dict[f':TYPE'].append(connection_name)
            for iproperty,name in enumerate(properties_names):
                node_dict[name].append(connection[1+iproperty+1])

    df=pd.DataFrame(node_dict)

    # Create a column with sorted tuples of START_ID and END_ID
    df['id_tuple'] = df.apply(lambda x: tuple(sorted([x[f':START_ID({node_a_id_space})'], x[f':END_ID({node_b_id_space})']])), axis=1)

    # Group by the sorted tuple and count occurrences
    grouped = df.groupby('id_tuple')
    weights = grouped.size().reset_index(name='weight')
    
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
    n_materials_combs=len(material_combs)
    material_combs_values = [None]*n_materials_combs

    for i, material_comb in enumerate(material_combs):
        mat_id_1,mat_id_2=material_comb

        row_1 = features.iloc[mat_id_1].values
        row_2 = features.iloc[mat_id_2].values
        similarity=cosine_similarity(a=row_1,b=row_2)


        material_comb_values=(mat_id_1,mat_id_2,'RELATIONSHP',similarity)
        material_combs_values[i] = material_comb_values

    return material_combs_values

def megnet_lookup_task(material_combs, features):
    n_materials_combs=len(material_combs)
    material_combs_values = [None]*n_materials_combs

    for i, material_comb in enumerate(material_combs):
        mat_id_1,mat_id_2=material_comb

        row_1 = features.iloc[mat_id_1].values
        row_2 = features.iloc[mat_id_2].values

        similarity=cosine_similarity(a=row_1,b=row_2)

        material_comb_values=(mat_id_1,mat_id_2,'RELATIONSHP',similarity)
        material_combs_values[i] = material_comb_values

    return material_combs_values

def get_structure_composition_task(material_file):
    # Load material data from file and get their pymatgen Structure and Compositions objects
    with open(material_file) as f:
        db = json.load(f)
        struct = pmat.Structure.from_dict(db['structure'])
        composition = struct.composition
    return struct, composition

@timeit
def create_material_material_relationship(material_file_csv, mp_task, similarity_task,features,chunk_size=1000,filepath=None):
    df=pd.read_csv(material_file_csv)[['materialsId:ID(materialsId-ID)','name']]
    material_ids=df['materialsId:ID(materialsId-ID)'].values[:]
    node_id_space = 'materialsId-ID'

    material_id_combs=tuple(itertools.combinations_with_replacement(material_ids, r=2 ))
    material_id_combs_chunks = chunk_list(material_id_combs, chunk_size)

    # Get the structures and compositions for each material
    with Pool(N_CORES) as p:
        structure_composition_tuples=p.map(mp_task,MATERIAL_FILES[:])

    structures=[]
    compositions=[]
    for structure_composition_tuple in structure_composition_tuples:
        structure,composition=structure_composition_tuple
        structures.append(structure)
        compositions.append(composition)

    
    # # Convert the structures and compositions to pandas dataframe. This is required to use Matminer featurizers
    # # structure_data = pd.DataFrame({'structure': structures}, index=material_ids)
    # composition_data = pd.DataFrame({'composition': compositions}, index=material_ids)
    
    # # structure_featurizer = MultipleFeaturizer([XRDPowderPattern()])
    # composition_featurizer = MultipleFeaturizer([ElementFraction()])

    # # structure_features = structure_featurizer.featurize_dataframe(structure_data,"structure")
    # composition_features = composition_featurizer.featurize_dataframe(composition_data,"composition")

    # # structure_features=structure_features.drop(columns=['structure'])
    # composition_features=composition_features.drop(columns=['composition'])

    # features=composition_features
    features=features
    with Pool(N_CORES) as p:
        material_combs_chunks_values=p.map(partial(similarity_task,features=features), material_id_combs_chunks)

    material_combs_values=[]
    for material_combs_chunk_values in material_combs_chunks_values:
        material_combs_values.extend(material_combs_chunk_values)

    df =pd.DataFrame(material_combs_values,
                     columns=[f':START_ID({node_id_space})',f':END_ID({node_id_space})',f':TYPE','similarity'])
 
    if filepath is not None:
        df.to_csv(filepath, index=False)

    return df



def main():
    
    save_path=os.path.join(RELATIONSHIP_DIR)
    print('Save_path : ', save_path)
    os.makedirs(save_path,exist_ok=True)
    print('Creating Relationship...')


    # # Element - Element Connections
    create_relationships(node_a_csv=os.path.join(NODE_DIR,'elements.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
                         mp_task=partial(create_element_element_task,use_chargemol_bonds=True), 
                         connection_name='ELECTRONICALLY_CONNECTS',
                         filepath=os.path.join(save_path,'element_element.csv'))
    
    # # Element - Element Connections
    create_relationships(node_a_csv=os.path.join(NODE_DIR,'elements.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
                         mp_task=partial(create_element_element_task,use_chargemol_bonds=False), 
                         connection_name='GEOMETRICALLY_CONNECTS',
                         filepath=os.path.join(save_path,'element_element.csv'))
    
    # Chemenv - Chemenv Connections by chargemol bonding information
    create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
                         mp_task=partial(create_chemenv_chemenv_task,use_chargemol_bonds=True), 
                         connection_name='ELECTRONICALLY_CONNECTS',
                         filepath=os.path.join(save_path,'chemenv_chemenv.csv'))
    
    # Chemenv - Chemenv Connections by chargemol bonding information
    create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'chemenv_names.csv'), 
                         mp_task=partial(create_chemenv_chemenv_task,use_chargemol_bonds=False), 
                         connection_name='GEOMETRICALLY_CONNECTS',
                         filepath=os.path.join(save_path,'chemenv_chemenv.csv'))

    # # Chemenv - Element Connections
    create_relationships(node_a_csv=os.path.join(NODE_DIR,'chemenv_names.csv'),
                         node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
                         mp_task=create_chemenv_element_task,
                         connection_name='CAN_OCCUR',
                         filepath=os.path.join(save_path,'chemenv_elements.csv'))

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
import json
import itertools
from multiprocessing import Pool

from matgraphdb.database.neo4j.node_types import *
from matgraphdb.database.neo4j.utils import execute_statements, create_relationship_statement
from matgraphdb.database.utils import N_CORES, GLOBAL_PROP_FILE
from matgraphdb.utils import LOGGER
from matgraphdb.utils.periodic_table import atomic_symbols_map

def populate_relationship(material_file):
    create_statements = []
    
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
    
    with open(GLOBAL_PROP_FILE) as f:
        db_gp = json.load(f)
    
    # Extract material project ID from file name
    mpid = material_file.split(os.sep)[-1].split('.')[0]

    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'].replace(':', '_') for coord_env in db['coordination_environments_multi_weight']]
        coord_connections = db['chargemol_bonding_connections']

        site_element_names = [x['label'] for x in db['structure']['sites']]
    
        # Iterate over each site and its coordination environment
        for i_site, coord_env in enumerate(coord_envs):
            site_element_env_name = site_element_names[i_site]
            neighbors = coord_connections[i_site]
            
            # Iterate over each neighbor of the site
            for i_coord_env_neighbor in neighbors:
                element_neighbor_name = site_element_names[i_coord_env_neighbor]

                i_element=atomic_symbols_map[site_element_env_name]
                j_element=atomic_symbols_map[element_neighbor_name]
                bond_order_avg= np.array(db_gp['bond_orders_avg'])[i_element,j_element]
                bond_order_std= np.array(db_gp['bond_orders_std'])[i_element,j_element]

                create_statement=create_relationship_statement(
                                    node_a = {'type': 'Element', 'name': site_element_env_name}, 
                                    node_b = {'type': 'Element', 'name': element_neighbor_name}, 
                                    relationship_type='CONNECTS', 
                                    attributes= {'type': "'Element-Element'", 
                                                 'bond_order_avg':bond_order_avg, 
                                                 'bond_order_std':bond_order_std})
                create_statements.append(create_statement)


    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
            
    return create_statements



def populate_element_relationships(material_files=MATERIAL_FILES, n_cores=N_CORES):
    if n_cores == 1:
        print("Multiprocessing is not used")
        statements=[]
        for material_file in material_files:
            create_statements=populate_relationship(material_file)
            statements.extend(create_statements)
    else:
        print("Multiprocessing is used")
        with Pool(n_cores) as p:
            grouped_statments=p.map(populate_relationship, material_files)
            statements = list(itertools.chain.from_iterable(grouped_statments))
    return statements



def main():

    print("Creating Element relationships")
    create_statements = populate_element_relationships(material_files=MATERIAL_FILES[:])
    
    print("Executing Statements")
    execute_statements(create_statements)


if __name__ == '__main__':
    main()

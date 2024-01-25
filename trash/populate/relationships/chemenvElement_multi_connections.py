import json
import itertools
from multiprocessing import Pool

from matgraphdb.database.neo4j.node_types import *
from matgraphdb.database.neo4j.utils import execute_statements, create_relationship_statement
from matgraphdb.database.utils import N_CORES, GLOBAL_PROP_FILE
from matgraphdb.utils import LOGGER
from matgraphdb.utils.periodic_table import atomic_symbols_map

def populate_relationships(material_file):
    create_statements = []
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
        bond_orders = db["chargemol_bonding_orders"]

        composition_elements=db['elements']
        mpid=db['material_id']
        mpid_name=mpid.replace('-','_')

        magnetic_states_name=db['ordering']
        crystal_system_name=db['symmetry']['crystal_system'].lower()
        spg_name='spg_'+str(db['symmetry']['number'])
        site_element_names=[x['label'] for x in db['structure']['sites']]
        

        # Calculate the bond order
        total_site_elements = len(site_element_names)
        avg_bond_orders = [ [bond_order / total_site_elements for bond_order in site ] for site in bond_orders]


        for i_site,coord_env in enumerate(coord_envs):
            site_coord_env_name=coord_env
            element_name=site_element_names[i_site].split('_')[0]

            chemenv_element_name=element_name+'_'+site_coord_env_name
            
            create_statement=create_relationship_statement(node_a = {'type': 'chemenvElement', 'name': chemenv_element_name}, 
                                          node_b = {'type': 'Structure', 'name': mpid_name}, 
                                          relationship_type='APART_OF', 
                                          attributes= {'type': "'chemenvElement-Structure'"})
            create_statements.append(create_statement)

            create_statement=create_relationship_statement(node_a = {'type': 'chemenvElement', 'name': chemenv_element_name}, 
                                          node_b = {'type': 'crystal_system', 'name': crystal_system_name}, 
                                          relationship_type='APART_OF', 
                                          attributes= {'type': "'chemenvElement-crystal_system'"})
            create_statements.append(create_statement)

            create_statement=create_relationship_statement(node_a = {'type': 'chemenvElement', 'name': chemenv_element_name}, 
                                          node_b = {'type': 'magnetic_states', 'name': magnetic_states_name}, 
                                          relationship_type='APART_OF', 
                                          attributes= {'type': "'chemenvElement-magnetic_states'"})
            create_statements.append(create_statement)

            create_statement=create_relationship_statement(node_a = {'type': 'chemenvElement', 'name': chemenv_element_name}, 
                                          node_b = {'type': 'space_group', 'name': spg_name}, 
                                          relationship_type='APART_OF', 
                                          attributes= {'type': "'chemenvElement-space_group'"})
            create_statements.append(create_statement)

            neighbors=coord_connections[i_site]
            for i_coord_env_neighbor in neighbors:
                coord_env_neighbor_name=coord_envs[i_coord_env_neighbor]
                element_neighbor_name=site_element_names[i_coord_env_neighbor].split('_')[0]
                chemenv_element_neighbor_name=element_neighbor_name+'_'+coord_env_neighbor_name


                i_element=atomic_symbols_map[element_name]
                j_element=atomic_symbols_map[element_neighbor_name]
                bond_order_avg= np.array(db_gp['bond_orders_avg'])[i_element,j_element]
                bond_order_std= np.array(db_gp['bond_orders_std'])[i_element,j_element]

                create_statement=create_relationship_statement(
                                    node_a = {'type': 'chemenvElement', 'name': chemenv_element_name}, 
                                    node_b = {'type': 'chemenvElement', 'name': chemenv_element_neighbor_name}, 
                                    relationship_type='CONNECTS', 
                                    attributes= {'type': "'chemenvElement-chemenvElement'", 
                                                 'bond_order_avg':bond_order_avg, 
                                                 'bond_order_std':bond_order_std})
                create_statements.append(create_statement)

    except Exception as e:
        # Log any errors encountered during processing
        LOGGER.error(f"Error processing file {mpid}: {e}")
            
    return create_statements

def populate_chemenvElement_relationships(material_files=MATERIAL_FILES, n_cores=N_CORES):
    if n_cores == 1:
        print("Multiprocessing is not used")
        statements=[]
        for material_file in material_files:
            create_statements=populate_relationships(material_file)
            statements.extend(create_statements)
    else:
        print("Multiprocessing is used")
        with Pool(n_cores) as p:
            grouped_statments=p.map(populate_relationships, material_files)
            statements = list(itertools.chain.from_iterable(grouped_statments))
    return statements

def main():

    print("Creating chemenvElement relationships")
    create_statements=populate_chemenvElement_relationships(material_files=MATERIAL_FILES[:])

    print("Executing Statements")
    execute_statements(create_statements)


if __name__ == '__main__':
    main()


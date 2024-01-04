import json

from poly_graphs_lib.database.neo4j.populate.nodes.node_types import *
from poly_graphs_lib.database.neo4j.utils import execute_statements

def populate_relationship(material_file):
    create_statements = []
    
    # Load material data from file
    with open(material_file) as f:
        db = json.load(f)
        
    try:
        # Extract coordination environments, connections, and site element names from the material data
        coord_envs = [coord_env[0]['ce_symbol'].replace(':', '_') for coord_env in db['coordination_environments_multi_weight']]
        coord_connections = db['chargemol_bonding_connections']
        bond_orders = db["chargemol_bonding_orders"]
        site_element_names = [x['label'] for x in db['structure']['sites']]
        
        # Calculate the bond order
        total_site_elements = len(site_element_names)
        avg_bond_orders = [ [bond_order / total_site_elements for bond_order in site ] for site in bond_orders]

        # Iterate over each site and its coordination environment
        for i_site, coord_env in enumerate(coord_envs):
            site_element_env_name = site_element_names[i_site]
            neighbors = coord_connections[i_site]
            
            # Iterate over each neighbor of the site
            for i_coord_env_neighbor in neighbors:
                element_neighbor_name = site_element_names[i_coord_env_neighbor]
                avg_bond_order= avg_bond_orders[i_site][i_coord_env_neighbor]

                relationship_name = 'CONNECTS'
                
                # Create a Cypher query to create or update the relationship between the elements
                create_statement = f"MATCH (a:Element {{name: '{site_element_env_name}'}}), (b:Element {{name: '{element_neighbor_name}'}})\n"
                create_statement += f"MERGE (b)-[r:{relationship_name} {{type: 'Element-Element'}}]-(a)\n"
                create_statement += f"ON CREATE SET r.weight = 1, r.bond_order_sum = {avg_bond_order}\n"
                create_statement += f"ON MATCH SET r.weight = r.weight + 1, r.bond_order_sum = r.bond_order_sum + {avg_bond_order}\n"

                create_statements.append(create_statement)


    except Exception as e:
        print(e)
            
    return create_statements

def populate_element_relationships(material_files=MATERIAL_FILES):
    statements = []
    
    # Iterate over each material file
    for material_file in material_files[:]:
        create_statements = populate_relationship(material_file)
        statements.extend(create_statements)
    
    return statements

def main():
    create_statements = populate_element_relationships(material_files=MATERIAL_FILES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()

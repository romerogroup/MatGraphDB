import json

from poly_graphs_lib.database.neo4j.populate.nodes.node_types import *
from poly_graphs_lib.database.neo4j.utils import execute_statements

def populate_relationship(material_file):
    create_statements = []
    with open(material_file) as f:
        db = json.load(f)
        
    try:
        
        coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]
        coord_connections=db['bonding_cutoff_connections']

        site_element_names=[x['label'] for x in db['structure']['sites']]
        
        for i_site,coord_env in enumerate(coord_envs):
            site_element_env_name=site_element_names[i_site]

            neighbors=coord_connections[i_site]
            for i_coord_env_neighbor in neighbors:
                element_neighbor_name=site_element_names[i_coord_env_neighbor]
                relationship_name='CONNECTS'
                create_statement="MATCH (a:Element {name: '" + f'{site_element_env_name}' + "'}),"
                create_statement+="(b:Element {name: '" + f'{element_neighbor_name}'+ "'})\n"
                create_statement+="MERGE (b)-[r: %s { type: 'Element-Element' }]-(a)\n" % relationship_name
                create_statement+="ON CREATE SET r.weight = 1\n"
                create_statement+="ON MATCH SET r.weight = r.weight + 1"
                create_statements.append(create_statement)

    
    except Exception as e:
        print(e)
            
    return create_statements

def populate_element_relationships(material_files=MATERIAL_FILES):
    statements=[]
    for material_file in material_files[:]:
        create_statements=populate_relationship(material_file)
        statements.extend(create_statements)
    return statements

def main():
    create_statements=populate_element_relationships(material_files=MATERIAL_FILES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()


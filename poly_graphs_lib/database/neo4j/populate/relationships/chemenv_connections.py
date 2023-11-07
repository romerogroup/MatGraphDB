import json

from poly_graphs_lib.database.neo4j.populate.nodes.node_types import *
from poly_graphs_lib.database.neo4j.utils import execute_statements


def populate_relationship(material_file):
    create_statements = []

    with open(material_file) as f:
        db = json.load(f)
    
    if db['coordination_environments']:
        coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments']]
    
        coord_connections=db['coordination_connections']
        composition_elements=db['elements']
        mpid=db['material_id']
        mpid_name=mpid.replace('-','_')

        magnetic_states_name=db['ordering']
        crystal_system_name=db['symmetry']['crystal_system'].lower()
        spg_name=db['symmetry']['number']
        site_element_names=[x['label'] for x in db['structure']['sites']]
        for i_site,coord_env in enumerate(coord_envs):
            site_coord_env_name=coord_env
            

            element_name=site_element_names[i_site]
            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
            create_statement+="(b:Element {name: '" + f'{element_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenv-Element' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
            create_statement+="(b:Structure {name: '" + f'{mpid_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenv-Structure' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
            create_statement+="(b:crystal_system {name: '" + f'{crystal_system_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenv-crystal_system' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
            create_statement+="(b:magnetic_states {name: '" + f'{magnetic_states_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenv-magnetic_states' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
            create_statement+="(b:space_group {name: '" + f'{spg_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenv-space_group' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

            neighbors=coord_connections[i_site]
            for i_coord_env_neighbor in neighbors:
                coord_env_neighbor_name=coord_envs[i_coord_env_neighbor]

                relationship_name='CONNECTS'
                create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
                create_statement+="(b:chemenv {name: '" + f'{coord_env_neighbor_name}'+ "'})\n"
                create_statement+="MERGE (b)-[r: %s { type: 'chemenv-chemenv' }]-(a)\n" % relationship_name
                create_statement+="ON CREATE SET r.weight = 1\n"
                create_statement+="ON MATCH SET r.weight = r.weight + 1"
                create_statements.append(create_statement)
    else:
        pass
            
    return create_statements

def populate_chemenv_relationships(material_files=MATERIAL_FILES):
    statements=[]
    for material_file in material_files[:]:
        create_statements=populate_relationship(material_file)
        statements.extend(create_statements)
    return statements

def main():
    create_statements=populate_chemenv_relationships(material_files=MATERIAL_FILES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()


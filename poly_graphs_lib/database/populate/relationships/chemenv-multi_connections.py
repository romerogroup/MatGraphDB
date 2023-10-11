import json
from glob import glob

from neo4j import GraphDatabase

from poly_graphs_lib.core.voronoi_structure import VoronoiStructure
from poly_graphs_lib.utils.periodic_table import atomic_symbols
from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR


from poly_graphs_lib.database.json import material_files

def populate_relationship(material_file):
    create_statements = []
    with open(material_file) as f:
        db = json.load(f)
        
    try:
        
        coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]
    
        coord_connections=db['bonding_cutoff_connections']
        elements=db['elements']
        mpid=db['material_id']
        mpid_name=mpid.replace('-','_')

        magnetic_states_name=db['ordering']
        crystal_system_name=db['symmetry']['crystal_system'].lower()
        spg_name=db['symmetry']['number']
        element_names=[x['label'] for x in db['structure']['sites']]
        for i_site,coord_env in enumerate(coord_envs):
            site_coord_env_name=coord_env
            

            element_name=element_names[i_site]
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
    except:
        pass
            
    return create_statements

def main():
    
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)
    

    for material_file in material_files:
        create_statements=populate_relationship(material_file)


        for execute_statment in create_statements:
            session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()





# def populate_chemenv_element_relationship(cif_file):
#     create_statements = []

    
#     voronoi_strucutre = VoronoiStructure(structure_id = cif_file)
#     voro_dict=voronoi_strucutre.as_dict()

#     for poly_info in voro_dict['voronoi_polyhedra_info']:
#         site_coord_env_name=poly_info['coordination_envrionment'][0]['ce_symbol']
#         site_coord_env_name=site_coord_env_name.replace(':','_')

#         element_name=atomic_symbols[poly_info['species']]
#         for coord_env_neighbor in poly_info['neighbor_coordination_envrionment']:
#             coord_env_neighbor_name=coord_env_neighbor[0]['ce_symbol']
#             coord_env_neighbor_name=coord_env_neighbor_name.replace(':','_')

#             relationship_name='CONNECTS'
#             create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
#             create_statement+="(b:chemenv {name: '" + f'{coord_env_neighbor_name}'+ "'})\n"
#             create_statement+=f"MERGE (a)-[r:{relationship_name}]-(b)\n"
#             create_statement+="ON CREATE SET r.weight = 1\n"
#             create_statement+="ON MATCH SET r.weight = r.weight + 1"
#             create_statements.append(create_statement)

#             relationship_name='HAS'
#             create_statement="MATCH (a:chemenv {name: '" + f'{site_coord_env_name}' + "'}),"
#             create_statement+="(b:Element {name: '" + f'{element_name}'+ "'})\n"
#             create_statement+=f"MERGE (b)-[r:{relationship_name}]->(a)\n"
#             create_statement+="ON CREATE SET r.weight = 1\n"
#             create_statement+="ON MATCH SET r.weight = r.weight + 1"
#             create_statements.append(create_statement)

#     return create_statements

import os
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

        print(element_names)

        for i_site,element in enumerate(element_names):

            element_name=element_names[i_site]
            relationship_name='CONNECTS'
            create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
            create_statement+="(b:Element {name: '" + f'{element_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'Structure-Element' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

        element_name=element_names[i_site]
        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:magnetic_states {name: '" + f'{magnetic_states_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-magnetic_states' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)

        element_name=element_names[i_site]
        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:crystal_system {name: '" + f'{crystal_system_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-crystal_system' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)

        element_name=element_names[i_site]
        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:space_group {name: '" + f'{spg_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-space_group' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)

    
    except Exception as e:
        print(e)
            
    return create_statements

def main():
    
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)
    
    
    for material_file in material_files[:]:
       
        if material_file.split(os.sep)[-1]=='mp-1238815.json':
            create_statements=populate_relationship(material_file)

            print(create_statements)


            # for execute_statment in create_statements:
            #     session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()





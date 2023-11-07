import json

from poly_graphs_lib.database.neo4j.populate.nodes.node_types import *
from poly_graphs_lib.database.neo4j.utils import execute_statements


def populate_relationship(material_file):
    create_statements = []
    with open(material_file) as f:
        db = json.load(f)
        
    try:
        composition_elements=db['elements']
        mpid=db['material_id']
        mpid_name=mpid.replace('-','_')
        magnetic_states_name=db['ordering']
        crystal_system_name=db['symmetry']['crystal_system'].lower()
        spg_name=db['symmetry']['number']

        for element in composition_elements:

            element_name=element
            relationship_name='CONNECTS'
            create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
            create_statement+="(b:Element {name: '" + f'{element_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'Structure-Element' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)

        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:magnetic_states {name: '" + f'{magnetic_states_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-magnetic_states' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)


        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:crystal_system {name: '" + f'{crystal_system_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-crystal_system' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)


        relationship_name='CONNECTS'
        create_statement="MATCH (a:Structure {name: '" + f'{mpid_name}' + "'}),"
        create_statement+="(b:space_group {name: '" + f'{spg_name}'+ "'})\n"
        create_statement+="MERGE (b)-[r: %s { type: 'Structure-space_group' }]-(a)\n" % relationship_name
        create_statement+="ON CREATE SET r.weight = 1\n"
        create_statement+="ON MATCH SET r.weight = r.weight + 1"
        create_statements.append(create_statement)

    
    except Exception as e:
        print('------------------------')
        print(mpid)
        print(e)
            
    return create_statements

def populate_structure_relationships(material_files=MATERIAL_FILES):
    statements=[]
    for material_file in material_files[:]:
        create_statements=populate_relationship(material_file)
        statements.extend(create_statements)
    return statements


def main():
    create_statements=populate_structure_relationships(material_files=MATERIAL_FILES)
    execute_statements(create_statements)



if __name__ == '__main__':
    main()

import os
import sys
import json

from matgraphdb.database.neo4j.node_types import *
from matgraphdb.database.neo4j.utils import execute_statements

def populate_relationship(material_file):
    create_statements = []
    with open(material_file) as f:
        db = json.load(f)
        
    try:
        
        coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db['coordination_environments_multi_weight']]

        coord_connections=db['chargemol_bonding_connections']


        composition_elements=db['elements']
        mpid=db['material_id']
        mpid_name=mpid.replace('-','_')

        magnetic_states_name=db['ordering']
        crystal_system_name=db['symmetry']['crystal_system'].lower()
        spg_name='spg_'+ str(db['symmetry']['number'])
        site_element_names=[x['label'] for x in db['structure']['sites']]

        wyckoff_letters=db['wyckoffs']
        
        for i_site,coord_env in enumerate(coord_envs):

            site_coord_env_name=coord_env
            element_name=site_element_names[i_site].split('_')[0]
            chemenv_element_name=element_name+'_'+site_coord_env_name


            wyckoff_letter=wyckoff_letters[i_site]
            spg_wyckoff_name=spg_name+'_'+wyckoff_letter

            relationship_name='CONNECTS'
            create_statement="MATCH (a:chemenvElement {name: '" + f'{chemenv_element_name}' + "'}),"
            create_statement+="(b:spg_wyckoff {name: '" + f'{spg_wyckoff_name}'+ "'})\n"
            create_statement+="MERGE (b)-[r: %s { type: 'chemenvElement-spg_wyckoff' }]-(a)\n" % relationship_name
            create_statement+="ON CREATE SET r.weight = 1\n"
            create_statement+="ON MATCH SET r.weight = r.weight + 1"
            create_statements.append(create_statement)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(exc_type, exc_value)
        print(f"Error occurred at line: {exc_traceback.tb_lineno}")
            
    return create_statements

def populate_spg_wyckoff_relationships(material_files=MATERIAL_FILES):
    statements=[]
    for material_file in material_files[:]:
        create_statements=populate_relationship(material_file)
        statements.extend(create_statements)
    return statements

def main():
    create_statements=populate_spg_wyckoff_relationships(material_files=MATERIAL_FILES)
    execute_statements(create_statements)



if __name__ == '__main__':
    main()




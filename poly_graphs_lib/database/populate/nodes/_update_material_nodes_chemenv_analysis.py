import os
from glob import glob

from neo4j import GraphDatabase

from poly_graphs_lib.utils.periodic_table import atomic_symbols
from poly_graphs_lib.core.voronoi_structure import VoronoiStructure
from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR


def update_materials(cif_files):

    create_statements=[]
    for cif_file in cif_files[:1]:
        mpid_name=cif_file.split(os.sep)[-1].split('.')[0]
        mpid_name=mpid_name.replace('-','_')


        voronoi_strucutre = VoronoiStructure(structure_id = cif_file)
        voro_dict=voronoi_strucutre.as_dict()

        coord_envs_unit=[]
        coord_envs_neighbors=[]
        
        for poly_info in voro_dict['voronoi_polyhedra_info']:
            site_coord_env_name=poly_info['coordination_envrionment'][0]['ce_symbol']
            site_coord_env_name=site_coord_env_name.replace(':','_')

            neighbor_unit_indices=poly_info['neighbor_unit_indices']
            element_name=atomic_symbols[poly_info['species']]

            coord_envs_unit.append(site_coord_env_name)
            coord_envs_neighbors.append(neighbor_unit_indices)


        coord_envs_neighbors = [[str(x) for x in inner_list] for inner_list in coord_envs_neighbors]
        coord_envs_neighbors=[','.join(x) for x in coord_envs_neighbors]
        coord_envs_neighbors='_'.join(coord_envs_neighbors)


        execute_statement="MATCH (m:Structure {name: '%s'})" % mpid_name
        execute_statement+=" SET m += { %s: %s}" % ('chemenv_unit',coord_envs_unit)
        create_statements.append(execute_statement)

        execute_statement="MATCH (m:Structure {name: '%s'})" % mpid_name
        execute_statement+=" SET m += { %s: '%s'}" % ('chemenv_neighbors',coord_envs_neighbors)
        create_statements.append(execute_statement)

    return create_statements


def main():
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)
    
    cif_files =  glob(CIF_DIR + '\*.cif')
    create_statements=update_materials(cif_files)

    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
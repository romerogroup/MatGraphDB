import os
import json
from glob import glob

from neo4j import GraphDatabase

from poly_graphs_lib.core.structure import Structure
from poly_graphs_lib.utils.periodic_table import atomic_symbols
from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.database.populate.nodes import Node

from poly_graphs_lib.database.json import material_files


def populate_materials(material_files):
    create_statements = []

    for i,mat_file in enumerate(material_files):
        if i%100==0:
            print(i)

        with open(mat_file) as f:
            db = json.load(f)

        mpid_name=mat_file.split(os.sep)[-1].split('.')[0]
        mpid_name=mpid_name.replace('-','_')
        node=Node(node_name=mpid_name,class_name='Structure')

        for key,value in db.items():
            # print(key, value)

            if key == 'structure':
                pass
            elif key=='sites':
                pass   
            elif key=='coordination_connections':
                pass
            elif key=='bonding_cutoff_connections':
                pass
            elif key=='composition_reduced' or key=='composition':
                comp_reduced=[ db[key][element] for element in db['elements']]
                node.add_property(name=key,value=comp_reduced)
            elif key=='symmetry':
                for symmetry_key in db[key].keys():
                    if symmetry_key=='version':
                        pass
                    elif symmetry_key=='crystal_system':
                        node.add_property(name=symmetry_key,value=db[key][symmetry_key].lower())
                    else:
                        node.add_property(name=symmetry_key,value=db[key][symmetry_key])
            elif key=='coordination_environments':
                try:
                    coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db[key]]
                    node.add_property(name='ce_symbol',value=coord_envs)
                except:
                    pass
            elif key=='coordination_environments_multi_weight':
                try:
                    coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db[key]]
                    node.add_property(name='ce_symbol',value=coord_envs)
                except:
                    pass
            else:
                node.add_property(name=key,value=value)

        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)

    return create_statements



def main():
    
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    # Create a constraint on the Structure.name field
    unique_constraint="CREATE CONSTRAINT UNIQUE_STRUCTURE_NAME IF NOT EXISTS ON (s:Structure) ASSERT s.name IS UNIQUE;"
    session.run(unique_constraint)

    create_statements=populate_materials(material_files)

    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()


# From cif files
# def populate_materials(cif_files):
#     create_statements = []

#     for cif_file in cif_files:
#         mpid_name=cif_file.split(os.sep)[-1].split('.')[0]
#         mpid_name=mpid_name.replace('-','_')
#         node=Node(node_name=mpid_name,class_name='Structure')

#         structure=Structure(structure_id=cif_file)

#         unqiue_atoms= sorted([comp for comp in structure.composition])
#         atoms_unit=[atomic_symbols[atomic_number] for atomic_number in structure.atoms_unit]

#         node.add_property(name='unique_atoms',value=unqiue_atoms)
#         node.add_property(name='atoms',value=atoms_unit)

#         node.add_property(name='spg_number',value=structure.spg_number)

#         node.add_points(name='frac_coords',points=structure.frac_coords_unit)
#         node.add_points(name='direct_lattice',points=structure.direct_lattice)
#         node.add_points(name='reciprocal_lattice',points=structure.direct_lattice)

#         execute_statement=node.final_execute_statement()
#         create_statements.append(execute_statement)


#     return create_statements
import json

import numpy as np

from neo4j import GraphDatabase

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.database.populate.nodes import Node



def populate_nodes(node_names):
    class_name='magnetic_states'
    create_statements = []
    for node_name in node_names:
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    # Used to get magnetic states
    # from poly_graphs_lib.database.json import material_files
    # magnetic_states=[]
    # for i,mat_file in enumerate(material_files):
    #     if i%100==0:
    #         print(i)
    #     with open(mat_file) as f:
    #         db = json.load(f)
    #     magnetic_state=db["ordering"]
    #     if magnetic_state not in magnetic_states:
    #         magnetic_states.append(magnetic_state)


    magnetic_states=['NM', 'FM', 'FiM', 'AFM', 'Unknown']
    create_statements=populate_nodes(node_names=magnetic_states)
    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
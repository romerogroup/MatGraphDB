import numpy as np

from neo4j import GraphDatabase

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.database.populate.nodes import Node

def populate_nodes(node_names):
    class_name='space_group'
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

    spg = np.arange(1,231)
    spg_names=[f'spg_{i}' for i in spg]
    create_statements=populate_nodes(node_names=spg_names)
    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
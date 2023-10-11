from neo4j import GraphDatabase

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.cfg.coordination_geometries_files import mp_coord_encoding
from poly_graphs_lib.database.populate.nodes import Node

def populate_chemenv_nodes(chemenv_names):
    class_name='chemenv'
    create_statements = []
    for node_name in chemenv_names:
        node_name=node_name.replace(':','_')
        
        node=Node(node_name=node_name,class_name=class_name)

        execute_statement=node.final_execute_statement()

        create_statements.append(execute_statement)
    return create_statements

def main():
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    chemenv_names=mp_coord_encoding.keys()
    create_statements=populate_chemenv_nodes(chemenv_names=chemenv_names)
    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
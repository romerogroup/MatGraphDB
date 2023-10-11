from neo4j import GraphDatabase

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.database.populate.nodes import Node

def populate_crystal_system_nodes(crystal_systems):
    class_name='crystal_system'
    create_statements = []
    for node_name in crystal_systems:
        
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    crystal_systems = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
    create_statements=populate_crystal_system_nodes(crystal_systems=crystal_systems)
    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
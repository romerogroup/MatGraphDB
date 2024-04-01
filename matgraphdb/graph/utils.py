from typing import List

from neo4j import GraphDatabase

from matgraphdb.utils import PASSWORD,USER,LOCATION,DB_NAME

def execute_statements(statements: List[str]):

    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    session = connection.session(database=DB_NAME)

    for execute_statement in statements:
        session.run(execute_statement)


    session.close()
    connection.close()
  

def create_database(connection):

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database="neo4j")

    execute_statment = f"CREATE DATABASE `{DB_NAME}`"
    session.run(execute_statment)

    session.close()
    connection.close()


def create_relationship_statement(node_a, node_b, relationship_type, attributes):
    """
    Generate a Cypher query for creating a relationship between two nodes.
    """
    attr_string = ', '.join(f"{key}: {value}" for key, value in attributes.items())
    return (
        f"MATCH (a:{node_a['type']} {{name: '{node_a['name']}'}}), "
        f"(b:{node_b['type']} {{name: '{node_b['name']}'}})\n"
        f"MERGE (a)-[r:{relationship_type} {{ {attr_string} }}]-(b)\n"
        f"ON CREATE SET r.weight = 1\n"
        f"ON MATCH SET r.weight = r.weight + 1\n"
    )


def delete_database():

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))

    # To read and write to the data base you must open a session
    session = connection.session(database="neo4j")

    execute_statment = f"DROP DATABASE `{DB_NAME}`"
    session.run(execute_statment)

    session.close()
    connection.close()


def delete_nodes_relationships():
    execute_statment = ["MATCH (n) DETACH DELETE n"]
    execute_statements(execute_statment,n_cores=1)

def delete_relationships():
    execute_statment = ["MATCH ()-[r]-() DELETE r"]
    execute_statements(execute_statment,n_cores=1)


from typing import List

from neo4j import GraphDatabase

from matgraphdb.utils import PASSWORD,USER,LOCATION,DB_NAME

from typing import List

def execute_statements(statements: List[str]):
    """
    Executes a list of statements on a graph database.

    Args:
        statements (List[str]): A list of statements to be executed.

    Returns:
        None
    """
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    session = connection.session(database=DB_NAME)

    for execute_statement in statements:
        session.run(execute_statement)

    session.close()
    connection.close()
  

def create_database(connection):
    """
    Creates a new database in the Neo4j graph database server.

    Args:
        connection: The connection object to the Neo4j graph database server.

    Returns:
        None
    """
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
    Creates a Cypher query statement to create or update a relationship between two nodes.

    Args:
        node_a (dict): The first node in the relationship, containing 'type' and 'name' attributes.
        node_b (dict): The second node in the relationship, containing 'type' and 'name' attributes.
        relationship_type (str): The type of relationship between the nodes.
        attributes (dict): Additional attributes to be added to the relationship.

    Returns:
        str: The Cypher query statement.

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
    """
    Deletes the specified database.

    This function connects to the database server, opens a session, and executes a statement to drop the specified database.

    Parameters:
    None

    Returns:
    None
    """

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))

    # To read and write to the data base you must open a session
    session = connection.session(database="neo4j")

    execute_statment = f"DROP DATABASE `{DB_NAME}`"
    session.run(execute_statment)

    session.close()
    connection.close()


def delete_nodes_relationships():
    """
    Deletes all nodes and their relationships in the graph database.

    This function executes a Cypher statement to match all nodes in the graph and detach/delete them along with their relationships.

    Parameters:
    None

    Returns:
    None
    """
    execute_statment = ["MATCH (n) DETACH DELETE n"]
    execute_statements(execute_statment, n_cores=1)

def delete_relationships():
    """
    Deletes all relationships in the graph database.

    This function executes a Cypher query to match all relationships in the graph database
    and deletes them.

    Parameters:
        None

    Returns:
        None
    """
    execute_statment = ["MATCH ()-[r]-() DELETE r"]
    execute_statements(execute_statment, n_cores=1)


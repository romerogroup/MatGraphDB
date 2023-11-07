import os

from neo4j import GraphDatabase

from poly_graphs_lib.database.neo4j import PASSWORD,USER,LOCATION,DB_NAME

def create_database():

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database="neo4j")

    execute_statment = f"CREATE DATABASE `{DB_NAME}`"
    session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    create_database()
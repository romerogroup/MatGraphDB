from typing import List
from multiprocessing import Pool

from neo4j import GraphDatabase
from neo4j.exceptions import TransientError

from poly_graphs_lib.database.utils import N_CORES, PASSWORD,USER,LOCATION,DB_NAME

def execute_statements_task(statement):
    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    session = connection.session(database=DB_NAME)

    for _ in range(20):  # Retry up to 3 times
        try:
            # Execute the statement
            # This is just a placeholder, replace with your actual code
            session.run(statement)
            break
        except TransientError as e:
            print(f"Transient error occurred: {e}, retrying...")

    session.close()
    connection.close()

def execute_statements(statements: List[str], n_cores=N_CORES):

    if n_cores == 1:
        connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
        session = connection.session(database=DB_NAME)

        for execute_statement in statements:
            session.run(execute_statement)
        connection.close()
    else:
        with Pool(n_cores) as p:
            p.map(execute_statements_task, statements)


# def execute_statements(statements:List[str], n_cores=N_CORES):
#     connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
#     if n_cores==1:
#         # This statement Connects to the database server
#         connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
#         # To read and write to the data base you must open a session
#         session = connection.session(database=DB_NAME)

#         for execute_statment in statements:
#             session.run(execute_statment)

#         session.close()
#         connection.close()
#     else:

#         with Pool(n_cores) as p:
#             p.map(mp_execute_statements, statements)
#     connection.close()
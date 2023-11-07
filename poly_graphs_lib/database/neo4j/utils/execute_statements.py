from typing import List
from multiprocessing import Pool

from neo4j import GraphDatabase

from poly_graphs_lib.database.neo4j import PASSWORD,USER,LOCATION,DB_NAME
from poly_graphs_lib.database import N_CORES


CONNECTION = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))

def mp_execute_statements(statment):

    session = CONNECTION.session(database=DB_NAME)
    session.run(statment)
    session.close()
    

    return None

def execute_statements(statements:List[str], n_cores=N_CORES):
    if n_cores==1:
        # To read and write to the data base you must open a session
        session = CONNECTION.session(database=DB_NAME)

        for execute_statment in statements:
            session.run(execute_statment)

        session.close()
    else:

        with Pool(n_cores) as p:
            p.map(mp_execute_statements, statements)

CONNECTION.close()


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
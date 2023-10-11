import os

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import Voronoi
from neo4j import GraphDatabase
import pymatgen.core as pmat

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME

def main():

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    execute_statment = "MATCH (n) DETACH DELETE n"
    session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()
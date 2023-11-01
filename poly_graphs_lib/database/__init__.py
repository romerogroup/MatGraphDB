import os
from poly_graphs_lib.utils import PROJECT_DIR

PASSWORD="password"
DBMS_NAME="neo4j"
LOCATION="bolt://localhost:7687"
DB_NAME='chemenvdb-multi'
# DB_NAME='chemenvdb-test'

CIF_DIR=os.path.join(PROJECT_DIR,'data','external','nelement_max_2_nsites_max_6_3d')

MP_DIR=os.path.join(PROJECT_DIR,'data','raw','materials_project')
DB_DIR=os.path.join(PROJECT_DIR,'data','raw','materials_project','json_database')
DB_CALC_DIR=os.path.join(PROJECT_DIR,'data','raw','materials_project','calculations','database')
N_CORES=20
import os
from poly_graphs_lib.utils import PROJECT_DIR


MP_DIR=os.path.join(PROJECT_DIR,'data','raw','materials_project_ternary')
DB_DIR=os.path.join(MP_DIR,'json_database')
DB_CALC_DIR=os.path.join(MP_DIR,'calculations','database')
N_CORES=6
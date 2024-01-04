import os

from poly_graphs_lib.utils import PROJECT_DIR
from poly_graphs_lib.utils import LOGGER

MP_DIR=os.path.join(PROJECT_DIR,'data','processed','materials_project_nelements_3')
DB_DIR=os.path.join(MP_DIR,'json_database')
DB_CALC_DIR=os.path.join(MP_DIR,'calculations','MaterialsData')
GLOBAL_PROP_FILE=os.path.join(MP_DIR,'global_properties.json')

LOGGER.info(f'Database Dir : {MP_DIR}')
print('Database Dir : ', MP_DIR)
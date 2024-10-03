import os
import logging 

from pathlib import Path
import numpy as np
import yaml
import tempfile

# numpy options
large_width = 400
precision=3
np.set_printoptions(linewidth=large_width,precision=precision)

logger = logging.getLogger(__name__)



# Important directory paths
FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])

# ROOT = str(FILE.parents[2])
# LOG_DIR=os.path.join(ROOT,'logs')
# DATA_DIR=os.path.join(ROOT,'data')
# CONFIG_FILE=os.path.join(ROOT,'config.yml')
# PRIVATE_CONFIG_FILE=os.path.join(ROOT,'private_config.yml')



# Load config from yaml file
with open(CONFIG_FILE, 'r') as f:
    CONFIG = yaml.safe_load(f)



# # Neo4j variables
# USER=CONFIG['USER']
# PASSWORD=CONFIG['PASSWORD']
# LOCATION=CONFIG['LOCATION']
# GRAPH_DB_NAME=CONFIG['GRAPH_DB_NAME']

# # MP_DIR=os.path.join(ROOT,'data','production',CONFIG['DB_NAME'])
# EXTERNAL_DATA_DIR=os.path.join(ROOT,'data','external')

# MP_DIR=os.path.join(ROOT,'data','production',CONFIG['DB_NAME'])
# ML_DIR=os.path.join(MP_DIR,'ML')
# ML_SCRATCH_RUNS_DIR=os.path.join(ML_DIR,'scratch_runs')
# MATERIAL_PARQUET_FILE=os.path.join(MP_DIR,'materials_database.parquet')

# TMP_DIR=os.path.join(MP_DIR,"tmp")
# DB_DIR=os.path.join(MP_DIR,'json_database')
# DATASETS_DIR=os.path.join(MP_DIR,'datasets')

# GRAPH_DIR=os.path.join(MP_DIR,'graph_database')
# MAIN_GRAPH_DIR=os.path.join(GRAPH_DIR,'main')


# ENCODING_DIR=os.path.join(MP_DIR,'encodings')
# SIMILARITY_DIR=os.path.join(MP_DIR,'similarities')
# NODE_DIR=os.path.join(GRAPH_DIR,'nodes')
# RELATIONSHIP_DIR=os.path.join(GRAPH_DIR,'relationships')
# DB_CALC_DIR=os.path.join(MP_DIR,'calculations','MaterialsData')
# GLOBAL_PROP_FILE=os.path.join(MP_DIR,'global_properties.json')

# NEO4J_DESKTOP_DIR=CONFIG['NEO4J_DESKTOP_DIR']
# DBMSS_DIR = os.path.join(NEO4J_DESKTOP_DIR,'relate-data','dbmss')





import os
from pathlib import Path
import numpy as np
import yaml
import yaml
# numpy options
large_width = 400
precision=3
np.set_printoptions(linewidth=large_width,precision=precision)

# Important directory paths
FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])  # poly_graph_lib
ROOT = str(FILE.parents[2])  # Graph_Network_Project
LOG_DIR=os.path.join(ROOT,'logs')
DATA_DIR=os.path.join(ROOT,'data')
CONFIG_FILE=os.path.join(ROOT,'config.yml')
PRIVATE_CONFIG_FILE=os.path.join(ROOT,'private_config.yml')

# Load config from yaml file
with open(CONFIG_FILE, 'r') as f:
    CONFIG = yaml.safe_load(f)

N_CORES=CONFIG['N_CORES']


# Load config from yaml file
with open(PRIVATE_CONFIG_FILE, 'r') as f:
    PRIVATE_CONFIG = yaml.safe_load(f)


MP_API_KEY=PRIVATE_CONFIG['MP_API_KEY']
OPENAI_API_KEY=PRIVATE_CONFIG['OPENAI_API_KEY']

MP_DIR=os.path.join(ROOT,'data','processed',CONFIG['DB_NAME'])
DB_DIR=os.path.join(MP_DIR,'json_database')
GRAPH_DIR=os.path.join(MP_DIR,'graph_database')
ENCODING_DIR=os.path.join(MP_DIR,'encodings')
NODE_DIR=os.path.join(GRAPH_DIR,'nodes')
RELATIONSHIP_DIR=os.path.join(GRAPH_DIR,'relationships')
DB_CALC_DIR=os.path.join(MP_DIR,'calculations','MaterialsData')
GLOBAL_PROP_FILE=os.path.join(MP_DIR,'global_properties.json')


# Neo4j variables
USER="neo4j"
PASSWORD="password"
LOCATION="bolt://localhost:7687"
DB_NAME='test2'


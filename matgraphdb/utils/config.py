import os
from pathlib import Path
import numpy as np
import yaml

# numpy options
large_width = 400
precision=3
np.set_printoptions(linewidth=large_width,precision=precision)


def get_cpus_per_node():
    cpu_per_node = os.getenv('SLURM_JOB_CPUS_PER_NODE')
    if cpu_per_node is None:
        cpus_node_list = 1
    elif '(x' in cpu_per_node:
        cpu_per_node, num_nodes= cpu_per_node.strip(')').split('(x')
        cpus_node_list = [int(cpu_per_node) for _ in range(int(num_nodes))]
    else:
        cpus_node_list = [int(x) for x in cpu_per_node.split(',')]
    return cpus_node_list

def get_num_tasks():
    num_tasks = os.getenv('SLURM_NTASKS')
    if num_tasks:
        num_tasks = int(num_tasks)
    return num_tasks

def get_num_nodes():
    num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
    return num_nodes

def get_total_cores(cpus_per_node):
    return sum(cpus_per_node)

def get_num_cores(n_cores=None):
    cpus_per_node = get_cpus_per_node()
    if n_cores:
        return n_cores
    elif isinstance(cpus_per_node,list):
        return get_total_cores(cpus_per_node)
    else:
        return cpus_per_node


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

N_CORES=get_num_cores(n_cores=CONFIG['N_CORES'])

# Neo4j variables
USER=CONFIG['USER']
PASSWORD=CONFIG['PASSWORD']
LOCATION=CONFIG['LOCATION']
GRAPH_DB_NAME=CONFIG['GRAPH_DB_NAME']

# MP_DIR=os.path.join(ROOT,'data','production',CONFIG['DB_NAME'])
EXTERNAL_DATA_DIR=os.path.join(ROOT,'data','external')

MP_DIR=os.path.join(ROOT,'data','production',CONFIG['DB_NAME'])
ML_DIR=os.path.join(MP_DIR,'ML')
ML_SCRATCH_RUNS_DIR=os.path.join(ML_DIR,'scratch_runs')
MATERIAL_PARQUET_FILE=os.path.join(MP_DIR,'materials_database.parquet')

TMP_DIR=os.path.join(MP_DIR,"tmp")
DB_DIR=os.path.join(MP_DIR,'json_database')
DATASETS_DIR=os.path.join(MP_DIR,'datasets')

GRAPH_DIR=os.path.join(MP_DIR,'graph_database')
MAIN_GRAPH_DIR=os.path.join(GRAPH_DIR,'main')


ENCODING_DIR=os.path.join(MP_DIR,'encodings')
SIMILARITY_DIR=os.path.join(MP_DIR,'similarities')
NODE_DIR=os.path.join(GRAPH_DIR,'nodes')
RELATIONSHIP_DIR=os.path.join(GRAPH_DIR,'relationships')
DB_CALC_DIR=os.path.join(MP_DIR,'calculations','MaterialsData')
GLOBAL_PROP_FILE=os.path.join(MP_DIR,'global_properties.json')

NEO4J_DESKTOP_DIR=CONFIG['NEO4J_DESKTOP_DIR']
DBMSS_DIR = os.path.join(NEO4J_DESKTOP_DIR,'relate-data','dbmss')





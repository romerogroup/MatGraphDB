# Important directory paths
from matgraphdb.utils.config import FILE, PKG_DIR, ROOT, LOG_DIR, DATA_DIR

# Impot config settings
from matgraphdb.utils.config import CONFIG, MP_API_KEY, N_CORES

# Database paths
from matgraphdb.utils.config import (MP_DIR, DB_DIR, GRAPH_DIR, NODE_DIR, RELATIONSHIP_DIR, 
                                          DB_CALC_DIR, GLOBAL_PROP_FILE, N_CORES)

# Neo4j variables
from matgraphdb.utils.config import USER, PASSWORD, LOCATION, DB_NAME

# Other important variables
from matgraphdb.utils.log_config import setup_logging
from matgraphdb.utils.timing import Timer, timeit

# Initialize logger
LOGGER = setup_logging(log_dir=LOG_DIR)
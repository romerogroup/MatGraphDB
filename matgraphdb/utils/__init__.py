# Important directory paths
from matgraphdb.utils.config import FILE, PKG_DIR, ROOT, DATA_DIR, EXTERNAL_DATA_DIR

# Impot config settings
from matgraphdb.utils.config import CONFIG, N_CORES

# Database paths
from matgraphdb.utils.config import (MP_DIR, DB_DIR, DATASETS_DIR, GRAPH_DIR, NODE_DIR, RELATIONSHIP_DIR, SIMILARITY_DIR,
                                          DB_CALC_DIR, GLOBAL_PROP_FILE, N_CORES, ENCODING_DIR,MATERIAL_PARQUET_FILE)

# Neo4j variables
from matgraphdb.utils.config import (USER, PASSWORD, LOCATION, GRAPH_DB_NAME, DBMSS_DIR,
                                     MAIN_GRAPH_DIR, GRAPH_DIR)

# ML variables
from matgraphdb.utils.config import (ML_DIR,ML_SCRATCH_RUNS_DIR)

# Multiprocessing variables
from matgraphdb.utils.multiprocess_utils import multiprocess_task

# Other important variables
from matgraphdb.utils.log_config import setup_logging, get_child_logger, get_logger
from matgraphdb.utils.timing import Timer, timeit

# Initialize logger
LOGGER = setup_logging()
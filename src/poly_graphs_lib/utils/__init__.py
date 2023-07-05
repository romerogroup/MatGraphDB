
from typing import Union

import os
import logging.config
from pathlib import Path
import yaml
import numpy as np

# numpy options
large_width = 400
np.set_printoptions(linewidth=large_width)

# Other Constants
FILE = Path(__file__).resolve()

PROJECT_DIR = str(FILE.parents[3])  # Graph Network Project
SRC_DIR = str(FILE.parents[2])  # src
ROOT = str(FILE.parents[1])  # poly_graph_lib

LOGGING_NAME = 'poly_graphs'

VERBOSE = str(os.getenv('poly_graphs_VERBOSE', True)).lower() == 'true'  # global verbose mode


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})
    
# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)


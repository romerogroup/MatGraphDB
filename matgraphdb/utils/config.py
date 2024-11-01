import os
import yaml
import re
import logging

import logging.config

from pathlib import Path
import numpy as np
from copy import deepcopy

from variconfig import LoggingConfig

FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])
UTILS_DIR = str(FILE.parents[0])


config = LoggingConfig.from_yaml(os.path.join(UTILS_DIR, 'config.yml'))


if config.log_dir:
    os.makedirs(config.log_dir, exist_ok=True)
if config.data_dir:
    os.makedirs(config.data_dir, exist_ok=True)

np.set_printoptions(**config.numpy_config.np_printoptions.to_dict())

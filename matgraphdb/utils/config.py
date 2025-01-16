import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from variconfig import LoggingConfig

load_dotenv()

FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])
UTILS_DIR = str(FILE.parents[0])
DATA_DIR = os.getenv("DATA_DIR")

config = LoggingConfig.from_yaml(os.path.join(UTILS_DIR, "config.yml"))

if DATA_DIR:
    config.data_dir = DATA_DIR


# if config.log_dir:
#     os.makedirs(config.log_dir, exist_ok=True)
# if config.data_dir:
#     os.makedirs(config.data_dir, exist_ok=True)

np.set_printoptions(**config.numpy_config.np_printoptions.to_dict())

import os
import warnings
import logging
from glob import glob

import pandas as pd
import numpy as np
from matgraphdb.stores.node_store import NodeStore
from matgraphdb import PKG_DIR

logger = logging.getLogger(__name__)

BASE_CHEMENV_FILE = os.path.join(PKG_DIR, 'utils', 'chem_utils', 'resources','coordination_geometries.parquet')

class ChemEnvNodes(NodeStore):
    def __init__(self, storage_path: str, base_file=BASE_CHEMENV_FILE):
        super().__init__(storage_path=storage_path, initialize_kwargs={'base_file':base_file})
        
    def initialize(self, base_file=BASE_CHEMENV_FILE):
        """
        Creates ChemEnv nodes if no file exists, otherwise loads them from a file.
        """
        # Get the chemical environment names from a dictionary (mp_coord_encoding)
        self.name_column = 'mp_symbol'
        try:
            file_ext = os.path.splitext(base_file)[-1][1:]
            logger.debug(f"File extension: {file_ext}")
            if file_ext == 'parquet':
                df = pd.read_parquet(os.path.join(PKG_DIR, 'utils', base_file))
            elif file_ext == 'csv':
                df = pd.read_csv(os.path.join(PKG_DIR, 'utils', base_file), index_col=0)
            else:
                raise ValueError(f"base_file must be a parquet or csv file")
            logger.debug(f"Read element dataframe shape {df.shape}")
            
            df.drop(columns=['_algorithms'], inplace=True)
            
        except Exception as e:
            logger.error(f"Error creating chemical environment nodes: {e}")
            return None

        return df

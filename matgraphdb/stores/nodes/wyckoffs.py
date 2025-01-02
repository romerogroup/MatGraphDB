import os
import warnings
import logging
from glob import glob

import pandas as pd
import numpy as np
from matgraphdb.stores.node_store import NodeStore
from matgraphdb import PKG_DIR

logger = logging.getLogger(__name__)

class WyckoffNodes(NodeStore):
        
    def initialize(self):
        """"
        Creates Wyckoff Position nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = 'spg_wyckoff'
        # Generate space group names from 1 to 230
        try:
            space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
            # Define Wyckoff letters
            wyckoff_letters = ['a', 'b', 'c', 'd', 'e', 'f']

            # Create a list of space group-Wyckoff position combinations
            spg_wyckoffs = [f"{spg}_{wyckoff_letter}" for wyckoff_letter in wyckoff_letters for spg in space_groups]

            # Create a list of dictionaries with 'spg_wyckoff'
            spg_wyckoff_properties = [{"spg_wyckoff": spg_wyckoff} for spg_wyckoff in spg_wyckoffs]

            # Create DataFrame with Wyckoff positions
            df = pd.DataFrame(spg_wyckoff_properties)
        except Exception as e:
            logger.error(f"Error creating Wyckoff position nodes: {e}")
            return None

        return df

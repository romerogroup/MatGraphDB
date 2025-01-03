import os
import warnings
import logging
from glob import glob

import pandas as pd
import numpy as np
from matgraphdb.stores.node_store import NodeStore
from matgraphdb import PKG_DIR

logger = logging.getLogger(__name__)

class OxidationStatesNodes(NodeStore):
        
    def initialize(self):
        """
        Creates Oxidation State nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = 'oxidation_state'
        try:
            oxidation_states = np.arange(-9, 10)
            oxidation_states_names = [f'OxidationState{i}' for i in oxidation_states]
            data={
                'oxidation_state': oxidation_states_names,
                'value': oxidation_states
            }
            df = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error creating oxidation state nodes: {e}")
            return None
        return df

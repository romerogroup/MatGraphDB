import logging
import os
import warnings
from glob import glob

import pandas as pd

from matgraphdb.core import NodeStore

logger = logging.getLogger(__name__)


class MagneticStatesNodes(NodeStore):

    def initialize(self):
        """
        Creates Magnetic State nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = "magnetic_state"
        # Define magnetic states
        try:
            magnetic_states = ["NM", "FM", "FiM", "AFM", "Unknown"]
            magnetic_states_properties = [
                {"magnetic_state": ms} for ms in magnetic_states
            ]
            df = pd.DataFrame(magnetic_states_properties)
        except Exception as e:
            logger.error(f"Error creating magnetic state nodes: {e}")
            return None
        return df

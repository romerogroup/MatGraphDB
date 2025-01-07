import logging
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd

from matgraphdb.core import NodeStore

logger = logging.getLogger(__name__)


class SpaceGroupNodes(NodeStore):

    def initialize(self):
        """
        Creates Space Group nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = "spg"
        # Generate space group numbers from 1 to 230
        try:
            space_groups = [f"spg_{i}" for i in np.arange(1, 231)]
            space_groups_properties = [
                {"spg": int(space_group.split("_")[1])} for space_group in space_groups
            ]

            # Create DataFrame with the space group properties
            df = pd.DataFrame(space_groups_properties)
        except Exception as e:
            logger.error(f"Error creating space group nodes: {e}")
            return None

        return df

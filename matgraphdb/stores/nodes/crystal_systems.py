import os
import warnings
import logging
from glob import glob

import pandas as pd
from matgraphdb.stores.node_store import NodeStore
from matgraphdb import PKG_DIR

logger = logging.getLogger(__name__)

class CrystalSystemNodes(NodeStore):
        
    def initialize(self):
        """
        Creates Crystal System nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = 'crystal_system'
        
        logger.info(f"Initializing CrystalSystemNodes with storage path: {self.storage_path}")
        try:
            crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
            crystal_systems_properties = [{"crystal_system": cs} for cs in crystal_systems]
            df = pd.DataFrame(crystal_systems_properties)
        except Exception as e:
            logger.error(f"Error creating crystal system nodes: {e}")
            return None

        return df

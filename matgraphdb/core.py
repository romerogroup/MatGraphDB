import os
from typing import List, Dict, Union
import pyarrow as pa
import logging

from matgraphdb.stores import GraphStore, MaterialStore

logger = logging.getLogger(__name__)

class MatGraphDB(GraphStore):
    """
    The main entry point for advanced material analysis and graph storage.
    
    It uses:
      - GraphStore for general node/edge operations.
      - MaterialStore for specialized 'material' node data.
    """

    def __init__(self, storage_path: str):
        """
        Parameters
        ----------
        storage_path : str
            The root directory for the entire MatGraphDB (nodes, edges, materials, etc.).
        """
        self.storage_path = os.path.abspath(storage_path)
        logger.info(f"Initializing MatGraphDB at: {self.storage_path}")

        self.materials_path = os.path.join(self.nodes_path, "material")
        self.add_node_store(MaterialStore(storage_path=self.materials_path))
        self.material_store = self.node_stores["material"]
        
    def create_materials(self, data, **kwargs):
        logger.info("Creating materials.")
        self.material_store.create_materials(data, **kwargs)

    def read_materials(self, ids: List[int] = None, columns: List[str] = None, **kwargs):
        logger.info("Reading materials.")
        return self.material_store.read_materials(ids=ids, columns=columns, **kwargs)

    def update_materials(self, data, **kwargs):
        logger.info("Updating materials.")
        self.material_store.update_materials(data, **kwargs)

    def delete_materials(self, ids: List[int] = None, columns: List[str] = None):
        logger.info("Deleting materials.")
        self.material_store.delete_materials(ids=ids, columns=columns)

    def normalize_materials(self):
        logger.info("Normalizing materials store.")
        self.material_store.normalize_materials()
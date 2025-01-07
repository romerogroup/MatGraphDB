import logging
import os
from typing import Dict, List, Union

import pyarrow as pa

from matgraphdb.core import GraphDB
from matgraphdb.materials.nodes import MaterialNodes

logger = logging.getLogger(__name__)


class MatGraphDB(GraphDB):
    """
    The main entry point for advanced material analysis and graph storage.

    It uses:
      - GraphStore for general node/edge operations.
      - MaterialStore for specialized 'material' node data.
    """

    def __init__(self, storage_path: str, load_custom_stores: bool = True):
        """
        Parameters
        ----------
        storage_path : str
            The root directory for the entire MatGraphDB (nodes, edges, materials, etc.).
        """
        self.storage_path = os.path.abspath(storage_path)
        super().__init__(
            storage_path=self.storage_path, load_custom_stores=load_custom_stores
        )
        logger.info(f"Initializing MatGraphDB at: {self.storage_path}")

        if not self.node_exists("materials"):
            self.materials_path = os.path.join(self.nodes_path, "materials")
            logger.info(
                "Material nodes do not exist. Adding empty materials node store"
            )
            self.add_node_store(MaterialNodes(storage_path=self.materials_path))
        self.material_nodes = self.node_stores["materials"]

    def create_materials(self, data, **kwargs):
        logger.info("Creating materials.")
        self.material_nodes.create_materials(data, **kwargs)

    def read_materials(
        self, ids: List[int] = None, columns: List[str] = None, **kwargs
    ):
        logger.info("Reading materials.")
        return self.material_nodes.read_materials(ids=ids, columns=columns, **kwargs)

    def update_materials(self, data, **kwargs):
        logger.info("Updating materials.")
        self.material_nodes.update_materials(data, **kwargs)

    def delete_materials(self, ids: List[int] = None, columns: List[str] = None):
        logger.info("Deleting materials.")
        self.material_nodes.delete_materials(ids=ids, columns=columns)

    def normalize_materials(self):
        logger.info("Normalizing materials store.")
        self.material_nodes.normalize_materials()

import logging
import os
from typing import Dict, List, Union

import pyarrow as pa

from matgraphdb.core import GraphDB
from matgraphdb.materials.nodes import MaterialStore

logger = logging.getLogger(__name__)


class MatGraphDB(GraphDB):
    """
    The main entry point for advanced material analysis and graph storage.

    It uses:
      - GraphStore for general node/edge operations.
      - MaterialStore for specialized 'material' node data.
    """

    def __init__(
        self,
        storage_path: str,
        materials_store: MaterialStore = None,
        load_custom_stores: bool = True,
    ):
        """
        Parameters
        ----------
        storage_path : str
            The root directory for the entire MatGraphDB (nodes, edges, materials, etc.).
        materials_store : MaterialsStore
            The materials store to use. If None, a new materials store will be created in the storage_path.
        load_custom_stores : bool
            Whether to load custom stores.
        """
        self.storage_path = os.path.abspath(storage_path)
        super().__init__(
            storage_path=self.storage_path, load_custom_stores=load_custom_stores
        )
        logger.info(f"Initializing MatGraphDB at: {self.storage_path}")

        self.materials_path = os.path.join(self.nodes_path, "materials")

        if not self.node_exists("materials"):
            logger.info(
                "Material nodes do not exist. Adding empty materials node store"
            )
            if materials_store is None:
                self.add_node_store(MaterialStore(storage_path=self.materials_path))
            else:
                self.add_node_store(materials_store)
        self.material_store = self.node_stores["materials"]

    def create_material(self, **kwargs):
        logger.info("Creating material.")
        self.material_store.create_material(**kwargs)
        self._run_dependent_generators("materials")

    def create_materials(self, data, **kwargs):
        logger.info("Creating materials.")
        self.material_store.create_materials(data, **kwargs)
        self._run_dependent_generators("materials")

    def read_materials(
        self, ids: List[int] = None, columns: List[str] = None, **kwargs
    ):
        logger.info("Reading materials.")
        return self.material_store.read_materials(ids=ids, columns=columns, **kwargs)

    def update_materials(self, data, **kwargs):
        logger.info("Updating materials.")
        self.material_store.update_materials(data, **kwargs)
        self._run_dependent_generators("materials")

    def delete_materials(self, ids: List[int] = None, columns: List[str] = None):
        logger.info("Deleting materials.")
        self.material_store.delete_materials(ids=ids, columns=columns)
        self._run_dependent_generators("materials")

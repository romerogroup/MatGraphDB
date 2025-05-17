import logging
import os
from typing import List

from parquetdb import ParquetGraphDB

from matgraphdb.core.material_store import MaterialStore

logger = logging.getLogger(__name__)


class MatGraphDB(ParquetGraphDB):
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
        **kwargs,
    ):
        """
        Parameters
        ----------
        storage_path : str
            The root directory for the entire MatGraphDB (nodes, edges, materials, etc.).
        materials_store : MaterialsStore
            The materials store to use. If None, a new materials store will be created in the storage_path.
        kwargs : dict
            Additional keyword arguments to pass to the ParquetGraphDB constructor.
        """
        self.storage_path = os.path.abspath(storage_path)
        super().__init__(
            storage_path=self.storage_path,
            **kwargs,
        )
        logger.info(f"Initializing MatGraphDB at: {self.storage_path}")

        self.materials_path = os.path.join(self.nodes_path, "material")

        if not self.node_exists("material"):
            logger.info(
                "Material nodes do not exist. Adding empty materials node store"
            )
            if materials_store is None:
                self.add_node_store(MaterialStore(storage_path=self.materials_path))
            else:
                self.add_node_store(materials_store)
        self.material_store = self.node_stores.get("material", None)
        if self.material_store is None:
            raise ValueError("Material node store not found.")

    def create_material(self, **kwargs):
        logger.info("Creating material.")
        self.material_store.create_material(**kwargs)
        self._run_dependent_generators("material")

    def create_materials(self, data, **kwargs):
        logger.info("Creating materials.")
        self.material_store.create_materials(data, **kwargs)
        self._run_dependent_generators("material")

    def read_materials(
        self, ids: List[int] = None, columns: List[str] = None, **kwargs
    ):
        logger.info("Reading materials.")
        return self.material_store.read_materials(ids=ids, columns=columns, **kwargs)

    def update_materials(self, data, **kwargs):
        logger.info("Updating materials.")
        self.material_store.update_materials(data, **kwargs)
        self._run_dependent_generators("material")

    def delete_materials(self, ids: List[int] = None, columns: List[str] = None):
        logger.info("Deleting materials.")
        self.material_store.delete_materials(ids=ids, columns=columns)
        self._run_dependent_generators("material")

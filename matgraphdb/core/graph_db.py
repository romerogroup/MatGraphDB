import importlib
import logging
import os
import shutil
import time
from glob import glob
from typing import Callable, Dict, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from pyarrow import parquet as pq

from matgraphdb.core.edge_store import EdgeStore
from matgraphdb.core.generator_store import GeneratorStore
from matgraphdb.core.node_store import NodeStore

logger = logging.getLogger(__name__)


class GraphDB:
    """
    A manager for a graph storing multiple node types and edge types.
    Each node type and edge type is backed by a separate ParquetDB instance
    (wrapped by NodeStore or EdgeStore).
    """

    def __init__(self, storage_path: str, load_custom_stores: bool = True):
        """
        Parameters
        ----------
        storage_path : str
            The root path for this graph, e.g. '/path/to/my_graph'.
            Subdirectories 'nodes/' and 'edges/' will be used.
        """
        logger.info(f"Initializing GraphDB at root path: {storage_path}")
        self.storage_path = os.path.abspath(storage_path)

        self.nodes_path = os.path.join(self.storage_path, "nodes")
        self.edges_path = os.path.join(self.storage_path, "edges")
        self.edge_generators_path = os.path.join(self.storage_path, "edge_generators")
        self.graph_path = os.path.join(self.storage_path, "graph")

        self.graph_name = os.path.basename(self.storage_path)

        # Create directories if they don't exist
        os.makedirs(self.nodes_path, exist_ok=True)
        os.makedirs(self.edges_path, exist_ok=True)
        os.makedirs(self.edge_generators_path, exist_ok=True)
        os.makedirs(self.graph_path, exist_ok=True)

        logger.debug(f"Node directory: {self.nodes_path}")
        logger.debug(f"Edge directory: {self.edges_path}")
        logger.debug(f"Graph directory: {self.graph_path}")

        #  Initialize empty dictionaries for stores, load existing stores
        self.node_stores = self._load_existing_node_stores(load_custom_stores)
        self.edge_stores = self._load_existing_edge_stores(load_custom_stores)

        self.edge_generator_store = GeneratorStore(
            storage_path=self.edge_generators_path
        )

    def _load_existing_node_stores(self, load_custom_stores: bool = True):
        logger.info(f"Loading existing node stores")
        return self._load_existing_stores(
            self.nodes_path,
            default_store_class=NodeStore,
            load_custom_stores=load_custom_stores,
        )

    def _load_existing_edge_stores(self, load_custom_stores: bool = True):
        logger.info(f"Loading existing edge stores")
        return self._load_existing_stores(
            self.edges_path,
            default_store_class=EdgeStore,
            load_custom_stores=load_custom_stores,
        )

    def _load_existing_stores(
        self,
        stores_path,
        default_store_class: Union[NodeStore, EdgeStore] = None,
        load_custom_stores: bool = True,
    ):

        if load_custom_stores:
            default_store_class = None

        logger.debug(f"Load custom stores: {load_custom_stores}")

        store_dict = {}
        store_types = os.listdir(stores_path)
        logger.info(f"Found {len(store_types)} store types")
        for store_type in store_types:
            logger.debug(f"Attempting to load store: {store_type}")

            store_path = os.path.join(stores_path, store_type)
            if os.path.isdir(store_path):
                store_dict[store_type] = load_store(store_path, default_store_class)
            else:
                raise ValueError(
                    f"Store path {store_path} is not a directory. Likely does not exist."
                )

        return store_dict

    # ------------------
    # Node-level methods
    # ------------------
    def add_nodes(self, node_type: str, data, **kwargs):
        logger.info(f"Creating nodes of type '{node_type}'")
        store = self.add_node_type(node_type)
        store.create_nodes(data, **kwargs)
        logger.debug(f"Successfully created nodes of type '{node_type}'")

    def add_node_type(self, node_type: str) -> NodeStore:
        """
        Create (or load) a NodeStore for the specified node_type.
        """
        if node_type in self.node_stores:
            logger.debug(f"Returning existing NodeStore for type: {node_type}")
            return self.node_stores[node_type]

        logger.info(f"Creating new NodeStore for type: {node_type}")
        storage_path = os.path.join(self.nodes_path, node_type)
        self.node_stores[node_type] = NodeStore(storage_path=storage_path)
        return self.node_stores[node_type]

    def add_node_store(
        self,
        node_store: NodeStore,
        overwrite: bool = False,
        remove_original: bool = False,
    ):
        logger.info(f"Adding node store of type {node_store.node_type}")

        # Check if node store already exists
        if node_store.node_type in self.node_stores:
            if overwrite:
                logger.warning(
                    f"Node store of type {node_store.node_type} already exists, overwriting"
                )
                self.remove_node_store(node_store.node_type)
            else:
                raise ValueError(
                    f"Node store of type {node_store.node_type} already exists, and overwrite is False"
                )

        # Move node store to the nodes directory
        new_path = os.path.join(self.nodes_path, node_store.node_type)
        if node_store.storage_path != new_path:
            logger.debug(
                f"Moving node store from {node_store.storage_path} to {new_path}"
            )
            shutil.copytree(node_store.storage_path, new_path)

            if remove_original:
                shutil.rmtree(node_store.storage_path)
            node_store.storage_path = new_path
        self.node_stores[node_store.node_type] = node_store

    def get_nodes(
        self, node_type: str, ids: List[int] = None, columns: List[str] = None, **kwargs
    ):
        logger.info(f"Reading nodes of type '{node_type}'")
        if ids:
            logger.debug(f"Filtering by {len(ids)} node IDs")
        if columns:
            logger.debug(f"Selecting columns: {columns}")
        store = self.get_node_store(node_type)
        return store.read_nodes(ids=ids, columns=columns, **kwargs)

    def get_node_store(self, node_type: str):
        # if node_type not in self.node_stores:
        node_store = self.node_stores.get(node_type, None)
        if node_store is None:
            raise ValueError(f"Node store of type {node_type} does not exist")
        return node_store

    def update_nodes(self, node_type: str, data, **kwargs):
        store = self.get_node_store(node_type)
        store.update_nodes(data, **kwargs)

    def delete_nodes(
        self, node_type: str, ids: List[int] = None, columns: List[str] = None
    ):
        store = self.get_node_store(node_type)
        store.delete_nodes(ids=ids, columns=columns)

    def remove_node_store(self, node_type: str):
        logger.info(f"Removing node store of type {node_type}")
        store = self.get_node_store(node_type)
        shutil.rmtree(store.storage_path)
        self.node_stores.pop(node_type)

    def remove_node_type(self, node_type: str):
        self.remove_node_store(node_type)

    def normalize_nodes(self, node_type: str, normalize_kwargs: Dict = None):
        store = self.add_node_type(node_type)
        store.normalize_nodes(**normalize_kwargs)

    def list_node_types(self):
        return list(self.node_stores.keys())

    def node_exists(self, node_type: str):
        logger.debug(f"Node type: {node_type}")
        logger.debug(f"Node stores: {self.node_stores}")

        return node_type in self.node_stores

    def node_is_empty(self, node_type: str):
        store = self.get_node_store(node_type)
        return store.is_empty()

    # ------------------
    # Edge-level methods
    # ------------------
    def add_edge_type(self, edge_type: str) -> EdgeStore:
        """
        Create (or load) an EdgeStore for the specified edge_type.
        """
        if edge_type in self.edge_stores:
            logger.debug(f"Returning existing EdgeStore for type: {edge_type}")
            return self.edge_stores[edge_type]

        logger.info(f"Creating new EdgeStore for type: {edge_type}")
        storage_path = os.path.join(self.edges_path, edge_type)
        self.edge_stores[edge_type] = EdgeStore(storage_path=storage_path)
        return self.edge_stores[edge_type]

    def add_edges(self, edge_type: str, data, **kwargs):
        logger.info(f"Creating edges of type '{edge_type}'")
        incoming_table = ParquetDB.construct_table(data)
        self._validate_edge_references(incoming_table)
        store = self.add_edge_type(edge_type)
        store.create_edges(incoming_table, **kwargs)
        logger.debug(f"Successfully created edges of type '{edge_type}'")

    def add_edge_store(self, edge_store: EdgeStore):
        logger.info(f"Adding edge store of type {edge_store.edge_type}")

        # Move edge store to the edges directory
        new_path = os.path.join(self.edges_path, edge_store.edge_type)
        if edge_store.storage_path != new_path:
            logger.debug(
                f"Moving edge store from {edge_store.storage_path} to {new_path}"
            )
            os.makedirs(new_path, exist_ok=True)
            for file in glob(os.path.join(edge_store.storage_path, "*")):
                new_file = os.path.join(new_path, os.path.basename(file))
                os.rename(file, new_file)
            edge_store.storage_path = new_path
        self.edge_stores[edge_store.edge_type] = edge_store

    def read_edges(
        self, edge_type: str, ids: List[int] = None, columns: List[str] = None, **kwargs
    ):
        store = self.add_edge_type(edge_type)
        return store.read_edges(ids=ids, columns=columns, **kwargs)

    def update_edges(self, edge_type: str, data, **kwargs):
        store = self.add_edge_type(edge_type)
        store.update_edges(data, **kwargs)

    def delete_edges(
        self, edge_type: str, ids: List[int] = None, columns: List[str] = None
    ):
        store = self.add_edge_type(edge_type)
        store.delete_edges(ids=ids, columns=columns)

    def remove_edge_store(self, edge_type: str):
        logger.info(f"Removing edge store of type {edge_type}")
        store = self.get_edge_store(edge_type)
        shutil.rmtree(store.storage_path)
        self.edge_stores.pop(edge_type)

    def remove_edge_type(self, edge_type: str):
        self.remove_edge_store(edge_type)

    def normalize_edges(self, edge_type: str):
        store = self.add_edge_type(edge_type)
        store.normalize_edges()

    def get_edge_store(self, edge_type: str):
        edge_store = self.edge_stores.get(edge_type, None)
        if edge_store is None:
            raise ValueError(f"Edge store of type {edge_type} does not exist")
        return edge_store

    def list_edge_types(self):
        return list(self.edge_stores.keys())

    def edge_exists(self, edge_type: str):
        return edge_type in self.edge_stores

    def edge_is_empty(self, edge_type: str):
        store = self.get_edge_store(edge_type)
        return store.is_empty()

    def _validate_edge_references(self, table: pa.Table) -> None:
        """
        Checks whether source_id and target_id in each edge record exist
        in the corresponding node stores.

        Parameters
        ----------
        table : pa.Table
            A table containing 'source_id' and 'target_id' columns.
        source_node_type : str
            The node type for the source nodes (e.g., 'user').
        target_node_type : str
            The node type for the target nodes (e.g., 'item').

        Raises
        ------
        ValueError
            If any source_id/target_id is not found in the corresponding node store.
        """
        # logger.debug(f"Validating edge references: {source_node_type} -> {target_node_type}")
        edge_table = table
        # 1. Retrieve the NodeStores
        names = edge_table.column_names
        logger.debug(f"Column names: {names}")

        assert "source_type" in names, "source_type column not found in table"
        assert "target_type" in names, "target_type column not found in table"
        assert "source_id" in names, "source_id column not found in table"
        assert "target_id" in names, "target_id column not found in table"

        node_types = pc.unique(table["source_type"]).to_pylist()

        for node_type in node_types:
            store = self.node_stores.get(node_type, None)
            if store is None:
                logger.error(f"No node store found for node_type='{node_type}'")
                raise ValueError(f"No node store found for node_type='{node_type}'.")

            # Read all existing source IDs from store_1
            source_table = store.read_nodes(columns=["id"])

            # Filter all source_ids and target_ids that are of the same type as store_1
            source_id_array = edge_table.filter(
                pc.field("source_type") == store.node_type
            )["source_id"].combine_chunks()
            target_id_array = edge_table.filter(
                pc.field("target_type") == store.node_type
            )["target_id"].combine_chunks()

            all_source_type_ids = pa.concat_arrays([source_id_array, target_id_array])
            is_source_ids_in_source_store = pc.index_in(
                source_table["id"], all_source_type_ids
            )
            invalid_source_ids = is_source_ids_in_source_store.filter(
                pc.is_null(is_source_ids_in_source_store)
            )
            if len(invalid_source_ids) > 0:
                raise ValueError(
                    f"Source IDs not found in source_store of type {store.node_type}: {invalid_source_ids}"
                )

        logger.debug("Edge reference validation completed successfully")

    def construct_table(self, data, schema=None, metadata=None, fields_metadata=None):
        logger.info("Validating data")
        return ParquetDB.construct_table(
            data, schema=schema, metadata=metadata, fields_metadata=fields_metadata
        )

    def add_edge_generator(
        self,
        name: str,
        generator_func: Callable,
        generator_args: Dict = None,
        generator_kwargs: Dict = None,
        create_kwargs: Dict = None,
    ) -> None:
        """
        Register a user-defined callable that can read from node stores,
        and then create or update edges as it sees fit.

        Parameters
        ----------
        name : str
            A unique identifier for this generator function.
        generator_func : Callable
            A Python callable with the signature:
               generator_func(graph_db: GraphDB, *args, **kwargs) -> None
            The function is expected to do any of the following:
            - read from node stores
            - call `self.add_edges(...)`
            - or call `self.update_edges(...)`
            - etc.
        """
        self.edge_generator_store.store_generator(
            generator_func=generator_func,
            generator_name=name,
            generator_args=generator_args,
            generator_kwargs=generator_kwargs,
            create_kwargs=create_kwargs,
        )
        logger.info(f"Added new edge generator: {name}")

    def run_edge_generator(
        self,
        name: str,
        generator_args: Dict = None,
        generator_kwargs: Dict = None,
        create_kwargs: Dict = None,
    ) -> None:
        """
        Execute a previously registered custom edge-generation function by name.

        Parameters
        ----------
        name : str
            The unique name used when registering the function.
        generator_args : Dict
            Additional arguments passed to the generator function.
        generator_kwargs : Dict
            Additional keyword arguments passed to the generator function.

        Raises
        ------
        ValueError
            If there is no generator function with the given name.
        """
        if create_kwargs is None:
            create_kwargs = {}

        table = self.edge_generator_store.run_generator(
            name, generator_args=generator_args, generator_kwargs=generator_kwargs
        )
        self.add_edges(edge_type=name, data=table, **create_kwargs)
        return table


def load_store(store_path: str, default_store_class=None):
    store_metadata = ParquetDB(store_path).get_metadata()
    class_module = store_metadata.get("class_module", None)
    class_name = store_metadata.get("class", None)

    logger.debug(f"Class module: {class_module}")
    logger.debug(f"Class: {class_name}")

    if class_module and class_name and default_store_class is None:
        logger.debug(f"Importing class from module: {class_module}")
        module = importlib.import_module(class_module)
        class_obj = getattr(module, class_name)
        store = class_obj(storage_path=store_path)
    else:
        logger.debug(f"Using default store class: {default_store_class.__name__}")
        store = default_store_class(storage_path=store_path)

    return store

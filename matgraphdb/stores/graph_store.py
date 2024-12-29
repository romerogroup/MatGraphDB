import os
import logging
from typing import List

import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

from matgraphdb.stores.node_store import NodeStore
from matgraphdb.stores.edge_store import EdgeStore


logger = logging.getLogger(__name__)

class GraphStore:
    """
    A manager for a graph storing multiple node types and edge types.
    Each node type and edge type is backed by a separate ParquetDB instance
    (wrapped by NodeStore or EdgeStore).
    """
    def __init__(self, root_path: str):
        """
        Parameters
        ----------
        root_path : str
            The root path for this graph, e.g. '/path/to/my_graph'.
            Subdirectories 'nodes/' and 'edges/' will be used.
        """
        logger.info(f"Initializing GraphStore at root path: {root_path}")
        self.root_path = os.path.abspath(root_path)
        
        self.nodes_path = os.path.join(self.root_path, 'nodes')
        self.edges_path = os.path.join(self.root_path, 'edges')
        
        self.graph_name = os.path.basename(self.root_path)
        
        # You might track node/edge stores by their types in dictionaries
        self.node_stores = {}
        self.edge_stores = {}

        os.makedirs(self.nodes_path, exist_ok=True)
        os.makedirs(self.edges_path, exist_ok=True)

        logger.debug(f"Created node directory at: {self.nodes_path}")
        logger.debug(f"Created edge directory at: {self.edges_path}")

    def add_node_type(self, node_type: str) -> NodeStore:
        """
        Create (or load) a NodeStore for the specified node_type.
        """
        if node_type in self.node_stores:
            logger.debug(f"Returning existing NodeStore for type: {node_type}")
            return self.node_stores[node_type]
        
        logger.info(f"Creating new NodeStore for type: {node_type}")
        storage_path = os.path.join(self.nodes_path, node_type)
        store = NodeStore(storage_path=storage_path)
        self.node_stores[node_type] = store
        return store

    def add_edge_type(self, edge_type: str) -> EdgeStore:
        """
        Create (or load) an EdgeStore for the specified edge_type.
        """
        if edge_type in self.edge_stores:
            logger.debug(f"Returning existing EdgeStore for type: {edge_type}")
            return self.edge_stores[edge_type]
        
        logger.info(f"Creating new EdgeStore for type: {edge_type}")
        storage_path = os.path.join(self.edges_path, edge_type)
        store = EdgeStore(storage_path=storage_path)
        self.edge_stores[edge_type] = store
        return store
    
    # ------------------
    # Node-level methods
    # ------------------
    def create_nodes(self, node_type: str, data, **kwargs):
        logger.info(f"Creating nodes of type '{node_type}'")
        store = self.add_node_type(node_type)
        store.create_nodes(data, **kwargs)
        logger.debug(f"Successfully created nodes of type '{node_type}'")

    def read_nodes(self, node_type: str, ids: List[int] = None, columns: List[str] = None, **kwargs):
        logger.info(f"Reading nodes of type '{node_type}'")
        if ids:
            logger.debug(f"Filtering by {len(ids)} node IDs")
        if columns:
            logger.debug(f"Selecting columns: {columns}")
        store = self.add_node_type(node_type)
        return store.read_nodes(ids=ids, columns=columns, **kwargs)

    def update_nodes(self, node_type: str, data, **kwargs):
        store = self.add_node_type(node_type)
        store.update_nodes(data, **kwargs)

    def delete_nodes(self, node_type: str, ids: List[int] = None, columns: List[str] = None):
        store = self.add_node_type(node_type)
        store.delete_nodes(ids=ids, columns=columns)

    def normalize_nodes(self, node_type: str):
        store = self.add_node_type(node_type)
        store.normalize_nodes()

    # ------------------
    # Edge-level methods
    # ------------------
    def create_edges(self, edge_type: str, data, **kwargs):
        logger.info(f"Creating edges of type '{edge_type}'")
        incoming_table = self._construct_table(data)
        
        source_node_type = incoming_table['source_type'].combine_chunks()[0].as_py()
        target_node_type = incoming_table['target_type'].combine_chunks()[0].as_py()
        
        logger.debug(f"Validating edge references between {source_node_type} and {target_node_type}")
        self._validate_edge_references(incoming_table, source_node_type, target_node_type)
        store = self.add_edge_type(edge_type)
        store.create_edges(incoming_table, **kwargs)
        logger.debug(f"Successfully created edges of type '{edge_type}'")

    def read_edges(self, edge_type: str, ids: List[int] = None, columns: List[str] = None, **kwargs):
        store = self.add_edge_type(edge_type)
        return store.read_edges(ids=ids, columns=columns, **kwargs)

    def update_edges(self, edge_type: str, data, **kwargs):
        store = self.add_edge_type(edge_type)
        store.update_edges(data, **kwargs)

    def delete_edges(self, edge_type: str, ids: List[int] = None, columns: List[str] = None):
        store = self.add_edge_type(edge_type)
        store.delete_edges(ids=ids, columns=columns)

    def normalize_edges(self, edge_type: str):
        store = self.add_edge_type(edge_type)
        store.normalize_edges()
        
    def _validate_edge_references(
        self,
        table: pa.Table,
        source_node_type: str,
        target_node_type: str
    ) -> None:
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
        logger.debug(f"Validating edge references: {source_node_type} -> {target_node_type}")
        
        # 1. Retrieve the NodeStores
        source_store = self.node_stores.get(source_node_type, None)
        target_store = self.node_stores.get(target_node_type, None)

        if source_store is None:
            logger.error(f"No node store found for source_node_type='{source_node_type}'")
            raise ValueError(f"No node store found for source_node_type='{source_node_type}'.")
        if target_store is None:
            logger.error(f"No node store found for target_node_type='{target_node_type}'")
            raise ValueError(f"No node store found for target_node_type='{target_node_type}'.")

        # 2. Read all existing source IDs from source_store
        source_table = source_store.read_nodes(columns=["id"])
        is_source_ids_in_source_store = pc.index_in(source_table['id'], table['source_id'])
        invalid_source_ids = is_source_ids_in_source_store.filter(pc.is_null(is_source_ids_in_source_store)).combine_chunks()
        if len(invalid_source_ids) > 0:
            raise ValueError(f"Source IDs not found in source_store of type {source_node_type}: {invalid_source_ids}")
        

        # 3. Read all existing target IDs from target_store
        target_table = target_store.read_nodes(columns=["id"])
        is_target_ids_in_target_store = pc.index_in(target_table['id'], table['target_id'])
        invalid_target_ids = is_target_ids_in_target_store.filter(pc.is_null(is_target_ids_in_target_store)).combine_chunks()
        if len(invalid_target_ids) > 0:
            raise ValueError(f"Target IDs not found in target_store of type {target_node_type}: {invalid_target_ids}")
        
        logger.debug("Edge reference validation completed successfully")
        
    def _construct_table(self, data, schema=None, metadata=None):
            logger.info("Validating data")
            if isinstance(data, dict):
                logger.info("The incoming data is a dictonary of arrays")
                for key, value in data.items():
                    if not isinstance(value, List):
                        data[key]=[value]
                table=pa.Table.from_pydict(data)
                incoming_array=table.to_struct_array()
                incoming_array=incoming_array.flatten()
                incoming_schema=table.schema
                
            elif isinstance(data, list):
                logger.info("Incoming data is a list of dictionaries")
                # Convert to pyarrow array to get the schema. This method is faster than .from_pylist
                # As from_pylist iterates through record in a python loop, but pa.array handles this in C++/cython
                incoming_array=pa.array(data)
                incoming_schema=pa.schema(incoming_array.type)
                incoming_array=incoming_array.flatten()
                
            elif isinstance(data, pd.DataFrame):
                logger.info("Incoming data is a pandas dataframe")
                table=pa.Table.from_pandas(data)
                incoming_array=table.to_struct_array()
                incoming_array=incoming_array.flatten()
                incoming_schema=table.schema
                
            elif isinstance(data, pa.lib.Table):
                incoming_schema=data.schema
                incoming_array=data.to_struct_array()
                incoming_array=incoming_array.flatten()
                
            else:
                raise TypeError("Data must be a dictionary or a list of dictionaries.")
            
            if schema is None:
                schema=incoming_schema
            schema=schema.with_metadata(metadata)

            incoming_table=pa.Table.from_arrays(incoming_array,schema=schema)
            return incoming_table
        

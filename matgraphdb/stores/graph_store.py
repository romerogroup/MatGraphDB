import os
import logging
import shutil
import time
import importlib
from typing import List, Union
from glob import glob

import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
from pyarrow import parquet as pq


from matgraphdb.stores.node_store import NodeStore
from matgraphdb.stores.edge_store import EdgeStore
from matgraphdb.stores.utils import load_store

# from matgraphdb.stores.nodes import *
from parquetdb import ParquetDB

logger = logging.getLogger(__name__)

class GraphStore:
    """
    A manager for a graph storing multiple node types and edge types.
    Each node type and edge type is backed by a separate ParquetDB instance
    (wrapped by NodeStore or EdgeStore).
    """
    def __init__(self, storage_path: str, load_custom_stores: bool=True):
        """
        Parameters
        ----------
        storage_path : str
            The root path for this graph, e.g. '/path/to/my_graph'.
            Subdirectories 'nodes/' and 'edges/' will be used.
        """
        logger.info(f"Initializing GraphDB at root path: {storage_path}")
        self.storage_path = os.path.abspath(storage_path)
        
        self.nodes_path = os.path.join(self.storage_path, 'nodes')
        self.edges_path = os.path.join(self.storage_path, 'edges')
        self.graph_path = os.path.join(self.storage_path, 'graph')
            
        self.graph_name = os.path.basename(self.storage_path)
        
        # Create directories if they don't exist
        os.makedirs(self.nodes_path, exist_ok=True)
        os.makedirs(self.edges_path, exist_ok=True)

        logger.debug(f"Created/found node directory at: {self.nodes_path}")
        logger.debug(f"Created/found edge directory at: {self.edges_path}")
        logger.debug(f"Created/found graph directory at: {self.graph_path}")

        #  Initialize empty dictionaries for stores, load existing stores
        self.node_stores = self._load_existing_node_stores(load_custom_stores)
        self.edge_stores = self._load_existing_edge_stores(load_custom_stores)


    def _load_existing_node_stores(self, load_custom_stores: bool = True):
        return self._load_existing_stores(self.nodes_path, default_store_class=NodeStore, load_custom_stores=load_custom_stores)

    def _load_existing_edge_stores(self, load_custom_stores: bool = True):
        return self._load_existing_stores(self.edges_path, default_store_class=EdgeStore, load_custom_stores=load_custom_stores)
        
    # def _load_existing_stores(self, store_class, storage_path, load_custom_stores: bool = True):
    #     logger.info(f"Attempting to load {store_class.__name__}")
    #     logger.debug(f"Load custom stores: {load_custom_stores}")

    #     store_dict = {}
    #     for store_type in os.listdir(storage_path):
    #         store_path = os.path.join(storage_path, store_type)
    #         logger.debug(f"Store path: {store_path}")
    #         if os.path.isdir(store_path):
    #             logger.info(f"Loading existing store: {store_type}")
                
    #             store_metadata=ParquetDB(store_path).get_metadata()
    #             class_module = store_metadata.get('class_module', None)
    #             class_name = store_metadata.get('class', None)
                
    #             if class_module and class_name and load_custom_stores:
    #                 try:
    #                     logger.debug(f"Attempting to import class from module: {class_module}")
    #                     module = importlib.import_module(class_module)
    #                     class_obj = getattr(module, class_name)
    #                     store_dict[store_type] = class_obj(storage_path=store_path)
    #                 except Exception as e:
    #                     logger.error(f"Error initializing store {store_type}: {e}")
    #                     logger.debug(f"Falling back to base default store for {store_type}")
    #                     store_dict[store_type] = store_class(storage_path=store_path)
    #             else:
    #                 logger.debug(f"Initializing default store")
    #                 store_dict[store_type] = store_class(storage_path=store_path)
    #     return store_dict
    
    def _load_existing_stores(self, stores_path, default_store_class: Union[NodeStore, EdgeStore]=None, load_custom_stores: bool = True):
        
        if load_custom_stores:
            default_store_class=None

        logger.info(f"Loading store from {stores_path}")
        logger.debug(f"Load custom stores: {load_custom_stores}")
        
        store_dict = {}
        for store_type in os.listdir(stores_path):
            store_path = os.path.join(stores_path, store_type)
            logger.debug(f"Found store: {store_type}")
            if os.path.isdir(store_path):
                store_dict[store_type] = load_store(store_path, default_store_class)
            else:
                raise ValueError(f"Store path {store_path} is not a directory. Likely does not exist.")
        return store_dict
                    
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
    
    def add_node_store(self, node_store: NodeStore, overwrite: bool=False, remove_original: bool=False):
        logger.info(f"Adding node store of type {node_store.node_type}")
        
        # Check if node store already exists
        if node_store.node_type in self.node_stores:
            if overwrite:
                logger.warning(f"Node store of type {node_store.node_type} already exists, overwriting")
                self.remove_node_store(node_store.node_type)
            else:
                raise ValueError(f"Node store of type {node_store.node_type} already exists, and overwrite is False")
        
        
        # Move node store to the nodes directory
        new_path = os.path.join(self.nodes_path, node_store.node_type)
        if node_store.storage_path != new_path:
            logger.debug(f"Moving node store from {node_store.storage_path} to {new_path}")
            # files=glob(os.path.join(node_store.storage_path, '*'))
            shutil.copytree(node_store.storage_path, new_path)
            # for file in files:
            #     new_file = os.path.join(new_path, os.path.basename(file))
            #     shutil.copy(file, new_file)
            if remove_original:
                shutil.rmtree(node_store.storage_path)
            node_store.storage_path = new_path
        self.node_stores[node_store.node_type] = node_store
        
    def add_edge_store(self, edge_store: EdgeStore):
        logger.info(f"Adding edge store of type {edge_store.edge_type}")
        
        # Move edge store to the edges directory
        new_path = os.path.join(self.edges_path, edge_store.edge_type)
        if edge_store.storage_path != new_path:
            logger.debug(f"Moving edge store from {edge_store.storage_path} to {new_path}")
            os.makedirs(new_path, exist_ok=True)
            for file in glob(os.path.join(edge_store.storage_path, '*')):
                new_file = os.path.join(new_path, os.path.basename(file))
                os.rename(file, new_file)
            edge_store.storage_path = new_path
        self.edge_stores[edge_store.edge_type] = edge_store
        
    def node_list(self):
        return list(self.node_stores.keys())
    
    def node_exists(self, node_type: str):
        logger.debug(f"Node type: {node_type}")
        logger.debug(f"Node stores: {self.node_stores}")
        return node_type in self.node_stores
    
    def node_is_empty(self, node_type: str):
        store = self.add_node_type(node_type)
        return store.is_empty()
    
    def edge_list(self):
        return list(self.edge_stores.keys())
    
    def edge_exists(self, edge_type: str):
        return edge_type in self.edge_stores
    
    def get_node_store(self, node_type: str):
        # if node_type not in self.node_stores:
        node_store = self.node_stores.get(node_type, None)
        if node_store is None:
            raise ValueError(f"Node store of type {node_type} does not exist")
        return node_store
    
    def get_edge_store(self, edge_type: str):
        edge_store = self.edge_stores.get(edge_type, None)
        if edge_store is None:
            raise ValueError(f"Edge store of type {edge_type} does not exist")
        return edge_store
             
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
        store = self.get_node_store(node_type)
        store.update_nodes(data, **kwargs)

    def delete_nodes(self, node_type: str, ids: List[int] = None, columns: List[str] = None):
        store = self.get_node_store(node_type)
        store.delete_nodes(ids=ids, columns=columns)
        
    def remove_node_store(self, node_type: str):
        store = self.get_node_store(node_type)
        shutil.rmtree(store.storage_path)
        self.node_stores.pop(node_type)
        
    def remove_edge_store(self, edge_type:str):
        store = self.get_edge_store(edge_type)
        shutil.rmtree(store.storage_path)
        self.edge_stores.pop(edge_type)
        

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
        target_node_type: str) -> None:
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
        

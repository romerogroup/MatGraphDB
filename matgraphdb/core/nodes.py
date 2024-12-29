import os
import logging
from glob import glob
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parquetdb import ParquetDB

logger = logging.getLogger(__name__)

class NodeStore:
    """
    A wrapper around ParquetDB specifically for storing node features
    of a given node type.
    """

    def __init__(self, storage_path: str):
        """
        Parameters
        ----------
        storage_path : str
            The path where ParquetDB files for this node type are stored.
        """
        os.makedirs(storage_path, exist_ok=True)
        self.node_type = os.path.basename(storage_path)
        self.db = ParquetDB(db_path=storage_path)
        logger.debug(f"Initialized NodeStore at {storage_path}")

    def create_nodes(self, data: Union[List[dict], dict, pd.DataFrame],
                     schema: pa.Schema = None, metadata: Dict = None):
        """
        Creates new node records in the ParquetDB.
        The 'id' column is automatically assigned by ParquetDB.
        """
        num_records = len(data) if isinstance(data, (list, pd.DataFrame)) else 1
        logger.info(f"Creating {num_records} node records")
        try:
            self.db.create(data=data, schema=schema, metadata=metadata)
            logger.debug("Node creation successful")
        except Exception as e:
            logger.error(f"Failed to create nodes: {str(e)}")
            raise

    def read_nodes(self, ids: List[int] = None, columns: List[str] = None, **kwargs):
        """
        Reads node records (optionally filtered by IDs or columns).
        Accepts additional arguments (filters, load_format, etc.) 
        which are passed to ParquetDB.read.
        """
        id_msg = f"for IDs {ids[:5]}..." if ids else "for all nodes"
        col_msg = f" columns: {columns}" if columns else ""
        logger.debug(f"Reading nodes {id_msg}{col_msg}")
        try:
            result = self.db.read(ids=ids, columns=columns, **kwargs)
            logger.debug(f"Successfully read {len(result) if hasattr(result, '__len__') else 'unknown'} records")
            return result
        except Exception as e:
            logger.error(f"Failed to read nodes: {str(e)}")
            raise

    def update_nodes(self, data: Union[List[dict], dict, pd.DataFrame],
                     schema: pa.Schema = None, metadata: Dict = None):
        """
        Updates node records; each record must include 'id'.
        """
        num_records = len(data) if isinstance(data, (list, pd.DataFrame)) else 1
        logger.info(f"Updating {num_records} node records")
        try:
            self.db.update(data=data, schema=schema, metadata=metadata)
            logger.debug("Node update successful")
        except Exception as e:
            logger.error(f"Failed to update nodes: {str(e)}")
            raise

    def delete_nodes(self, ids: List[int] = None, columns: List[str] = None):
        """
        Deletes specific node records by IDs or entire columns.
        """
        if ids:
            logger.info(f"Deleting {len(ids)} nodes")
        if columns:
            logger.info(f"Deleting columns: {columns}")
        try:
            self.db.delete(ids=ids, columns=columns)
            logger.debug("Node deletion successful")
        except Exception as e:
            logger.error(f"Failed to delete nodes: {str(e)}")
            raise

    def normalize_nodes(self):
        """
        Triggers file restructuring and compaction to optimize node storage.
        """
        logger.info("Starting node store normalization")
        try:
            self.db.normalize()
            logger.debug("Node store normalization completed")
        except Exception as e:
            logger.error(f"Failed to normalize node store: {str(e)}")
            raise

import os
import logging
from typing import Union, List, Dict

import pandas as pd
import pyarrow as pa

from parquetdb import ParquetDB

logger = logging.getLogger(__name__)


class EdgeStore:
    """
    A wrapper around ParquetDB specifically for storing edge features
    of a given edge type.
    """
    required_fields = ['source_id', 'target_id', 'source_type', 'target_type']

    def __init__(self, storage_path: str):
        """
        Parameters
        ----------
        storage_path : str
            The path where ParquetDB files for this edge type are stored.
        """
        os.makedirs(storage_path, exist_ok=True)
        self.db_path = storage_path
        self.storage_path = storage_path
        self.edge_type = os.path.basename(storage_path)
        self.db = ParquetDB(db_path=storage_path)
        logger.info(f"Initialized EdgeStore at {storage_path}")

    def create_edges(self, data: Union[List[dict], dict, pd.DataFrame, pa.Table, pa.RecordBatch],
                     schema: pa.Schema = None, metadata: Dict = None):
        """
        Creates new edge records in the ParquetDB.
        """
        logger.debug(f"Creating edges with schema: {schema}")

        if not self.validate_edges(data):
            logger.error(
                "Edge data validation failed - missing required fields")
            raise ValueError(
                "Edge data is missing required fields. Must include: " + ", ".join(EdgeStore.required_fields))

        self.db.create(data=data, schema=schema, metadata=metadata)
        logger.info(f"Successfully created edges")

    def read_edges(self, ids: List[int] = None, columns: List[str] = None, **kwargs):
        """
        Reads edge records (optionally filtered by IDs or columns).
        """
        logger.debug(f"Reading edges with ids: {ids}, columns: {columns}")
        return self.db.read(ids=ids, columns=columns, **kwargs)

    def update_edges(self, data: Union[List[dict], dict, pd.DataFrame],
                     schema: pa.Schema = None, metadata: Dict = None):
        """
        Updates edge records; each record must include 'id'.
        """
        logger.debug(f"Updating edges with schema: {schema}")

        if not self.validate_edges(data):
            logger.error(
                "Edge data validation failed - missing required fields")
            raise ValueError(
                "Edge data is missing required fields. Must include: " + ", ".join(EdgeStore.required_fields))

        self.db.update(data=data, schema=schema, metadata=metadata)
        logger.info("Successfully updated edges")

    def delete_edges(self, ids: List[int] = None, columns: List[str] = None):
        """
        Deletes specific edge records by IDs or entire columns.
        """
        logger.debug(f"Deleting edges with ids: {ids}, columns: {columns}")
        self.db.delete(ids=ids, columns=columns)
        logger.info(f"Successfully deleted edges")

    def normalize_edges(self):
        """
        Triggers file restructuring and compaction to optimize edge storage.
        """
        logger.info("Starting edge store normalization")
        self.db.normalize()
        logger.info("Completed edge store normalization")

    def validate_edges(self, data: Union[List[dict], dict, pd.DataFrame, pa.Table, pa.RecordBatch]):
        """
        Validates the edges to ensure they contain the required fields.
        """
        logger.debug("Validating edge data")
        if isinstance(data, pd.DataFrame):
            fields = data.columns.tolist()
        elif isinstance(data, dict):
            fields = list(data.keys())
        elif isinstance(data, list):
            fields = list(data[0].keys())
        elif isinstance(data, pa.Table) or isinstance(data, pa.RecordBatch):
            fields = data.schema.names
        else:
            logger.error(
                f"Invalid data type for edge validation: {type(data)}")
            raise ValueError("Invalid data type for edge validation")

        is_valid = True
        missing_fields = []
        for required_field in EdgeStore.required_fields:
            if required_field not in fields:
                is_valid = False
                missing_fields.append(required_field)

        if not is_valid:
            logger.warning(
                f"Edge validation failed. Missing fields: {missing_fields}")
        else:
            logger.debug("Edge validation successful")

        return is_valid

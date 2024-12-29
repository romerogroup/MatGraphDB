import os
import logging
from typing import Union, List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from parquetdb import ParquetDB
from parquetdb.core.parquetdb import NormalizeConfig, LoadConfig

logger = logging.getLogger(__name__)


class EdgeStore(ParquetDB):
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
        self.storage_path = storage_path
        self.edge_type = os.path.basename(storage_path)
        logger.info(f"Initialized EdgeStore at {storage_path}")
        super().__init__(db_path=storage_path)

    def create_edges(self,
               data:Union[List[dict],dict,pd.DataFrame],
               schema:pa.Schema=None,
               metadata:dict=None,
               treat_fields_as_ragged:List[str]=None,
               convert_to_fixed_shape:bool=True,
               normalize_dataset:bool=False,
               normalize_config:dict=NormalizeConfig()
               ):
        """
        Adds new data to the database.

        Parameters
        ----------
        data : dict, list of dict, or pandas.DataFrame
            The data to be added to the database.
        schema : pyarrow.Schema, optional
            The schema for the incoming data.
        metadata : dict, optional
            Metadata to be attached to the table.
        normalize_dataset : bool, optional
            If True, the dataset will be normalized after the data is added (default is True).
        treat_fields_as_ragged : list of str, optional
            A list of fields to treat as ragged arrays.
        convert_to_fixed_shape : bool, optional
            If True, the ragged arrays will be converted to fixed shape arrays.
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.
        Example
        -------
        >>> db.create(data=my_data, schema=my_schema, metadata={'source': 'api'}, normalize_dataset=True)
        """
        logger.debug(f"Creating edges with schema: {schema}")

        if not self.validate_edges(data):
            logger.error(
                "Edge data validation failed - missing required fields")
            raise ValueError(
                "Edge data is missing required fields. Must include: " + ", ".join(EdgeStore.required_fields))
        create_kwargs = dict(data=data, schema=schema, metadata=metadata,
                             treat_fields_as_ragged=treat_fields_as_ragged,
                             convert_to_fixed_shape=convert_to_fixed_shape,
                             normalize_dataset=normalize_dataset,
                             normalize_config=normalize_config)
        self.create(**create_kwargs)
        logger.info(f"Successfully created edges")

    def read_edges(self, ids: List[int] = None,
        columns: List[str] = None,
        filters: List[pc.Expression] = None,
        load_format: str = 'table',
        batch_size:int=None,
        include_cols: bool = True,
        rebuild_nested_struct: bool = False,
        rebuild_nested_from_scratch: bool = False,
        load_config:LoadConfig=LoadConfig(),
        normalize_config:NormalizeConfig=NormalizeConfig()
        ):
        """
        Reads data from the database.

        Parameters
        ----------
        
        ids : list of int, optional
            A list of IDs to read. If None, all data is read (default is None).
        columns : list of str, optional
            The columns to include in the output. If None, all columns are included (default is None).
        filters : list of pyarrow.compute.Expression, optional
            Filters to apply to the data (default is None).
        load_format : str, optional
            The format of the returned data: 'table' or 'batches' (default is 'table').
        batch_size : int, optional
            The batch size to use for loading data in batches. If None, data is loaded as a whole (default is None).
        include_cols : bool, optional
            If True, includes only the specified columns. If False, excludes the specified columns (default is True).
        rebuild_nested_struct : bool, optional
            If True, rebuilds the nested structure (default is False).
        rebuild_nested_from_scratch : bool, optional
            If True, rebuilds the nested structure from scratch (default is False).
        load_config : LoadConfig, optional
            Configuration for loading data, optimizing performance by managing memory usage.
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.
        
        Returns
        -------
        pa.Table, generator, or dataset
            The data read from the database. The output can be in table format or as a batch generator.
        
        Example
        -------
        >>> data = db.read_edges(ids=[1, 2, 3], columns=['name', 'age'], filters=[pc.field('age') > 18])
        """
        logger.debug(f"Reading edges with ids: {ids}, columns: {columns}")
        
        read_kwargs = dict(ids=ids, columns=columns, filters=filters, load_format=load_format, 
                           batch_size=batch_size, include_cols=include_cols, 
                           rebuild_nested_struct=rebuild_nested_struct, 
                           rebuild_nested_from_scratch=rebuild_nested_from_scratch, 
                           load_config=load_config, normalize_config=normalize_config)
        return self.read(**read_kwargs)
    
    def update(self, 
            data: Union[List[dict], dict, pd.DataFrame], 
            schema:pa.Schema=None, 
            metadata:dict=None,
            update_key:str='id',
            treat_fields_as_ragged=None,
            convert_to_fixed_shape:bool=True,
            normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Updates existing records in the database.

        Parameters
        ----------
        data : dict, list of dicts, or pandas.DataFrame
            The data to be updated in the database. Each record must contain an 'id' key 
            corresponding to the record to be updated.
        schema : pyarrow.Schema, optional
            The schema for the data being added. If not provided, it will be inferred.
        metadata : dict, optional
            Additional metadata to store alongside the data.
        treat_fields_as_ragged : list of str, optional
            A list of fields to treat as ragged arrays.
        convert_to_fixed_shape : bool, optional
            If True, the ragged arrays will be converted to fixed shape arrays.
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.
        
        Example
        -------
        >>> db.update(data=[{'id': 1, 'name': 'John', 'age': 30}, {'id': 2, 'name': 'Jane', 'age': 25}])
        """
        
        if not self.validate_edges(data):
            logger.error( "Edge data validation failed - missing required fields")
            raise ValueError("Edge data is missing required fields. Must include: " + ", ".join(EdgeStore.required_fields))
            
        update_kwargs = dict(data=data, schema=schema, metadata=metadata,
                             update_key=update_key,
                             treat_fields_as_ragged=treat_fields_as_ragged,
                             convert_to_fixed_shape=convert_to_fixed_shape,
                             normalize_config=normalize_config)
        super().update(**update_kwargs) 

    def update_edges(self, 
            data: Union[List[dict], dict, pd.DataFrame], 
            schema:pa.Schema=None, 
            metadata:dict=None,
            update_key:str='id',
            treat_fields_as_ragged=None,
            convert_to_fixed_shape:bool=True,
            normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Updates existing records in the database.

        Parameters
        ----------
        data : dict, list of dicts, or pandas.DataFrame
            The data to be updated in the database. Each record must contain an 'id' key 
            corresponding to the record to be updated.
        schema : pyarrow.Schema, optional
            The schema for the data being added. If not provided, it will be inferred.
        metadata : dict, optional
            Additional metadata to store alongside the data.
        treat_fields_as_ragged : list of str, optional
            A list of fields to treat as ragged arrays.
        convert_to_fixed_shape : bool, optional
            If True, the ragged arrays will be converted to fixed shape arrays.
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.
        
        Example
        -------
        >>> db.update(data=[{'id': 1, 'name': 'John', 'age': 30}, {'id': 2, 'name': 'Jane', 'age': 25}])
        """
        logger.debug(f"Updating edges with schema: {schema}")

        if not self.validate_edges(data):
            logger.error(
                "Edge data validation failed - missing required fields")
            raise ValueError(
                "Edge data is missing required fields. Must include: " + ", ".join(EdgeStore.required_fields))
        update_kwargs = dict(data=data, schema=schema, metadata=metadata,
                             update_key=update_key,
                             treat_fields_as_ragged=treat_fields_as_ragged,
                             convert_to_fixed_shape=convert_to_fixed_shape,
                             normalize_config=normalize_config)
        self.update(**update_kwargs)
        logger.info("Successfully updated edges")

    def delete_edges(self,  
            ids:List[int]=None, 
            columns:List[str]=None, 
            normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Deletes records from the database.

        Parameters
        ----------
        ids : list of int
            A list of record IDs to delete from the database.
        columns : list of str, optional
            A list of column names to delete from the dataset. If not provided, it will be inferred from the existing data (default: None).
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.
            
        Returns
        -------
        None

        Example
        -------
        >>> db.delete(ids=[1, 2, 3])
        """
        logger.debug(f"Deleting edges with ids: {ids}, columns: {columns}")
        self.delete(ids=ids, columns=columns, normalize_config=normalize_config)
        logger.info(f"Successfully deleted edges")

    def normalize_edges(self, normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Triggers file restructuring and compaction to optimize edge storage.
        """
        logger.info("Starting edge store normalization")
        self.normalize(normalize_config=normalize_config)
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

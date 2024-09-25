
import os
import json
import logging
from glob import glob
import logging
from multiprocessing import Pool
from functools import partial
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


logger = logging.getLogger(__name__)


# https://arrow.apache.org/docs/python/api/datatypes.html
t_string=pa.string()
t_int32=pa.int32()
t_int64=pa.int64()
t_float32=pa.float32()
t_float64=pa.float64()
t_bool=pa.bool_()

# Create variable-length or fixed size binary type.
t_binary = pa.binary()

#one of ‘s’ [second], ‘ms’ [millisecond], ‘us’ [microsecond], or ‘ns’ [nanosecond]
t_timestamp=pa.timestamp('ms')


def append_to_parquet_table(dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer

class ParquetDatabase:
    def __init__(self, db_path='ParquetDatabase', schema=None, n_cores=4):
        """
        Initializes the ParquetDatabase object.

        Args:
            db_path (str): The path to the root directory of the database.
            n_cores (int): The number of CPU cores to be used for parallel processing.
        """

        self.db_path = db_path
        self.db_file = os.path.join(self.db_path, 'db.parquet')
        self.parquet_schema_file = os.path.join(self.db_path, 'schema.parquet')

        os.makedirs(self.db_path, exist_ok=True)

        if schema:
            self.set_schema(schema)

        self.n_cores = n_cores
        self.metadata = {}
        # self._load_state()

        logger.info(f"db_file: {self.db_file }")
        logger.info(f"n_cores: {self.n_cores}")
        for x in self.metadata.items():
            logger.info(f"{x[0]}: {x[1]}")

    def _load_parquet_schema(self):
        """
        Loads the schema from the Parquet file if it exists. If the file does not exist, 
        an empty schema is returned.

        Returns:
            pyarrow.Schema or list: The schema of the Parquet file, or an empty list if the file doesn't exist.
        """
        if os.path.exists(self.parquet_schema_file):
            logger.debug(f"Loading Parquet schema from {self.parquet_schema_file}")
            table = pq.read_table(self.parquet_schema_file)
            return table.schema
        else:
            logger.warning(f"Parquet schema file {self.parquet_schema_file} does not exist.")
            return []
        
    @property
    def schema(self):
        """
        Returns the schema of the Parquet file. If the schema file does not exist, 
        it will attempt to create one.

        Returns:
            pyarrow.Schema: The current schema stored in the Parquet file.
        """
        parquet_schema = self._load_parquet_schema()
        logger.debug("Retrieved Parquet schema.")
        return parquet_schema
    
    def set_schema(self, schema):
        """
        Sets and saves a new schema for the Parquet file. This overwrites any existing schema.

        Args:
            schema (pyarrow.Schema): The new schema to be saved in the Parquet file.

        Returns:
            pyarrow.Schema: The schema that was saved.
        """
        logger.info("Setting new Parquet schema.")
        empty_table = pa.Table.from_pandas(
                                pd.DataFrame(
                                    columns=[field.name for field in schema]), 
                                    schema=schema
                                    )
        pq.write_table(empty_table, self.parquet_schema_file )

        logger.info(f"Schema updated and saved with fields: {[field.name for field in schema]}")
        return schema
        
    def add_field_to_schema(self, new_fields:List[pa.field]):
        """
        Adds new fields to the existing Parquet schema and saves the updated schema.

        Args:
            new_fields (list of pyarrow.field): A list of new fields to be added to the existing schema.

        Returns:
            pyarrow.Schema: The updated schema with the new fields added.
        """
        logger.info(f"Adding new fields to Parquet schema: {[field.name for field in new_fields]}")
        parquet_schema = self._load_parquet_schema()

        # Create a dictionary of current fields to easily replace or add new fields
        schema_fields_dict = {field.name: field for field in parquet_schema}

        # Update or add new fields
        for new_field in new_fields:
            schema_fields_dict[new_field.name] = new_field
        
        # Create the updated schema from the updated field dictionary
        updated_schema = pa.schema(list(schema_fields_dict.values()))

        empty_table = pa.Table.from_pandas(
                                pd.DataFrame(
                                    columns=[field.name for field in updated_schema]), 
                                    schema=updated_schema
                                    )
        pq.write_table(empty_table, self.parquet_schema_file )

        logger.info(f"Schema updated and saved with fields: {[field.name for field in updated_schema]}")
        return updated_schema

if __name__ == '__main__':

    data_dir='data/raw/ParquetDB'
    # table1 = pd.DataFrame({'one': [-1, np.nan, 2.5], 'two': ['foo', 'bar', 'baz'], 'three': [True, False, True]})
    # table2 = pd.DataFrame({'one': [-1, np.nan, 2.5], 'two': ['foo', 'bar', 'baz'], 'three': [True, False, True]})
    # table3 = pd.DataFrame({'one': [-1, np.nan, 2.5], 'two': ['foo', 'bar', 'baz'], 'three': [True, False, True]})
    # writer = None
    filepath = os.path.join(data_dir, 'test.parquet')
    # os.makedirs(data_dir, exist_ok=True)
    # table_list = [table1, table2, table3]

    # for table in table_list:
    #     writer = append_to_parquet_table(table, filepath, writer)

    # if writer:
    #     writer.close()

    # table = pd.read_parquet(filepath)
    # print(table)

    # writer = append_to_parquet_table(table, filepath, writer)

    # table3 = pd.DataFrame({'one': [-1, np.nan, 2.5], 'two': ['foo', 'bar', 'baz'], 'three': [True, False, True]})
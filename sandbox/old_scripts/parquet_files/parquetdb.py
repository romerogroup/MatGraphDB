from collections import defaultdict
import os
import json
from typing import Callable, List, Union
import uuid
import logging
from xml.etree.ElementInclude import include
from attr import field
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool
from functools import partial

from tomlkit import table

from matgraphdb.utils import timeit


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


# Logger setup
logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Serialization functions
def serialize(data):
    """
    Serializes the given data using the json.dumps() function.
    
    Args:
        data (dict or list of dicts): The data to be serialized.
    
    Returns:
        str: The serialized data.
    """
    return json.dumps(data)

def deserialize(data, filter_func=None):
    """
    Deserializes the given data using the json.loads() function.
    
    Args:
        data (str): The data to be deserialized.
        filter_func (function): A function to filter the data. This function should take a dictionary as input and return a dictionary.

    
    Returns:
        dict or list of dicts: The deserialized data.
    """
    if filter_func is None:
        return json.loads(data)
    else:
        deserialized_data = json.loads(data)
        filtered_results = filter_func(deserialized_data)
        return deserialized_data, filtered_results
    
def get_field_names(filepath, columns=None, include_cols=True):
    if not include_cols:
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)

        columns = [col for col in all_columns if col not in columns]
    return columns

def is_contained(list1, list2):
    return set(list1).issubset(set(list2))

class ParquetDatabase:
    def __init__(self, db_path='Database', n_cores=8):
        """
        Initializes the ParquetDatabase object.

        Args:
            db_path (str): The path to the root directory of the database.
            n_cores (int): The number of CPU cores to be used for parallel processing.
        """
        self.db_path = db_path
        self.table_names=['main']
        self.main_table_file=os.path.join(self.db_path, 'main.parquet')

        os.makedirs(self.db_path, exist_ok=True)

        self.n_cores = n_cores
        self.metadata = {}
        logger.info(f"db_path: {self.db_path}")
        logger.info(f"table_names: {self.table_names}")
        logger.info(f"main_table_file: {self.main_table_file}")
        logger.info(f"n_cores: {self.n_cores}")

    def get_schema(self, table_name:str ='main'):
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')
        schema = pq.read_schema(table_path)
        return schema
        
    @timeit
    def get_batch_generator(self, table_name:str='main', 
                            columns:List[str]=None, 
                            include_cols:bool=True, 
                            deserialize_data:bool=False, 
                            filter_data_func:Callable=None,
                            unpack_filter_results:bool=False,
                            batch_size=10000):
        """
        Reads data from the database in batches.

        Args:
            table_name (str): The name of the table to read data from.
            columns (list): A list of columns to include in the returned data. By default, all columns are included.
            include_cols (bool): If True, includes the only the fields listed in columns
                If False, includes all fields except the ones listed in columns.
            deserialize_data (bool): If True, deserializes the data in a multiprocessing fashion.
            filter_data_func (function): A function to filter the data column while deserialing. 
                This function expects the a deserialized data dictionary as input 
                and returns the output of the filter function as another column. 
                By defualt the results are packed into tuple and stored in a single column. By setting this to True, 
                the results are unpacked and stored in separate columns.
            unpack_filter_results (bool): If True, unpacks the filter results.
            batch_size (int): The number of rows to read in each batch.

        Returns:
            pandas.DataFrame: A DataFrame containing the requested data.
        """
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        parquet_file = pq.ParquetFile(table_path)

        columns_to_read=columns
        if columns_to_read:
            columns_to_read = get_field_names(table_path, columns=columns, include_cols=include_cols)
                
        # Iterate through batches
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read):
            df_batch = record_batch.to_pandas()

            if 'data' in df_batch.columns and deserialize_data:
                df_batch=self._deserialize_data(df_batch, filter_data_func=filter_data_func, unpack_filter_results=unpack_filter_results)

            yield df_batch

    @timeit
    def create(self, data:Union[List[dict],dict,pd.DataFrame], 
               field_data:Union[List[dict],dict]=None, 
               table_name:str='main', 
               serialize_data:bool=True,
               serialize_output:bool=True,
               batch_size:int=None):
        """
        Adds new data to the database.

        Args:
            data (dict or list of dicts): The data to be added to the database. 
                This must contain
        """

        # Add the table name to the list of table names
        self.table_names.append(table_name)
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        # Prepare the data and field data
        data_list=self._validate_data(data)
        field_data_dict=self._validate_field_data(field_data)
        
        # Get new ids
        new_ids = self._get_new_ids(table_path, data_list)
        
        # Serialize the data in a multiprocessing fashion
        # if serialize_data:
        #     data_list=self._serialize_data(data_list)
        # else:
        #     logger.info("Data is assumed to be serialized already. Data is not being serialized.")

        # Create a DataFrame from the serialized data
        column_data={'id': new_ids, 'data': data_list}

        # Handle field data if any
        if field_data_dict:
            logger.warning("When adding field data, it will input None for prexisiting records")
            column_data.update(field_data_dict)

        new_df = pd.DataFrame(column_data)

        # Create new table with or without batches
        if batch_size:
            self._create_in_batches(df=new_df, table_name=table_name, batch_size=batch_size, serialize_output=serialize_output)
        else:
            self._create_without_batches(df=new_df, table_name=table_name, serialize_output=serialize_output)

        logger.info("Data added successfully.")

    @timeit
    def read(self, ids=None, table_name:str='main', 
                columns:List[str]=None, 
                include_cols:bool=True, 
                deserialize_data:bool=False, 
                filter_data_func:Callable=None,
                unpack_filter_results:bool=False):
        """
        Reads data from the database.

        Args:
            ids (list): A list of IDs to read. If None, reads all data.
            table_name (str): The name of the table to read data from.
            columns (list): A list of columns to include in the returned data. By default, all columns are included.
            include_cols (bool): If True, includes the only the fields listed in columns
                If False, includes all fields except the ones listed in columns.
            deserialize_data (bool): If True, deserializes the data in a multiprocessing fashion.
            filter_data_func (function): A function to filter the data column while deserialing. 
                This function expects the a deserialized data dictionary as input 
                and returns the output of the filter function as another column. 
                By defualt the results are packed into tuple and stored in a single column. By setting this to True, 
                the results are unpacked and stored in separate columns.
            unpack_filter_results (bool): If True, unpacks the filter results.

        Returns:
            pandas.DataFrame or list: The data read from the database. If deserialize_data is True,
            returns a list of dictionaries with their 'id's. Otherwise, returns the DataFrame with serialized data.
        """
        if filter_data_func and not deserialize_data:
            logger.warning("Deserialize_data must be True if filter_data_func is provided. Otherwise, it does nothing")

        # Check if the table name is in the list of table names
        # self._check_table_name(table_name)
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        columns_to_read=columns
        if columns:
            columns_to_read = get_field_names(table_path, columns=columns, include_cols=include_cols)

        df = self._load_data(table_path, columns=columns_to_read)

        if df.empty:
            logger.info("No data to read.")
            return [] if deserialize_data else df

        if ids is not None and 'id' in df.columns:
            df = df[df['id'].isin(ids)]

        if 'data' in df.columns and deserialize_data:
           df=self._deserialize_data(df, filter_data_func=filter_data_func, unpack_filter_results=unpack_filter_results)
           
        return df
    
    @timeit
    def update(self, ids:List[int],
               data: Union[List[dict], dict, pd.DataFrame]=None,
               field_data: Union[List[dict], dict] = None,
               table_name: str = 'main',
               serialize_data: bool = False,
               serialize_output: bool = False,
               batch_size: int = None):
        """
        Updates existing data in the database.

        Args:
            data (dict or list of dicts or DataFrame): The data to be updated. Must include 'id' field.
            field_data (dict or list of dicts): Additional field data to be updated.
            table_name (str): The name of the table to update data in.
            serialize_data (bool): If True, serializes the 'data' field in a multiprocessing fashion.
            batch_size (int): The batch size for processing. If None, processes all data at once.
        """

        if data is None and field_data is None:
            raise ValueError("Either data or field_data must be provided.")
        # self.table_names.append('main_new')
        # Check if the table name is in the list of table names
        self._check_table_name(table_name)
        table_path = os.path.join(self.db_path, f'{table_name}.parquet')

        # Validate data
        data_list = self._validate_data(data)
        field_data_dict = self._validate_field_data(field_data)

        # # Create DataFrame from data_list
        column_data={'id': ids}

        # Handle data if any
        if data is not None:
            logger.info('Found data to update')

            if serialize_data:
                data_list=self._serialize_data(data_list)

            column_data.update({'data':data_list})

        # Handle field data if any
        if field_data_dict:
            logger.info('Found field data to update')
            logger.warning("When adding field data, it will input None for prexisiting records")
            column_data.update(field_data_dict)

        update_df = pd.DataFrame(column_data)

        # if batch_size:
        #     self._update_in_batches(update_df, table_name=table_name, batch_size=batch_size)
        # else:
        self._update_without_batches(update_df, table_name=table_name)

        logger.info("Data updated successfully.")

    def _update_in_batches(self, update_df, table_name='main', batch_size=1000, schema=None):
        logger.info("Updating data in batches.")

        table_path = os.path.join(self.db_path, f'{table_name}.parquet')
        new_table_path = os.path.join(self.db_path, f'{table_name}_new.parquet')

        # If schema is None, get the existing schema
        if schema is None:
            schema = self.get_schema(table_name)

        # Create ParquetWriter with new schema
        writer = pq.ParquetWriter(new_table_path, schema)

        # Get a set of IDs to update
        update_ids = set(update_df['id'].tolist())

        # Create a mapping from id to updated data
        update_records = update_df.set_index('id')

        batch_generator = self.get_batch_generator(table_name=table_name,
                                                   batch_size=batch_size,
                                                   deserialize_data=False)

        # Keep track of IDs that have been updated
        updated_ids = set()

        for batch_df in batch_generator:
            # Ensure batch_df has all columns in schema
            missing_columns = set(schema.names) - set(batch_df.columns)
            for col in missing_columns:
                batch_df[col] = None

            # For each batch, update records that have matching IDs
            batch_ids = set(batch_df['id'].tolist())
            ids_to_update = batch_ids.intersection(update_ids)

            if ids_to_update:
                # Update records in batch_df
                for id_to_update in ids_to_update:
                    # Get the updated record
                    updated_record = update_records.loc[id_to_update]
                    # Update batch_df
                    idx = batch_df['id'] == id_to_update
                    # Update existing columns
                    common_cols = batch_df.columns.intersection(updated_record.index)
                    batch_df.loc[idx, common_cols] = updated_record[common_cols].values
                    # Add new columns if any
                    new_cols = updated_record.index.difference(batch_df.columns)
                    for col in new_cols:
                        batch_df[col] = None
                        batch_df.loc[idx, col] = updated_record[col]
                    updated_ids.add(id_to_update)
            else:
                # Ensure batch_df has all columns in schema
                missing_columns = set(schema.names) - set(batch_df.columns)
                for col in missing_columns:
                    batch_df[col] = None

            # Write the batch to the new Parquet file
            writer.write_table(pa.Table.from_pandas(batch_df, schema=schema))

        # Check for IDs that were not found in existing data
        not_found_ids = update_ids - updated_ids
        if not_found_ids:
            logger.warning(f"The following IDs were not found in the existing data and were not updated: {not_found_ids}")

        # Close the writer
        writer.close()

        # Replace the old Parquet file with the new one
        os.remove(table_path)
        os.rename(new_table_path, table_path)

        logger.info(f"Parquet file {table_path} updated successfully in batches.")

    def _update_without_batches(self, update_df, table_name='main'):
        logger.info("Updating data without batches.")

        # Load existing data
        table_path = os.path.join(self.db_path, f'{table_name}.parquet')
        existing_df = self._load_data(table_path)

        if existing_df.empty:
            logger.warning("No existing data found. Update operation cannot proceed.")
            return

        incoming_columns = set(update_df.columns)
        original_columns = set(existing_df.columns)
        # Ensure existing_df has all columns in update_df
        new_fields = set(incoming_columns) - set(original_columns)

        field_names= set(incoming_columns) - set(['id','data'])


        for col in new_fields:
            existing_df[col] = None

        if 'data' in incoming_columns:
            existing_df=self._deserialize_data(existing_df)
        
        for irow, row in update_df.iterrows():
            id=row['id']
            new_data=row['data']
            
            match_idx = existing_df[existing_df['id'] == id].index

            if not match_idx.empty:
                
                # Handle field data
                for field_name in field_names:
                    field_data=row[field_name]
                    existing_df.at[match_idx[0], field_name] = field_data
                    
                # If id exists in existing_df, update the dictionary in the 'data' column
                existing_data = existing_df.at[match_idx[0], 'data']

                if isinstance(existing_data, dict):
                    # Update the existing dictionary with new_data
                    existing_data.update(new_data)
                else:
                    # If 'data' is not a dictionary, replace it with new_data
                    existing_df.at[match_idx[0], 'data'] = new_data



        # Save data
        new_table_path = os.path.join(self.db_path, f'{table_name}_new.parquet')
        self._save_data(existing_df, table_path=new_table_path)

        logger.info(f"Parquet file {table_path} updated successfully without batches.")

    def create_table(self, func: Callable, table_name='main2', use_batches=False, batch_size=10000):
        """
        Creates a new Parquet table by applying a function to the data.

        Args:
            func (Callable): A function that takes a DataFrame and returns a DataFrame.
            table_name (str): The name of the new table to be created.
            use_batches (bool): Whether to read and process the data in batches.
            batch_size (int): The number of rows to read in each batch if using batches.
        """
        new_table_path = os.path.join(self.db_path, f'{table_name}.parquet')

        if use_batches:
            batch_generator = self.get_batch_generator(table_name='main', 
                                                        batch_size=batch_size, 
                                                        deserialize_data=True)
            self._write_parquet_in_batches(batch_generator, func, file_path=new_table_path)
        else:
            # Load entire dataset if not using batches
            df = self.read(table_name='main', deserialize_data=True)
            if df.empty:
                logger.info("No data to create table from.")
                return
            # Apply the function
            final_df = func(df)
            # Convert the DataFrame to a PyArrow Table
            table = pa.Table.from_pandas(final_df)
            # Write the table to the Parquet file
            pq.write_table(table, new_table_path)
            logger.info(f"Parquet file {new_table_path} created successfully.")

    def _load_data(self, table_path, columns=None):
        """
        Loads the data from the Parquet file, if it exists.
        """
        if os.path.exists(table_path):
            table = pq.read_table(table_path, columns=columns)

            df = table.to_pandas()
            logger.info(f"Loaded data from {table_path}")
            return df
        else:
            logger.info(f"No data found at {table_path}, returning an empty DataFrame.")
            return pd.DataFrame(columns=['id','data'])

    def _save_data(self, df, table_path):
        """
        Saves the provided DataFrame to a Parquet file.
        """
        table = pa.Table.from_pandas(df)

        pq.write_table(table, table_path)
        logger.info(f"Data saved to {table_path}")

    def _process_task(self, func, items, **kwargs):
        logger.info(f"Processing tasks using {self.n_cores} cores")
        with Pool(self.n_cores) as p:
            if isinstance(items[0], tuple):
                logger.info("Using starmap")
                results = p.starmap(partial(func, **kwargs), items)
            else:
                logger.info("Using map")
                results = p.map(partial(func, **kwargs), items)
        return results
    
    def _get_new_ids(self, table_path, data_list):
        df = self._load_data(table_path, columns=['id'])

        # Find the starting ID for the new data
        if df.empty:
            start_id = 1
        elif 'id' in df.columns:
            start_id = df['id'].max() + 1  # Start from the next available ID
    
        # Create a list of new IDs
        new_ids = list(range(start_id, start_id + len(data_list)))
        return new_ids
    
    def _create_in_batches(self, df, table_name='main', batch_size=1000):
        logger.info("Creating new data in batches.")

        table_path=os.path.join(self.db_path, f'{table_name}.parquet')
        new_table_path = os.path.join(self.db_path, f'{table_name}_new.parquet')

        schema = self.get_schema(table_name)

        geneator = self.get_batch_generator(table_name=table_name, 
                                            batch_size=batch_size, 
                                            deserialize_data=False)
        
        writer = pq.ParquetWriter(new_table_path, schema)
        for batch_df in geneator:
            writer.write_table(pa.Table.from_pandas(batch_df))
        writer.write_table(pa.Table.from_pandas(df))
        writer.close()

        os.remove(table_path)
        os.rename(new_table_path, table_path)

        logger.info(f"Parquet file {table_path} created successfully in batches.")

    def _create_without_batches(self, df, table_name='main', serialize_output=False):
        logger.info("Creating new data without batches.")

        table_path=os.path.join(self.db_path, f'{table_name}.parquet')
        original_df = self._load_data(table_path)

        new_field_names=set(df.columns)
        orginal_field_names=set(original_df.columns)
        new_field_names=list(new_field_names - orginal_field_names)

        for field_name in new_field_names:
            original_df[field_name]=None
        
        # Append new data to the existing data
        df = pd.concat([original_df, df], ignore_index=True)
        # Save the data
        self._save_data(df,table_path=table_path)

    def _serialize_data(self, data_list):
        logger.info("Serializing data.")
        mp_results = self._process_task(serialize, data_list)
        return mp_results

    def _unpack_filter_results(self, df, filter_results):
        try:
            filter_results = zip(*filter_results)
        except:
            logger.error("Filter results must return tuple of same length.")
            
        for i, filter_result in enumerate(filter_results):
            df[f'filter_result-{i}'] = filter_result
        return filter_results
    
    def _deserialize_data(self, df, filter_data_func=None, unpack_filter_results=False):
        """
        Deserializes data in a multiprocessing fashion.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to be deserialized.
            filter_func (function): A function to filter the data. This function should take a dictionary as input and return a dictionary.

        Returns:
            pandas.DataFrame: The DataFrame with the deserialized data.
        """
        logger.info("Deserializing data.")

        serialized_data_list = df['data'].tolist()
        mp_results = self._process_task(deserialize, serialized_data_list, filter_func=filter_data_func)

        if filter_data_func:
            deserialized_data, filter_results = zip(*mp_results)
        else:
            deserialized_data = mp_results
            filter_results = None

        if unpack_filter_results:
            filter_results = self._unpack_filter_results(df, filter_results)
        elif filter_data_func and not unpack_filter_results:
            df['filter_results'] = filter_results
            

        df['data'] = deserialized_data
        return df

    def _check_table_name(self, table_name):
        if table_name not in self.table_names:
            raise ValueError(f"Table name {table_name} not found in the database.")
    
    def _validate_data(self, data):
        if isinstance(data, dict):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        elif isinstance(data, pd.DataFrame):
            data_list = data.to_dict(orient='records')
        elif data is None:
            data_list = None
        else:
            raise TypeError("Data must be a dictionary or a list of dictionaries.")
        return data_list
    
    def _validate_field_data(self, field_data):
        if isinstance(field_data, dict):
            field_data_list = field_data
        elif isinstance(field_data, list):
            feild_data_dict = defaultdict(list)
            for fields_dict in field_data:
                for key, value in fields_dict.items():
                    feild_data_dict[key].append(value)
        elif field_data is None:
            field_data_list = None
        else:
            raise TypeError("Field data must be a dictionary or a list of dictionaries.")
        return field_data_list

    # def create_parquet_from_data(self, func:Callable, schema, table_name: str =  'main' ):
    #     """
    #     Creates a Parquet file from data in the material database, transforming each row using the provided function.

    #     Args:
    #         func (Callable): A function that processes each row of data from the database and returns a tuple of 
    #                         column names and corresponding values.
    #         schema (pyarrow.Schema): The schema for the output Parquet file.
    #         output_file (str): Path to the output Parquet file. Defaults to 'materials_database.parquet'.

    #     Raises:
    #         Exception: If there is an issue processing rows.

    #     Returns:
    #         None
    #     """
    #     table_path=os.path.join(self.db_path, table_name)

    #     logger.info(f"Creating Parquet file from data: {table_path}")
    #     error_message = "\n".join([
    #             "Make the function return a tuple with column_names and values.",
    #             "Also make sure the order of the column_names and values correspond with each other.",
    #             "I would also recommend using .get({column_name}, None) to get the value of a column if it exists.",
    #             "Indexing would maybe cause errors if the property does not exist for a material."
    #         ])
    #     df=self.read(table_name='main', deserialize_data=True)

    #     processed_data = []
    #     column_names=None
    #     for row in rows:
    #         try:
    #             column_names, row_data=func(row.data)
    #             logger.debug(f"Processed row ID {row.id}: {row_data}")
    #         except Exception as e:
    #             logger.error(f"Error processing row ID {row.id}: {e}")
    #             logger.debug(f"Row data: {row.data}")
    #             logger.debug(error_message)

    #     #     processed_data.append(row_data)

    #     # Convert the processed data into a DataFrame for Parquet export
    #     df = pd.DataFrame(processed_data, columns=column_names)

    #     # Write the DataFrame to a Parquet file
    #     df.to_parquet(table_path, engine='pyarrow', schema=schema, index=False)
    #     logger.info(f"Data exported to {table_path}")

    #     # self.set_schema(schema)
    #     logger.info(f"Schema file updated and saved to {self.table_path}")


# def test_filter_func(data):
#     if data['nsites']==44:
#         return data['nsites'], data['material_id']
#     else:
#         return None, None
    

    
if __name__ == '__main__':
    db = ParquetDatabase(db_path='data/raw/ParquetDB')

    ex_json_file_1='data/production/materials_project/json_database/mp-27352.json'
    ex_json_file_2='data/production/materials_project/json_database/mp-1000.json'

    # files=glob('data/production/materials_project/json_database/*.json')


    with open(ex_json_file_1, 'r') as f:
        data_1 = json.load(f)

    with open(ex_json_file_2, 'r') as f:
        data_2 = json.load(f)


    # print(data_1)
    # print('-'*100)
    # print(data_2)

    # json_string_1 = json.dumps(data_1)
    # json_string_2 = json.dumps(data_2)

    data_list=[]
    field_data={'field1':[]}

    json_string_1 = json.dumps(data_1)
    json_string_2 = json.dumps(data_2)
    # for i in range(1000):
    #     if i%2==0:
    #         data_list.append(json_string_1)
    #     else:
    #         data_list.append(json_string_2)

    #     field_data['field1'].append(i)
    # db.create(data=data_list, field_data={'field1':1}, serialize_data=False)
    # db.create(data=data_list, field_data={'field1':1}, batch_size=1000, serialize_data=False)

    
    for i in range(1000):
        if i%2==0:
            data_list.append(data_1)
        else:
            data_list.append(data_2)
        field_data['field1'].append(i)

    db.create(data=data_list, serialize_data=False)

    #####################################################################################
    # # Testing ids
    # df=db.read(ids=[1,2], table_name='main', deserialize_data=False)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # Testing deserialization
    df=db.read(table_name='main', deserialize_data=False)

    data=df.iloc[1]['data']
    lattice=data['structure']['lattice']['matrix']
    for i in range(3):
        print(lattice[i])
    print(type(lattice))
    print(lattice.shape)
    print(data)
    print(df.head())
    print(df.tail())
    print(df.shape)

    # # Testing unpacking filter function
    # df=db.read(table_name='main', deserialize_data=True, filter_func=test_filter_func)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # Testing unpacking filter results
    # df=db.read(table_name='main', deserialize_data=True, filter_func=test_filter_func, unpack_filter_results=True)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # Testing unpacking filter results
    # df=db.read(table_name='main', columns=['data'], deserialize_data=True, filter_data_func=test_filter_func, unpack_filter_results=True)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # Testing batch generator
    # batch_df=next(db.get_batch_generator(columns=['data'], 
    #                                     deserialize_data=True, 
    #                                     filter_data_func=test_filter_func,
    #                                     unpack_filter_results=True,
    #                                     batch_size=100))
    # print(batch_df.head())
    # print(batch_df.tail())
    # print(batch_df.shape)



    # # Testing update
    # update_data=[{'nsites':100, 'nelements':200}, {'nsites':100, 'nelements':200}]
    # field_data={'field1':[1,2]}

    # Testing updating only data
    # db.update(ids=[1,2], data=update_data, serialize_data=False)

    # # Testing updating only field data
    # db.update(ids=[1,2], field_data=field_data)

    # # Testing updating data and field data
    # db.update(ids=[1,2], data=update_data, field_data=field_data)



    # df=db.read(table_name='main_new', deserialize_data=False)
    # data=df.iloc[0]['data']
    # print(type(data))
    # print(type(data['structure']['lattice']['matrix']))

    # print()

    # for key, value in data.items():
    #     print(key, value)

    # print(data)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)










    # print(df.iloc[9995:10005])
    # print(df.iloc[19995:20005])

    # db._create_in_batches(data=data_list, table_name='main', serialize_data=False)



    # db.create_fields(['field1'],[t_int64],[1])

    # schema=db.get_schema('main')
    # print(schema)
    # print(schema.names)

    # print(dir(schema))
    # print(type(schema))





    # print(df.tail())
    # print(df.shape)

    # batch_df=db.read_batches(deserialize_data=True)
    # print(batch_df.head())

    # data_dict={'data':{'nsites':100}, ''}
    # batch_df=next(db.get_batch_generator(columns=['data'], deserialize_data=True))

    # # print(batch_df.head())

    # nsites=batch_df.iloc[0]['data']['nsites']
    # print(nsites)


    # for batch_df in db.get_batch_generator(deserialize_data=True):
    #     pass
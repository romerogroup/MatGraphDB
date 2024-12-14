from collections import defaultdict
import copy

from msilib import schema
import os
import json
import time
from typing import Callable, List, Tuple, Union
import uuid
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
from multiprocessing import Pool
from functools import partial

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

def find_difference_between_pyarrow_schemas(schema1, schema2):
    """
    Finds the difference between two PyArrow schemas.
    """
    # Create a set of field names from the first schema
    field_names1 = set(schema1)
    # Create a set of field names from the second schema
    field_names2 = set(schema2)
    # Find the difference between the two sets
    difference = field_names1.difference(field_names2)
    return difference

class ParquetDatabase:
    def __init__(self, db_path='Database', n_cores=8):
        """
        Initializes the ParquetDatabase object.

        Args:
            db_path (str): The path to the root directory of the database.
            n_cores (int): The number of CPU cores to be used for parallel processing.
        """
        self.db_path = db_path
        self.datasets_dir=os.path.join(self.db_path,'datasets')
        self.table_names=['main']
        self.main_table_file=os.path.join(self.db_path, 'main.parquet')

        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

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
                            filter_func:Callable=None,
                            filter_args:Tuple=None,
                            batch_size=1000,
                            output_format='pandas'):
        """
        Reads data from the database in batches.

        Args:
            table_name (str): The name of the table to read data from.
            columns (list): A list of columns to include in the returned data. By default, all columns are included.
            include_cols (bool): If True, includes the only the fields listed in columns
                If False, includes all fields except the ones listed in columns.
            filter_func (function): A function to filter the data column while deserialing. 
                This function expects the a deserialized data dictionary as input 
                and returns the output of the filter function as another column. 
            batch_size (int): The number of rows to read in each batch.

        Returns:
            pandas.DataFrame: A DataFrame containing the requested data.
        """
        self._validate_output_format(output_format)

        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        parquet_file = pq.ParquetFile(table_path)

        columns_to_read=columns
        if columns_to_read:
            columns_to_read = get_field_names(table_path, columns=columns, include_cols=include_cols)
        
        generator=parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read)
        # Iterate through batches
        for record_batch in generator:
            if output_format=='pandas':
                df_batch = record_batch.to_pandas()
                if filter_func:
                    df_batch=filter_func(df_batch)

                if filter_args:
                    df_batch=self._filter_data(df_batch, filter_args)

                yield df_batch
            elif output_format=='pyarrow':
                yield record_batch
            else:
                raise ValueError("output_format must be either 'pandas' or 'pyarrow'")

    @timeit
    def create(self, data:Union[List[dict],dict,pd.DataFrame], 
               table_name:str='main', 
               batch_size:int=None):
        """
        Adds new data to the database.

        Args:
            data (dict or list of dicts): The data to be added to the database. 
                This must contain
        """

        # Add the table name to the list of table names
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        # Prepare the data and field data
        data_list=self._validate_data(data)
        
        # Get new ids
        new_ids = self._get_new_ids(table_path, data_list)
        
        new_df = pd.DataFrame(data_list)

        new_df['id'] = new_ids

        # # Create new table with or without batches
        if batch_size and os.path.exists(table_path):
            self._create_in_batches(df=new_df, table_name=table_name, batch_size=batch_size)
        else:
            self._create_without_batches(df=new_df, table_name=table_name)

        logger.info("Data added successfully.")

    @timeit
    def read(self, ids=None, table_name:str='main', 
                columns:List[str]=None, 
                include_cols:bool=True, 
                filter_func:Callable=None,
                filter_args:Tuple=None,
                output_format='pandas'):
        """
        Reads data from the database.

        Args:
            ids (list): A list of IDs to read. If None, reads all data.
            table_name (str): The name of the table to read data from.
            columns (list): A list of columns to include in the returned data. By default, all columns are included.
            include_cols (bool): If True, includes the only the fields listed in columns
                If False, includes all fields except the ones listed in columns.
            deserialize_data (bool): If True, deserializes the data in a multiprocessing fashion.
            filter_func (function): A function to filter the data column while deserialing. 
            It should operate on a dataframe and return the modifies dataframe
            unpack_filter_results (bool): If True, unpacks the filter results.

        Returns:
            pandas.DataFrame or list: The data read from the database. If deserialize_data is True,
            returns a list of dictionaries with their 'id's. Otherwise, returns the DataFrame with serialized data.
        """
        self._validate_output_format(output_format)
        # Check if the table name is in the list of table names
        # self._check_table_name(table_name)
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        columns_to_read=columns
        if columns:
            columns_to_read = get_field_names(table_path, columns=columns, include_cols=include_cols)

        df = self._load_data(table_path, columns=columns_to_read)

        if ids is not None and 'id' in df.columns:
            df = df[df['id'].isin(ids)]

        if filter_func:
            df=filter_func(df)

        if filter_args:
            df=self._filter_data(df, filter_args)

        if output_format=='pandas':
            return df
        elif output_format=='pyarrow':
            return pa.Table.from_pandas(df)

    

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
            df=pd.DataFrame(columns=['id'])
            self._save_data(df, table_path)
            return df

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
        

        original_schema = self.get_schema(table_name)

        incoming_field_names=set(df.columns)
        orginal_field_names=set(original_schema.names)

        new_field_names=list(incoming_field_names - orginal_field_names)
        field_names_missing_from_original=list(orginal_field_names - incoming_field_names)

        # Creating new schema if there are new fields
        new_schema=original_schema
        for new_field_name in new_field_names:
            new_field_type=pa.infer_type(df[new_field_name])
            new_schema = new_schema.append(pa.field(new_field_name, new_field_type))

        logger.debug(f"Incoming field names: {incoming_field_names}")
        logger.debug(f"Original field names: {orginal_field_names}")
        logger.debug(f"New field names: {new_field_names}")
        logger.debug(f"Field names Income is missing compared to original: {field_names_missing_from_original}")

        new_table_path = os.path.join(self.db_path, f'{table_name}_new.parquet')
        writer = pq.ParquetWriter(new_table_path, new_schema)
        geneator = self.get_batch_generator(table_name=table_name, batch_size=batch_size)
        for batch_df in geneator:
            for field_name in new_field_names:
                if field_name not in batch_df.columns:
                    batch_df[field_name] = None
            table = pa.Table.from_pandas(batch_df, schema=new_schema)

            writer.write_table(table)

        for field_name in field_names_missing_from_original:
            df[field_name]=None
        table = pa.Table.from_pandas(df, schema=new_schema)

        logger.debug(f"Difference between Incoming and new schema: {find_difference_between_pyarrow_schemas(table.schema, new_schema)}")
        logger.debug(f"Schema Difference between new and Incoming schema: {find_difference_between_pyarrow_schemas(new_schema, table.schema)}")

        writer.write_table(table)
        writer.close()

        os.remove(table_path)
        os.rename(new_table_path, table_path)

        logger.info(f"Parquet file {table_path} created successfully in batches.")

    def _create_without_batches(self, df, table_name='main'):
        logger.info("Creating new data without batches.")

        table_path=os.path.join(self.db_path, f'{table_name}.parquet')
        original_df = self._load_data(table_path)

        # Determine what dataframes are missing what fields then solving
        original_schema = self.get_schema(table_name)

        incoming_field_names=set(df.columns)
        orginal_field_names=set(original_schema.names)

        new_field_names=list(incoming_field_names - orginal_field_names)
        field_names_missing_from_original=list(orginal_field_names - incoming_field_names)

        logger.debug(f"Incoming field names: {incoming_field_names}")
        logger.debug(f"Original field names: {orginal_field_names}")
        logger.debug(f"New field names: {new_field_names}")
        logger.debug(f"Field names Income is missing compared to original: {field_names_missing_from_original}")

        # Update the original schema with new incoming fields
        for field_name in new_field_names:
            original_df[field_name]=None
        
        # Add missing fields from the original schema to the incoming data
        for field_name in field_names_missing_from_original:
            df[field_name]=None

        # Append new data to the existing data
        df = pd.concat([original_df, df], ignore_index=True)
        # Save the data
        self._save_data(df,table_path=table_path)

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
    
    def _validate_output_format(self, output_format):
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
    def _filter_data(self, df, filter_tuple):
        filter_string= "x {} {}".format( filter_tuple[1], filter_tuple[2])
        filtered_df = df[df[filter_tuple[0]].apply(lambda x: eval(filter_string))]
        return filtered_df

def test_filter_func(df):
    filtered_df = df[df['band_gap'].apply(lambda x: x == 1.593)]
    return filtered_df
    
     
if __name__ == '__main__':
    save_dir='data/raw/ParquetDB'
    db = ParquetDatabase(db_path=save_dir)

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

    data_1['field2']=1
    data_2['field2']=2

    for i in range(1000):
        if i%2==0:
            data_list.append(data_1)
        else:
            data_list.append(data_2)
# 
    # print(db.get_schema('main'))
    # db.create(data=data_list, batch_size=100)
    # db.create(data=data_list)

    # df=db.read(table_name='main')
    # print(df.columns)
    # print(df.head())
    # print(df.tail())
    # print(df['field1'])
    # print(df['field2'])
    # print(df.shape)


    # data_1['field2']=1
    # data_2['field2']=2

    # for i in range(1000):
    #     if i%2==0:
    #         data_list.append(data_1)
    #     else:
    #         data_list.append(data_2)

    # db.create(data=data_list, batch_size=100)

    
    ################################################################################################
    # Testing Read functionality
    ################################################################################################
    # # Testing ids
    # df=db.read(ids=[1,2], table_name='main')
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # # Testing columns include_cols
    # df=db.read(table_name='main', columns=['volume'], include_cols=True)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # # Testing columns include_cols False
    # df=db.read(table_name='main', columns=['volume'], include_cols=False)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # # # Testing columns filter_func
    # df=db.read(table_name='main', filter_func=test_filter_func)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    # # # # Testing columns filter_args
    # df=db.read(table_name='main', filter_args=('band_gap',  '==', 1.593))
    # print(df.head())
    # print(df.tail())
    # print(df.shape)

    ################################################################################################
    
    df=db.read(table_name='main', output_format='pandas')

    # Converting table to dataframe take 0.52 seconds with 1000 rows
    start_time=time.time()
    table=pa.Table.from_pandas(df)
    print("Time taken to convert pandas dataframe to pyarrow table: ", time.time() - start_time)
    
    # Converting table to dataframe take 0.42 seconds with 1000 rows
    start_time=time.time()
    df=table.to_pandas()
    print("Time taken to convert pyarrow table to pandas dataframe: ", time.time() - start_time)

    # Converting table to dataset take 0.16 seconds with 1000 rows
    # table=pa.Table.from_pandas(df)
    dataset_dir=os.path.join(save_dir,'datasets','main')
    # start_time=time.time()
    ds.write_dataset(table,  
                     base_dir=dataset_dir, 
                     basename_template='main_{i}.parquet',
                     format="parquet",
                    #  partitioning=ds.partitioning(
                    #     pa.schema([table.schema.field(dataset_field)])),
                     max_partitions=1024,
                     max_open_files=1024,
                     max_rows_per_file=200, # Controls the number of rows per parquet file
                     min_rows_per_group=0,
                     max_rows_per_group=200, # This must be less than or equal to max_rows_per_file
                     existing_data_behavior='error', # 'error', 'overwrite_or_ignore', 'delete_matching'
                     )
    dataset = ds.dataset(dataset_dir, format="parquet")
    print("Time taken to convert pandas dataframe to pyarrow table and write to parquet: ", time.time() - start_time)


    
    
    
    
    
    
    
    
    
    # df=db.read(table_name='main')
    # print(df.columns)
    # print(df.head())
    # print(df.tail())
    # print(df.shape)
    # print(df['field1'])
    # df=db.read(table_name='main')

    # data=df.iloc[1]['data']
    # lattice=data['structure']['lattice']['matrix']
    # for i in range(3):
    #     print(lattice[i])
    # print(type(lattice))
    # print(lattice.shape)
    # print(data)

    # df=db.read(table_name='main_new')
    # print(df.columns)
    # # print(df.head())
    # # print(df.tail())
    # print(df.shape)

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
    # batch_df=next(db.get_batch_generator(batch_size=100))
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








    #####################################################################################
    # # Testing pyarrow dataset formats 
    #####################################################################################


    # # # Testing output_format
    # table=db.read(table_name='main', output_format='pyarrow')
    # print(table)


    # # # Testing output_format
    # table=db.read(table_name='main', output_format='pyarrow')
    # print(table)
    # # Testing dataset partitioning
    # datasets_dir=os.path.join(save_dir,'datasets')
    # os.makedirs(datasets_dir, exist_ok=True)
    # dataset_field='nelements'
    # dataset_dir=os.path.join(datasets_dir,dataset_field)

    # if not os.path.exists(dataset_dir):
        # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.partitioning.html
        # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
        # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset.to_table
        # ds.write_dataset(table, dataset_dir, format="parquet",
        #             partitioning=ds.partitioning(
        #                 pa.schema([table.schema.field(dataset_field)])
        #             ))
        # ds.write_dataset(table,  
        #                 base_dir=dataset_dir, 
        #                 basename_template='main_{i}.parquet',
        #                 format="parquet",
        #                 max_partitions=1024,
        #                 max_open_files=1024,
        #                 max_rows_per_file=200, # Controls the number of rows per parquet file
        #                 min_rows_per_group=0,
        #                 max_rows_per_group=200, # This must be less than or equal to max_rows_per_file
        #                 existing_data_behavior='error', # 'error', 'overwrite_or_ignore', 'delete_matching'
        #                 )
    
    # dataset = ds.dataset(dataset_dir, format="parquet", partitioning=[dataset_field])
    # print(dataset)
    # print(type(dataset))
    # print(dataset.files)
    # # Filtering example
    # table=dataset.to_table(columns=['volume','nelements'], filter=ds.field('nelements') == 2)
    # print(table)


    # # Projecting example
    # projection = {
    #     "volume_float32": ds.field("volume").cast("float32"),
    #     "nelements_int64": ds.field("nelements").cast("int64"),
    # }

    # table=dataset.to_table(columns=projection)
    # print(table)

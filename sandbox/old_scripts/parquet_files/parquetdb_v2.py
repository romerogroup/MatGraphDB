import logging
import os
import shutil
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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

def is_directory_empty(directory_path):
    return not os.listdir(directory_path)

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
        self.tmp_dir=os.path.join(self.datasets_dir,'tmp')

        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        self.n_cores = n_cores

        self.output_formats=['batch_generator','table','dataset']
        self.reserved_table_names=['tmp']
        self.table_names=os.listdir(self.datasets_dir)
        

        self.metadata = {}
        logger.info(f"db_path: {self.db_path}")
        logger.info(f"table_names: {self.table_names}")
        logger.info(f"reserved_table_names: {self.reserved_table_names}")
        logger.info(f"n_cores: {self.n_cores}")
        logger.info(f"output_formats: {self.output_formats}")

    def get_schema(self, table_name:str ='main'):
        schema = self._load_data(table_name=table_name, output_format='dataset').schema
        return schema
        
    @timeit
    def create(self, data:Union[List[dict],dict,pd.DataFrame], 
               table_name:str='main', 
               batch_size:int=None,
               max_rows_per_file=10000,
               min_rows_per_group=0,
               max_rows_per_group=10000,
               schema=None,
               metadata=None,
               **kwargs):
        """
        Adds new data to the database.

        Args:
            data (dict or list of dicts): The data to be added to the database. 
                This must contain
            table_name (str): The name of the table to add the data to.
            batch_size (int): The batch size. 
                If provided, create will return a generator that yields batches of data.
            max_rows_per_file (int): The maximum number of rows per file.
            min_rows_per_group (int): The minimum number of rows per group.
            max_rows_per_group (int): The maximum number of rows per group.
            schema (pyarrow.Schema): The schema of the incoming table.
            metadata (dict): Metadata to be added to the table.
            **kwargs: Additional keyword arguments to pass to the create function.
        """
        
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        os.makedirs(dataset_dir, exist_ok=True)
        self._check_table_name(table_name)
        
        n_files=len(os.listdir(dataset_dir))

        original_files=glob(os.path.join(dataset_dir,'*.parquet'))
        
        # Prepare the data and field data
        data_list=self._validate_data(data)
        
        # Get new ids
        new_ids = self._get_new_ids(table_name, data_list)

        incoming_table=pa.Table.from_pylist(data_list, schema=schema, metadata=metadata)
        incoming_table=incoming_table.append_column(pa.field('id', pa.int64()), [new_ids])
        incoming_schema=incoming_table.schema

        first_table=pq.read_table(os.path.join(dataset_dir,f'{table_name}_0.parquet'))
        original_schema=first_table.schema

        incoming_field_names=set(incoming_schema.names)
        orginal_field_names=set(original_schema.names)

        field_names_original_is_missing=list(incoming_field_names - orginal_field_names)
        field_names_incoming_is_missing=list(orginal_field_names - incoming_field_names)

        logger.info(f"Field names original is missing: {field_names_original_is_missing}")
        logger.info(f"Field names incoming is missing: {field_names_incoming_is_missing}")

        original_column_names=original_schema.names
        original_column_names.extend(field_names_original_is_missing)

        if field_names_original_is_missing:
            for original_file in original_files:
                original_table=pq.read_table(original_file)
                for new_field_name in field_names_original_is_missing:
                    # Get the expected data type from the new schema
                    field_type = incoming_table.field(new_field_name).type
                    # Create a null array with the correct data type
                    null_array = pa.nulls(original_table.shape[0], type=field_type)
                    # Append the column to the table (note that append_column returns a new table)
                    original_table = original_table.append_column(new_field_name, null_array)
                pq.write_table(original_table, original_file)

        
        for field_name in field_names_incoming_is_missing:
            # Get the expected data type from the new schema
            field_type = original_schema.field(field_name).type
            # Create a null array with the correct data type
            null_array = pa.nulls(incoming_table.shape[0], type=field_type)
            # Append the column to the table (note that append_column returns a new table)
            incoming_table = incoming_table.append_column(field_name, null_array)

        incoming_table=incoming_table.select(original_column_names)

        if first_table.shape[0]==0:
            incoming_save_path=os.path.join(dataset_dir,f'{table_name}_0.parquet')
        else:
            incoming_save_path=os.path.join(dataset_dir,f'{table_name}_{n_files}.parquet')

        pq.write_table(incoming_table, incoming_save_path)

        if batch_size:
            new_schema=incoming_table.schema
            output_format='batch_generator'
        else:
            output_format='table'
            new_schema=None

        final_table=self._load_data(table_name, batch_size=batch_size, output_format=output_format)
        basename_template=f'{table_name}'+'_{i}.parquet'

        self._write_tmp_files(table_name)
        try:
            logger.info(f"Writing final table to {dataset_dir}")
            ds.write_dataset(final_table,  
                            dataset_dir, 
                            basename_template=basename_template,
                            schema=new_schema,
                            format="parquet",
                            max_partitions=kwargs.get('max_partitions',1024),
                            max_open_files=kwargs.get('max_open_files',1024),
                            max_rows_per_file=max_rows_per_file, 
                            min_rows_per_group=min_rows_per_group,
                            max_rows_per_group=max_rows_per_group,
                            existing_data_behavior='overwrite_or_ignore',
                            )
        except Exception as e:
            logger.error(f"Error writing final table to {dataset_dir}: {e}")
            # If something goes wrong, restore the original files
            self._restore_tmp_files(table_name)

            raise e
    
    @timeit
    def read(self, ids=None, table_name:str='main', 
                columns:List[str]=None, 
                include_cols:bool=True, 
                filters=[],
                output_format='table',
                batch_size=None):
        """
        Reads data from the database.

        Args:
            ids (list): A list of IDs to read. If None, reads all data.
            table_name (str): The name of the table to read data from.
            columns (list): A list of columns to include in the returned data. By default, all columns are included.
            include_cols (bool): If True, includes the only the fields listed in columns
                If False, includes all fields except the ones listed in columns.
            filters (List): A list of fliters to apply to the data.
            It should operate on a dataframe and return the modifies dataframe
            batch_size (int): The batch size. 
                If provided, read will return a generator that yields batches of data.

        Returns:
            pandas.DataFrame or list: The data read from the database. If deserialize_data is True,
            returns a list of dictionaries with their 'id's. Otherwise, returns the DataFrame with serialized data.
        """
        self._check_table_name(table_name)

        # Check if the table name is in the list of table names
        table_path=os.path.join(self.db_path, f'{table_name}.parquet')

        columns_to_read=columns
        if columns:
            columns_to_read = get_field_names(table_path, columns=columns, include_cols=include_cols)

        final_filters=[]
        
        if ids:
            id_filter=pc.field('id').isin(ids)
            final_filters.append(id_filter)

        for filter in filters:
            final_filters.append(filter)

        filter_expression=None
        for filter in final_filters:
            if filter_expression is not None:
                filter_expression=(filter_expression & filter)
            else:
                filter_expression=filter

        data = self._load_data(table_name=table_name, columns=columns_to_read, filter=filter_expression, 
                               batch_size=batch_size, output_format=output_format)
        return data
    
    @timeit
    def update(self, data: Union[List[dict], dict, pd.DataFrame], table_name='main', field_type_dict=None):
        """
        Updates data in the database.

        Args:
            data (dict or list of dicts or pandas.DataFrame): The data to be updated.
                Each dict should have an 'id' key corresponding to the record to update.
            table_name (str): The name of the table to update data in.
            field_type_dict (dict): A dictionary where the keys are the field names and the values are the new field types.

            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If new fields are found in the update data that do not exist in the schema.
        """
        self._check_table_name(table_name)

        # Data processing and validation.
        data_list = self._validate_data(data)

        main_id_column=self._load_data(table_name='main', columns=['id'], output_format='table')['id'].to_pylist()

        update_dict={}
        incoming_field_names = set()
        update_ids=[]
        infered_types={}
        for data_dict in data_list:
            id=data_dict.get('id', None)
            if id is None:
                raise ValueError("Each data dict must have an 'id' key.")
            if id not in main_id_column:
                raise ValueError(f"The id {id} is not in the main table. It must have been deleted at an earlier time or the id is incorrect.")
            update_dict.update({data_dict['id']: data_dict})
            incoming_field_names.update(data_dict.keys())
            update_ids.append(data_dict['id'])
            
            # Detemine the infered types for each field
            for key,value in data_dict.items():
                if key=='id':
                    continue
                infered_types[key]=pa.infer_type([value])

        infered_types=self._check_infered_field_types_vs_original_field_types(table_name, infered_types)
        infered_types.update(field_type_dict)

        logger.info(f"Found these keys in the update data: {incoming_field_names}")
        logger.debug(f"The infered types are: {infered_types}")

        dataset_dir=os.path.join(self.datasets_dir,table_name)
        logger.info(f"Dataset directory: {dataset_dir}")
        original_files=glob(os.path.join(dataset_dir,f'{table_name}_*.parquet'))

        # Iterate over the original files
        self._write_tmp_files(table_name)
        for i_file, original_file in enumerate(original_files):
            original_table=pq.read_table(original_file)
            id_column=original_table['id']
            filename=os.path.basename(original_file)

            logger.debug(f"Processing file {filename}. It has a shape of {original_table.shape}")

            # Get the new field names
            original_field_names=set(original_table.column_names)
            new_field_names=list(incoming_field_names - original_field_names)

            # Add new column if there is a new field to add
            for new_field_name in new_field_names:
                logger.debug(f"Found new_field name. Adding {new_field_name} field to {filename}")
                # Get the expected data type from the new schema
                field_type = infered_types[new_field_name]
                # Create a null array with the correct data type
                null_array = pa.nulls(original_table.shape[0], type=field_type)
                # Append the column to the table (note that append_column returns a new table)
                original_table = original_table.append_column(new_field_name, null_array)

            updated_schema=original_table.schema
            original_ids_list=id_column.to_pylist()

            logger.debug(f"After adding null columns to {filename} it has a shape of {original_table.shape}")
            # Iterate over the columns in the original table if there are ids to update
            # Get the ids and fields that need updated in the current table
            ids_in_table=[]
            field_names_to_update_in_table=[]
            for id in update_ids:
                if id in original_ids_list:
                    ids_in_table.append(id)
                    field_names_to_update_in_table.extend(update_dict[id].keys())
            
            if ids_in_table:
                logger.debug(f"Found indices to update. Updating {filename}.")
                for i,column in enumerate(original_table.itercolumns()):
                    column_name=column._name
                    # Skip updating the 'id' column
                    if column_name == 'id':
                        continue
                    
                    # If the column name is in the field_names_to_update_in_table, update the column
                    if column_name in field_names_to_update_in_table:
                        column_array=column.to_pylist()

                        # Update the values at the indices
                        for id in ids_in_table:
                            index = original_ids_list.index(id)
                            update_value=update_dict[id].get(column_name,None)

                            # If the update value is None for a column, skip it. 
                            # This can happen when a field is added to another file but not the current one
                            if update_value is None:
                                continue

                            column_array[index] = update_value

                        field=updated_schema.field(column_name)
                        original_table=original_table.set_column(i, field, [column_array])
            new_table=original_table

            # Saving the updated table
            try:
                pq.write_table(new_table, original_file)
            except Exception as e:
                logger.error(f"Error processing {original_file}: {e}")
                # If something goes wrong, restore the original file
                self._restore_tmp_files(table_name)

                raise e

            logger.info(f"Updated {filename} with {original_table.shape}")
             

        logger.info(f"Updated {table_name} table.")

    @timeit
    def delete(self, ids:List[int], table_name:str='main'):
        """
        Deletes data from the database.

        Args:
            ids (list): A list of IDs to delete.
            table_name (str): The name of the table to delete data from.

        Returns:
            None
        """
        self._check_table_name(table_name)
        logger.info(f"Deleting data from {table_name}")

        main_id_column=self._load_data(table_name='main', columns=['id'], output_format='table')
        id_filter = pc.field('id').isin(ids)
        filtered_table = main_id_column.filter(id_filter)

        if filtered_table.num_rows==0:
            logger.info(f"No data found to delete.")
            return None
        
        # Iterate over the original files
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        logger.info(f"Dataset directory: {dataset_dir}")
        original_files=glob(os.path.join(dataset_dir,f'{table_name}_*.parquet'))
        self._write_tmp_files(table_name)
        for i_file, original_file in enumerate(original_files):
            filename=os.path.basename(original_file)

            original_table=pq.read_table(original_file)
            shape_before=original_table.shape

            # Get the original ids
            original_ids_list=original_table['id'].to_pylist()
            
            # Get the ids that are in the current table
            ids_in_table=[]
            for id in ids:
                if id in original_ids_list:
                    ids_in_table.append(id)

            # Applying the negative id filter
            neg_id_filter = ~pc.field('id').isin(ids)
            new_table = original_table.filter(neg_id_filter)

            shape_after=new_table.shape
            n_rows_deleted=shape_before[0]-shape_after[0]

            # Write the updated table to the original file
            logger.info(f"Writing updated table to {filename}")


            # Saving the updated table
            try:
                pq.write_table(new_table, original_file)

            except Exception as e:
                logger.error(f"Error processing {original_file}: {e}")
                # If something goes wrong, restore the original file
                self._restore_tmp_files(table_name)
                break
            
            logger.info(f"Deleted {n_rows_deleted} Indices from {filename}. The shape is now {new_table.shape}")

        logger.info(f"Updated {table_name} table.")

    def update_schema(self, table_name, field_dict=None, schema=None):
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        os.makedirs(dataset_dir, exist_ok=True)
        original_files=glob(os.path.join(dataset_dir,f'{table_name}_*.parquet'))
        self._write_tmp_files(table_name)
        for i_file, original_file in enumerate(original_files):
            filename=os.path.basename(original_file)
            original_table=pq.read_table(original_file)
            
            original_schema=original_table.schema
            original_field_names=original_table.names

            if field_dict:
                new_schema=original_schema
                for field_name, new_field in field_dict.items():
                    field_index=original_schema.get_field_index(field_name)

                    if field_name in original_field_names:
                        new_schema=new_schema.set(field_index, new_field)

            if schema:
                new_schema=schema

            pylist=original_table.to_pylist()
            new_table=pa.Table.from_pylist(pylist, schema=new_schema)

            # Saving the updated table
            try:
                pq.write_table(new_table, original_file)
            except Exception as e:
                logger.error(f"Error processing {original_file}: {e}")
                # If something goes wrong, restore the original file
                self._restore_tmp_files(table_name)
                break
            logger.info(f"Updated {filename} with {original_table.shape}")
        logger.info(f"Updated Fields in {table_name} table.")

                
            


    @timeit
    def _load_data(self, table_name, columns=None, filter=None, batch_size=None, output_format='table'):
        """
        This method loads the data in the database. It can either load the data as a PyArrow Table, PyArrow Dataset, PyArrow generator.
        """
        logger.info(f"Loading data from {table_name}")
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        os.makedirs(dataset_dir, exist_ok=True)

        is_empty=is_directory_empty(dataset_dir)
        if is_empty:
            logger.info(f"No data found at {dataset_dir}, creating an empty parquet file with id column.")
            schema=pa.schema([('id', pa.int64())])
            table = pa.Table.from_batches([], schema=schema)
            basename=f'{table_name}'+'_0.parquet'
            pq.write_table(table, os.path.join(dataset_dir, basename))

        dataset = ds.dataset(dataset_dir, format="parquet")
        if output_format=='batch_generator':
            if batch_size is None:
                raise ValueError("batch_size must be provided when output_format is batch_generator")
            logger.info(f"Loading data from {dataset_dir} in batches")
            logger.info(f"Loading only columns: {columns}")
            logger.info(f"Using filter: {filter}")
            return dataset.to_batches(columns=columns,filter=filter,batch_size=batch_size)
        elif output_format=='table':
            logger.info(f"Loading data from {dataset_dir}")
            logger.info(f"Loading only columns: {columns}")
            logger.info(f"Using filter: {filter}")
            return dataset.to_table(columns=columns,filter=filter)
        elif output_format=='dataset':
            return dataset
        else:
            raise ValueError(f"output_format must be one of the following: {self.output_formats}")
 
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
    
    @timeit
    def _get_new_ids(self, table_name, data_list):
        table = self._load_data(table_name, columns=['id'],output_format='table')
        if table.num_rows==0:
            start_id = 0
        else:
            max_val=pc.max(table.column('id')).as_py()
            start_id = max_val + 1  # Start from the next available ID
    
        # Create a list of new IDs
        new_ids = list(range(start_id, start_id + len(data_list)))
        return new_ids
    
    @timeit
    def _write_tmp_files(self, table_name):
        shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        original_filepaths=glob(os.path.join(dataset_dir,f'{table_name}_*.parquet'))
        for i_file, original_filepath in enumerate(original_filepaths):
            basename=os.path.basename(original_filepath)
            
            tmp_filepath = os.path.join(self.tmp_dir, basename)
            shutil.copyfile(original_filepath, tmp_filepath)

    @timeit
    def _restore_tmp_files(self, table_name):
        dataset_dir=os.path.join(self.datasets_dir,table_name)
        tmp_filepaths=glob(os.path.join(self.tmp_dir,f'{table_name}_*.parquet'))
        for i_file, tmp_filepath in enumerate(tmp_filepaths):
            basename=os.path.basename(tmp_filepath)
            original_filepath = os.path.join(dataset_dir, basename)
            shutil.copyfile(tmp_filepath, original_filepath)
            os.remove(tmp_filepath)
    
    @timeit
    def _check_table_name(self, table_name, by_pass_existence_check=False):
        if table_name in self.reserved_table_names:
            raise ValueError(f"Table name {table_name} is reserved. Please choose a different name.")
        
        # if not by_pass_existence_check:
        if table_name not in os.listdir(self.datasets_dir):
            raise ValueError(f"Table name {table_name} not found in the database.")
    
    @timeit
    def _check_infered_field_types_vs_original_field_types(self, table_name, infered_types):
        logger.info(f"Checking infered field types vs original field types")
        schema=self.get_schema(table_name=table_name)

        for key, value in infered_types.items():
            if key in schema.names:
                type=schema.field(key).type
                if type!=value:
                    logger.info(f"The infered type for {key} is {type} but the original type is {value}")
                    infered_types[key]=type
                    logger.info(f"Replacing infered type for {key} with {type}")
        return infered_types


    @timeit
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


    data_1['field5']=1
    data_2['field5']=2
    data_list=[]
    for i in range(1000):
        if i%2==0:
            data_list.append(data_1)
        else:
            data_list.append(data_2)
    # db.create(data=data_list, batch_size=100)
    # db.create(data=data_list, table_name='main_test')

    # dataset_dir=os.path.join(save_dir,'datasets','main')
    # for i in range(len(os.listdir(dataset_dir))):
    #     table=pq.read_table(os.path.join(dataset_dir,f'main_{i}.parquet'))
    #     print(table.shape)
    #     # print(table.column_names)


    # filters=[pc.equal(pc.field('band_gap'), 1.593)]
    # table=db.read(table_name='main', ids = [0,1,2,3,4], filters=filters)

    table=db.read(table_name='main_test', columns=['id'])
    df=table.to_pandas()
    print(df.columns)
    print(df.head())
    print(df.tail())

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
    # Testing Update functionality
    ################################################################################################

    # data_list=[]
    # for i in range(1000):
    #     data={}
    #     data['id']=i
    #     data['field5']=5
    #     data_list.append(data)

    # data_list=[{'id':1, 'field5':5, 'field6':6},
    #            {'id':2, 'field5':5, 'field6':6},
    #            {'id':1000, 'field5':5, 'field8':6},
    #            ]

    # db.update(data_list)

    # table=db.read(ids=[1,2,1000],  table_name='main', columns=['id','field5','field6', 'field8'])
    # print(table)


    ################################################################################################
    # Testing Delete functionality
    ################################################################################################

    # Deleting rows in table
    # db.delete(ids=[1,3,1000,2])
    # db.delete(ids=[2])
    # table=db.read(ids=[1,3,1000,2],  table_name='main', columns=['id','field5','field6', 'field8'])
    # print(table)


    # Trying to delete a row that does not exist
    # db.delete(ids=[10000000])

    ################################################################################################
    # Testing Read functionality
    ################################################################################################
    # # Basic read
    # table=db.read( table_name='main')
    # df=table.to_pandas()
    # print(df.head())
    # print(df.tail())
    # print(df.shape)
    
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
    # db._load_data('main')
    # df=db.read(table_name='main', output_format='pandas')

    # # Converting table to dataframe take 0.52 seconds with 1000 rows
    # start_time=time.time()
    # table=pa.Table.from_pandas(df)
    # print("Time taken to convert pandas dataframe to pyarrow table: ", time.time() - start_time)
    
    # # Converting table to dataframe take 0.42 seconds with 1000 rows
    # start_time=time.time()
    # df=table.to_pandas()
    # print("Time taken to convert pyarrow table to pandas dataframe: ", time.time() - start_time)

    # # Converting table to dataset take 0.16 seconds with 1000 rows
    # # table=pa.Table.from_pandas(df)
    # dataset_dir=os.path.join(save_dir,'datasets','main')
    # # start_time=time.time()
    # # ds.write_dataset(table,  
    # #                  base_dir=dataset_dir, 
    # #                  basename_template='main_{i}.parquet',
    # #                  format="parquet",
    # #                 #  partitioning=ds.partitioning(
    # #                 #     pa.schema([table.schema.field(dataset_field)])),
    # #                  max_partitions=1024,
    # #                  max_open_files=1024,
    # #                  max_rows_per_file=200, # Controls the number of rows per parquet file
    # #                  min_rows_per_group=0,
    # #                  max_rows_per_group=200, # This must be less than or equal to max_rows_per_file
    # #                  existing_data_behavior='error', # 'error', 'overwrite_or_ignore', 'delete_matching'
    # #                  )
    # dataset = ds.dataset(dataset_dir, format="parquet")
    # print("Time taken to convert pandas dataframe to pyarrow table and write to parquet: ", time.time() - start_time)


    
    
    
    
    
    
    
    
    
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

from multiprocessing import Pool
import os
import logging
from typing import Callable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from functools import partial

from matgraphdb.data.material_manager import MaterialDatabaseManager
from matgraphdb.data.calc_manager import CalculationManager
from matgraphdb.graph_kit.graph_manager import GraphManager
from matgraphdb.utils import N_CORES

logger = logging.getLogger(__name__)

t_string=pa.string()
t_int=pa.int64()
t_float=pa.float64()
t_bool=pa.bool_()

def calculation_error_handler(calc_func: Callable):
    """
    A decorator that wraps the user-defined calculation function in a try-except block.
    This allows graceful handling of any errors that may occur during the calculation process.
    
    Args:
        calc_func (Callable): The user-defined calculation function to be wrapped.
    
    Returns:
        Callable: A wrapped function that executes the calculation and handles any exceptions.
    """
    def wrapper(data):
        try:
            # Call the user-defined calculation function
            return calc_func(data)
        except Exception as e:
            # Log the error (you could customize this as needed)
            logger.error(f"Error in calculation function: {e}\nData: {data}")
            # Optionally, you could return a default or partial result, or propagate the error
            return {}
    
    return wrapper


def disk_calculation(disk_calc_func: Callable, base_directory: str):
    """
    A decorator that wraps the disk-based calculation function in a try-except block.
    It also ensures that the specified directory for each row (ID) and calculation name 
    is created before the calculation function is executed.

    Args:
        disk_calc_func (Callable): The user-defined disk-based calculation function to be wrapped.
        base_directory (str): The base directory where files related to the calculation will be stored.
        verbose (bool): If True, prints error messages. Defaults to False.

    Returns:
        Callable: A wrapped function that handles errors and ensures the calculation directory exists.
    """

    def wrapper(row_data, row_id, calculation_name):
        # Define the directory for this specific row (ID) and calculation
        material_specific_directory = os.path.join('MaterialsData',row_id,calculation_name)
        calculation_directory = os.path.join(base_directory, material_specific_directory)
        os.makedirs(calculation_directory, exist_ok=True)

        try:
            # Call the user-defined calculation function, passing the directory as a parameter
            return disk_calc_func(row_data, directory=calculation_directory)
        except Exception as e:
            # Log the error
            logger.error(f"Error in disk calculation function: {e}\nData: {row_data}")
            # Optionally, return a default or partial result, or propagate the error
            return {}

    return wrapper

class MatGraphDB:
    """
    A handler class for managing material repository operations, including 
    database interactions, running calculations, and handling schema for Parquet files.

    Attributes:
        main_directory (str): The main directory where the calculations, schema, and database are stored.
        calculation_directory (str): Directory path for storing calculations.
        db_path (str): Path to the SQLite database file used for material data.
        n_cores (int): Number of cores to use for parallel processing.
        manager (MaterialDatabaseManager): Instance of MaterialDatabaseManager for database operations.
        calc_manager (CalculationManager): Manager for handling material-related calculations.
        parquet_schema_file (str): Path to the Parquet file where the material schema is stored.
    """
    def __init__(self, main_dir: str,
                 calculation_dirname='calculations',
                 graph_dirname='graph_database',
                 db_file='materials.db', 
                 n_cores=N_CORES,
                 **kwargs):
        """
        Initializes the MaterialRepositoryHandler by setting up directories, 
        the database manager, and the calculation manager.

        Args:
            main_directory (str): Path to the main directory where all data will be stored.
            calculation_dirname (str): Subdirectory name for storing calculations. Defaults to 'calculations'.
            db_file (str): Filename for the SQLite database. Defaults to 'materials.db'.
            n_cores (int): Number of CPU cores to use for parallel processing.
            **kwargs: Additional keyword arguments for configuring the CalculationManager.
        """
        logger.info("Initializing MaterialRepositoryHandler.")
        # Set up directories and database path
        self.main_dir = main_dir
        self.calculation_dir=os.path.join(self.main_dir, calculation_dirname)
        self.graph_dir=os.path.join(self.main_dir, graph_dirname)
        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        logger.debug(f"Main directory set to {self.main_dir}")
        logger.debug(f"Calculation directory set to {self.calculation_dir}")
        logger.debug(f"Graph directory set to {self.graph_dir}")

        self.db_path = os.path.join(main_dir, db_file)
        self.n_cores = n_cores

        logger.debug(f"Database path set to {self.db_path}")
        logger.debug(f"Number of cores set to {self.n_cores}")


        self.db_manager = MaterialDatabaseManager(db_path=self.db_path)
        logger.debug("MaterialDatabaseManager initialized.")

        self.calc_manager = CalculationManager(main_dir=self.calculation_dir, 
                                               db_manager=self.db_manager, 
                                               n_cores=self.n_cores,
                                               **kwargs)
        logger.debug("CalculationManager initialized.")

        self.graph_manager = GraphManager(graph_dir=self.graph_dir)
        logger.debug("GraphManager initialized.")

        self.parquet_schema_file = os.path.join(main_dir, 'material_schema.parquet')
        logger.debug(f"Parquet schema file set to {self.parquet_schema_file}")

    def _process_task(self, func, list, **kwargs):
        """
        Processes tasks in parallel using a pool of worker processes.

        Args:
            func (Callable): The function to be applied to each item in the list.
            list (list): A list of items to be processed by the function.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            list: The results of applying the function to each item in the input list.
        """
        logger.info(f"Processing tasks in parallel using {self.n_cores} cores.")
        with Pool(self.n_cores) as p:
            results=p.map(partial(func,**kwargs), list)
        logger.info("Tasks processed successfully.")
        return results
    
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
    
    def create_parquet_from_data(self, func:Callable, schema, output_filename: str =  'materials_database.parquet' ):
        """
        Creates a Parquet file from data in the material database, transforming each row using the provided function.

        Args:
            func (Callable): A function that processes each row of data from the database and returns a tuple of 
                            column names and corresponding values.
            schema (pyarrow.Schema): The schema for the output Parquet file.
            output_file (str): Path to the output Parquet file. Defaults to 'materials_database.parquet'.

        Raises:
            Exception: If there is an issue processing rows.

        Returns:
            None
        """
        output_file=os.path.join(self.main_dir, output_filename)
        logger.info(f"Creating Parquet file from data: {output_file}")
        error_message = "\n".join([
                "Make the function return a tuple with column_names and values.",
                "Also make sure the order of the column_names and values correspond with each other.",
                "I would also recommend using .get({column_name}, None) to get the value of a column if it exists.",
                "Indexing would maybe cause errors if the property does not exist for a material."
            ])
        rows=self.db_manager.read()

        processed_data = []
        column_names=None
        for row in rows:
            try:
                column_names, row_data=func(row.data)
                logger.debug(f"Processed row ID {row.id}: {row_data}")
            except Exception as e:
                logger.error(f"Error processing row ID {row.id}: {e}")
                logger.debug(f"Row data: {row.data}")
                logger.debug(error_message)

            processed_data.append(row_data)

        # Convert the processed data into a DataFrame for Parquet export
        df = pd.DataFrame(processed_data, columns=column_names)


        # Write the DataFrame to a Parquet file
        df.to_parquet(output_file, engine='pyarrow', schema=schema, index=False)
        logger.info(f"Data exported to {output_file}")

        self.set_schema(schema)
        logger.info(f"Schema file updated and saved to {self.parquet_schema_file}")
        
    def run_inmemory_calculation(self, calc_func:Callable, save_results=False, verbose=False, **kwargs):
        """
        Runs a calculation on the data retrieved from the MaterialDatabaseManager.

        This method is designed to process data from the material database using a user-defined 
        calculation function. The `calc_func` is expected to be a callable that accepts a 
        dictionary-like object (each row's data) and returns a dictionary representing the results 
        of the calculation. The results can optionally be saved back to the database.

        Args:
            calc_func (Callable): A function that processes each row of data. This function should 
                                accept a dictionary-like object representing the row's data and return 
                                a dictionary containing the calculated results for that row.
            save_results (bool): A flag indicating whether to save the results back to the database 
                                after processing. Defaults to True.
            verbose (bool): A flag indicating whether to print error messages. Defaults to False.
            **kwargs: Additional keyword arguments that are passed to the calculation function.

        Returns:
            list: A list of result dictionaries returned by the `calc_func` for each row in the database.

        Example:
            def my_calc_func(row_data):
                # Perform some calculations on row_data
                return {'result': sum(row_data.values())}
            
            handler.run_calculation(my_calc_func)

        Notes:
            - The data is read from the database using `self.db_manager.read()`.
            - If `save_results` is True, the method will update the database with the results.
            - The `update_many` method is used to update the database by mapping each row's ID 
            to its corresponding calculated result.
        """
        logger.info("Running in-memory calculation on material data.")
        rows=self.db_manager.read()
        ids=[]
        data=[]
        for row in rows:
            ids.append(row.id)
            data.append(row.data)
        logger.debug(f"Retrieved {len(rows)} rows from the database.")

        calc_func = calculation_error_handler(calc_func,verbose=verbose)
        results=self._process_task(calc_func, data, **kwargs)
        logger.info("Calculation completed.")

        if save_results:
            update_list=[(id,result) for id,result in zip(ids,results)]
            self.db_manager.update_many(update_list)
            logger.info("Results saved back to the database.")

        return results
    




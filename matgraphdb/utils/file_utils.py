import json
import logging
import os
import traceback
from typing import List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def load_json(json_file, mode='r', **kwargs):
    """
    Load JSON file.
    
    Args:
        json_file (str): Path to the JSON file.
    
    Returns:
        dict: Dictionary containing bond orders data.

    
    """
    # Filter kwargs for the 'open' function
    open_args = {k: v for k, v in kwargs.items() if k in ['buffering', 'encoding', 
                                                          'errors', 'newline', 
                                                          'closefd', 'opener']}
    
    # Filter kwargs for the 'json.load' function
    json_load_args = {k: v for k, v in kwargs.items() if k in ['cls', 'object_hook', 
                                                               'parse_float', 'parse_int', 
                                                               'parse_constant', 
                                                               'object_pairs_hook']}
    try:
        logger.info(f"Loading data from {json_file}")
        logger.debug(f"open arguments.", extra=open_args)
        logger.debug(f"json.load arguments.", extra=json_load_args)
        with open(json_file, mode, **open_args) as f:
            data = json.load(f, **json_load_args)
    except:
        logger.exception("Error loading JSON file:")
        data = {}
    logger.info(f"Data loaded from {json_file}")
    return data

def save_json(data, json_file, mode='w', indent=4, **kwargs):
    """
    Save JSON data to a file.
    
    Args:
        data (dict): Dictionary containing bond orders data.
        json_file (str): Path to the JSON file.
    """
    # Filter kwargs for the 'open' function
    open_args = {k: v for k, v in kwargs.items() if k in ['buffering', 'encoding', 
                                                          'errors', 'newline', 
                                                          'closefd', 'opener']}
    
    # Filter kwargs for the 'json.dump' function
    json_dump_args = {k: v for k, v in kwargs.items() if k in ['skipkeys', 'ensure_ascii', 'check_circular', 
                                                               'allow_nan', 'cls', 'separators', 
                                                               'default', 'sort_keys']}
    try:
        logger.info(f"Saving data to {json_file}")
        logger.debug(f"open arguments.", extra=open_args)
        logger.debug(f"json.dump arguments.", extra=json_dump_args)
        with open(json_file, mode, **open_args) as f:
            json.dump(data, f, indent=indent, **json_dump_args)
    except Exception as e:
        logger.exception("Error saving JSON file.")
    return None



def load_parquet(parquet_file, **kwargs):
    """
    Load a Parquet file.

    This function loads a Parquet file and returns the data in the specified format. 
    It supports loading the data as a pandas DataFrame, a pyarrow Table, or a dictionary. 
    The `columns` and `filters` parameters can be used to specify the columns and filters 
    to be applied when loading the data.

    Parameters
    ----------
    parquet_file : str
        The path to the Parquet file.
    columns : list, optional
        A list of column names to load. If None, all columns are loaded.
    kwargs
        Additional keyword arguments to pass to the  `pq.read_table`.
    """
    logger.info(f"Loading data from {parquet_file}")
    logger.debug(f"read_table arguments.", extra=kwargs)
    table = pq.read_table(parquet_file, **kwargs)
    logger.info(f"Data loaded from {parquet_file}")
    return table

def save_parquet(data:Union[dict,pd.DataFrame, List[dict]], parquet_file:str,  
                 from_pydict_args:dict=None, 
                 from_pandas_args:dict=None, 
                 from_pylist_args:dict=None,
                 write_args:dict=None):
    """
    Save data to a Parquet file.

    This function takes data in the form of a dictionary, a pandas DataFrame, or a list of dictionaries, 
    converts it into an Apache Arrow Table, and saves it as a Parquet file. Various options for 
    customization of the conversion and writing process can be provided via additional arguments.

    Parameters
    ----------
    data : Union[dict, pd.DataFrame, List[dict]]
        The data to be saved. It can be a dictionary, a pandas DataFrame, or a list of dictionaries.
    parquet_file : str
        The file path where the Parquet file will be saved.
    from_pydict_args : dict, optional
        Additional arguments to be passed to `pa.Table.from_pydict` when `data` is a dictionary. Defaults to None.
    from_pandas_args : dict, optional
        Additional arguments to be passed to `pa.Table.from_pandas` when `data` is a pandas DataFrame. Defaults to None.
    from_pylist_args : dict, optional
        Additional arguments to be passed to `pa.Table.from_pylist` when `data` is a list of dictionaries. Defaults to None.
    write_args : dict, optional
        Additional arguments to be passed to the Parquet writer (`pq.write_table`). Defaults to None.
    Returns
    -------
    pa.Table
        The Apache Arrow Table created from the input data. This can be useful for further processing or testing.
    """
    logger.info(f"Saving data to {parquet_file}")
    if write_args is None:
        write_args={}
    if from_pydict_args is None:
        from_pydict_args={}
    if from_pandas_args is None:
        from_pandas_args={}
    if from_pylist_args is None:
        from_pylist_args={}

    if isinstance(data, dict):
        logger.debug("Converting dictionary to Apache Arrow Table.")
        logger.debug(f"from_pydict arguments.", extra=from_pydict_args)
        table=pa.Table.from_pydict(data, **from_pydict_args)
    elif isinstance(data, pd.DataFrame):
        logger.debug("Converting pandas DataFrame to Apache Arrow Table.")
        logger.debug(f"from_pandas arguments.", extra=from_pandas_args)
        table=pa.Table.from_pandas(data, **from_pandas_args)
    elif isinstance(data, list):
        logger.debug("Converting list of dictionaries to Apache Arrow Table.")
        logger.debug(f"from_pylist arguments.", extra=from_pylist_args)
        table=pa.Table.from_pylist(data, **from_pylist_args)
    else:
        error_message=f"data must be a dictionary, pandas DataFrame, or list of dictionaries. Received type: {type(data)}"
        logger.error(error_message)
        raise ValueError(error_message)

    logger.debug(f"write_table arguments.", extra=write_args)
    pq.write_table(table, parquet_file, **write_args)

    logger.info(f"Parquet file saved to {parquet_file}")
    return table



def print_directory_tree(directory, skip_dirs=None):
    """
    Prints the directory tree structure starting from the given directory.
    
    This function recursively traverses the directory tree starting from the 
    specified root directory and prints the structure in a tree format. It allows 
    skipping specific directories if needed.

    Parameters
    ----------
    directory : str
        The root directory from which to start printing the tree structure.
    skip_dirs : list of str, optional
        A list of directory names to skip during the traversal. If not provided, 
        no directories will be skipped. Defaults to None.

    Returns
    -------
    None
        This function only prints the directory structure and does not return any value.
    """
    logger.info(f"Printing directory tree structure starting from {directory}")
    if skip_dirs is None:
        logger.info("No directories to skip.")
        skip_dirs = []

    logger.info(f"Skipping directories", extra=dict(skip_dirs=skip_dirs))

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]  # Filter out skipped directories
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level

        logger.info(f'{indent}{os.path.basename(root)}/')
        for file in files:
            logger.info(f'{indent}{file}')

def get_os():
    """
    Returns the name of the operating system.

    Returns:
        str: The name of the operating system.
    """
    if os.name == 'nt':
        logger.info("Windows detected.")
        return "Windows"
    elif os.name == 'posix':
        logger.info("Linux or macOS detected.")
        return "Linux or macOS"
    else:
        logger.info("Unknown OS detected.")
        return "Unknown OS"
    
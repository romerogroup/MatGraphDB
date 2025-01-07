import inspect
import logging
import os
import sys
from typing import Callable, Dict, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.core import types
from parquetdb.core.parquetdb import NormalizeConfig

logger = logging.getLogger(__name__)

class GeneratorStore(ParquetDB):
    """
    A store for managing generator functions in a graph database.
    This class handles serialization, storage, and loading of functions
    that generate edges between nodes.
    """

    required_fields = ['generator_name', 'generator_func']
    metadata_keys = ['class', 'class_module']

    def __init__(self, storage_path: str, initial_fields: List[pa.Field]=None):
        """
        Initialize the EdgeGeneratorStore.

        Parameters
        ----------
        storage_path : str
            Path where the generator functions will be stored
            
        """
        if initial_fields is None:
            initial_fields = []
            
        initial_fields.extend([
            pa.field('generator_name', pa.string()),
            pa.field('generator_func', types.PythonObjectArrowType())
        ])
        super().__init__(db_path=storage_path,initial_fields=initial_fields)
        self._initialize_metadata()
        logger.debug(f"Initialized GeneratorStore at {storage_path}")

    def _initialize_metadata(self):
        """Initialize store metadata if not present."""
        metadata = self.get_metadata()
        update_metadata = False
        for key in self.metadata_keys:
            if key not in metadata:
                update_metadata = True
                break

        if update_metadata:
            self.set_metadata({
                'class': f"{self.__class__.__name__}",
                'class_module': f"{self.__class__.__module__}"
            })

    def store_generator(self, 
                       generator_func: Callable, 
                       generator_name: str,
                       generator_args: Dict=None,
                       generator_kwargs: Dict=None,
                       create_kwargs: Dict=None,
                       ) -> None:
        """
        Store an edge generator function.

        Parameters
        ----------
        generator_func : Callable
            The function that generates edges
        generator_name : str
            Name to identify the generator function
        generator_args : Dict
            Arguments to pass to the generator function
        generator_kwargs : Dict
            Keyword arguments to pass to the generator function
        create_kwargs : Dict
            Keyword arguments to pass to the create method
        """
        if create_kwargs is None:
            create_kwargs = {}
        try:
            df = self.read(columns=['generator_name']).to_pandas()
            
            if generator_name in df['generator_name'].values:
                raise ValueError(f"Generator '{generator_name}' already exists")
            
            # Serialize the function using dill
            
            # Create data record
            extra_fields = {}
            for key, value in generator_args.items():
                extra_fields[f'generator_args.{key}'] = value
                
            for key, value in generator_kwargs.items():
                extra_fields[f'generator_kwargs.{key}'] = value
            data = [{
                'generator_name': generator_name,
                'generator_func': generator_func,
                **extra_fields
            }]
            # Store the function data
            self.create(data=data, **create_kwargs)
            logger.info(f"Successfully stored generator '{generator_name}'")
            
        except Exception as e:
            logger.error(f"Failed to store generator '{generator_name}': {str(e)}")
            raise

    def load_generator_data(self, generator_name: str) -> pd.DataFrame:
        filters = [pc.field('generator_name')==generator_name]
        table = self.read(filters=filters)
        
        # table = table.drop_nulls()
        # for col in table.column_names:

        if len(table) == 0:
            raise ValueError(f"No generator found with name '{generator_name}'")
        return table.to_pandas()
    
    def is_in(self, generator_name: str) -> bool:
        filters = [pc.field('generator_name')==generator_name]
        table = self.read(filters=filters)
        return len(table) > 0
        
    def load_generator(self, generator_name: str) -> Callable:
        """
        Load an edge generator function by name.

        Parameters
        ----------
        generator_name : str
            Name of the generator function to load

        Returns
        -------
        Callable
            The loaded generator function
        """
        try:
            df = self.load_generator_data(generator_name)
            generator_func = df['generator_func'].iloc[0]
            return generator_func
            
        except Exception as e:
            logger.error(f"Failed to load generator '{generator_name}': {str(e)}")
            raise

    def list_generators(self) -> List[Dict]:
        """
        List all stored edge generators.

        Returns
        -------
        List[Dict]
            List of dictionaries containing generator information
        """
        try:
            result = self.read(columns=['generator_name'])
            return result.to_pylist()
        except Exception as e:
            logger.error(f"Failed to list generators: {str(e)}")
            raise

    def delete_generator(self, generator_name: str) -> None:
        """
        Delete a generator by name.

        Parameters
        ----------
        generator_name : str
            Name of the generator to delete
        """
        try:
            filters = [pc.field('generator_name')==generator_name]
            self.delete(filters=filters)
            logger.info(f"Successfully deleted generator '{generator_name}'")
        except Exception as e:
            logger.error(f"Failed to delete generator '{generator_name}': {str(e)}")
            raise
        
    def run_generator(self, generator_name: str, generator_args: Dict=None, generator_kwargs: Dict=None) -> None:
        """
        Run a generator function by name.
        """
        
        if generator_args is None:
            generator_args = {}
        if generator_kwargs is None:
            generator_kwargs = {}
        
        df=self.load_generator_data(generator_name)
        for column_name in df.columns:
            value = df[column_name].iloc[0]
            if 'generator_args' in column_name:
                arg_name = column_name.split('.')[-1]
                
                # Do not overwrite user-provided args
                if arg_name not in generator_args:
                    generator_args[arg_name] = value
            elif 'generator_kwargs' in column_name and value is not None:
                kwarg_name = column_name.split('.')[-1]
                
                # Do not overwrite user-provided kwargs
                if kwarg_name not in generator_kwargs:
                    generator_kwargs[kwarg_name] = value
                    
        generator_func = df['generator_func'].iloc[0]
        
        logger.debug(f"Generator func: {generator_func}")
        logger.debug(f"Generator args: {generator_args}")
        logger.debug(f"Generator kwargs: {generator_kwargs}")
        
        return generator_func(*generator_args.values(), **generator_kwargs)


import os
import sys
import logging
import inspect
import dill
import pyarrow as pa
from typing import Callable, Dict, List, Union
from parquetdb import ParquetDB
from parquetdb.core.parquetdb import NormalizeConfig

import pyarrow.compute as pc
import pyarrow as pa

from matgraphdb.stores.generator_store import GeneratorStore

logger = logging.getLogger(__name__)

class EdgeGeneratorStore(GeneratorStore):
    """
    A store for managing edge generator functions in a graph database.
    This class handles serialization, storage, and loading of functions
    that generate edges between nodes.
    """

    metadata_keys = ['class', 'class_module']

    

    def store_generator(self, 
                       generator_func: Callable, 
                       generator_name: str,
                       generator_args: Dict=None,
                       generator_kwargs: Dict=None,
                      ) -> None:
        """
        Store an edge generator function.

        Parameters
        ----------
        generator_func : Callable
            The function that generates edges
        generator_name : str
            Name to identify the generator function
        source_type : str
            Type of source nodes
        target_type : str
            Type of target nodes
        edge_type : str
            Type of edges to be generated
        """
        
        store_kwargs = dict(generator_func=generator_func, generator_name=generator_name, source_type=source_type, target_type=target_type, edge_type=edge_type)
        super().store_generator(**store_kwargs)

    def list_generators(self) -> List[Dict]:
        """
        List all stored edge generators.

        Returns
        -------
        List[Dict]
            List of dictionaries containing generator information
        """
        try:
            result = self.read()
            return result.to_pylist()
        except Exception as e:
            logger.error(f"Failed to list edge generators: {str(e)}")
            raise
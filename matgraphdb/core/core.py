import os
import logging
from typing import Callable, Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pymatgen.core import Structure

from matgraphdb.data.matdb import MatDB
from matgraphdb.data.calc_manager import CalculationManager
from matgraphdb.graph_kit.graph_manager import GraphManager
from matgraphdb.utils.mp_utils import multiprocess_task

logger = logging.getLogger(__name__)

class MatGraphDB:
    """
        A class for managing material repository operations, including database interactions, 
        running calculations, and managing the graph database.

        `MatGraphDB` serves as the main interface for handling material data and computations. 
        It contains instances of `MatDB`, `CalculationManager`, and `GraphManager`, 
        providing methods for interacting with the material database, performing calculations, 
        and managing graph-based data. The class also organizes directories for database files, 
        calculations, and graph data.

        Attributes:
        -----------
        matdb : MatDB
            Manages interactions with the material database.
        calc_manager : CalculationManager
            Handles calculation operations, including HPC-based tasks.
        graph_manager : GraphManager
            Manages the graph database for material data.
        """
    def __init__(self, db_path: str = 'MatGraphDB',
                 calculation_dirname='calculations',
                 graph_dirname='graph_database',
                 db_dirname='materials', 
                 n_cores=1,
                 **kwargs):
        """
        Initializes the `MatGraphDB` class by setting up directories, database, calculation manager, and graph manager.

        This constructor sets up the main directory structure and initializes the managers for 
        handling material database operations, running calculations, and managing graph-based data.

        Parameters:
        -----------
        db_path : str
            The path to the main directory where all data, including calculations and graphs, will be stored.
        calculation_dirname : str, optional
            Subdirectory name for storing calculation files (default is 'calculations').
        graph_dirname : str, optional
            Subdirectory name for the graph database (default is 'graph_database').
        db_dirname : str, optional
            Subdirectory name for the material database (default is 'materials').
        n_cores : int, optional
            The number of CPU cores to use for parallel processing (default is `N_CORES`).
        **kwargs
            Additional keyword arguments for configuring the `CalculationManager`.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Initialize MatGraphDB with a custom main directory
        .. highlight:: python
        .. code-block:: python

            matgraphdb = MatGraphDB(db_path='/path/to/db_path', n_cores=4)
        """

        logger.info("Initializing MaterialRepositoryHandler.")
        # Set up directories and database path
        self.n_cores = n_cores
        self.db_path = db_path
        self.calculation_dir=os.path.join(self.db_path, calculation_dirname)
        self.graph_dir=os.path.join(self.db_path, graph_dirname)
        self.matdb_path=os.path.join(self.db_path, db_dirname)

        os.makedirs(self.matdb_path, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

        self.matdb = MatDB(db_path=self.matdb_path)
        logger.debug("MatDB initialized.")

        self.calc_manager = CalculationManager(main_dir=self.calculation_dir, 
                                               matdb=self.matdb, 
                                               n_cores=self.n_cores,
                                               **kwargs)
        logger.debug("CalculationManager initialized.")

        self.graph_manager = GraphManager(graph_dir=self.graph_dir)
        logger.debug("GraphManager initialized.")



        logger.info(f"Main directory: {self.db_path}")
        logger.info(f"Material Database directory: {self.matdb_path}")
        logger.info(f"Material Calculation directory: {self.calculation_dir}")
        logger.info(f"Graph directory: {self.graph_dir}")
        logger.info(f"Cores: {self.n_cores}")


    
if __name__=='__main__':
    mgdb=MatGraphDB(db_path=os.path.join('data','MatGraphDB'))

    mgdb.get_materials()
        


    




import os
import logging
from typing import Callable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matgraphdb.data.material_manager import MaterialDatabaseManager
from matgraphdb.data.calc_manager import CalculationManager
from matgraphdb.graph_kit.graph_manager import GraphManager
from matgraphdb.utils import N_CORES, multiprocess_task

logger = logging.getLogger(__name__)

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
                 db_dirname='materials', 
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
        self.n_cores = n_cores
        self.main_dir = main_dir
        self.calculation_dir=os.path.join(self.main_dir, calculation_dirname)
        self.graph_dir=os.path.join(self.main_dir, graph_dirname)
        self.db_dir=os.path.join(self.main_dir, db_dirname)

        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

        self.db_manager = MaterialDatabaseManager(db_dir=self.db_dir)
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


        logger.info(f"Main directory: {self.main_dir}")
        logger.info(f"Material Database directory: {self.db_dir}")
        logger.info(f"Material Calculation directory: {self.calculation_dir}")
        logger.info(f"Graph directory: {self.graph_dir}")
        logger.info(f"Cores: {self.n_cores}")


    




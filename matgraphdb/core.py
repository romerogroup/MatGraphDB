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

from matgraphdb.data.material_manager import MaterialDatabaseManager
from matgraphdb.data.calc_manager import CalculationManager
from matgraphdb.graph_kit.graph_manager import GraphManager
from matgraphdb.utils import N_CORES, multiprocess_task

logger = logging.getLogger(__name__)

class MatGraphDB:
    """
        A class for managing material repository operations, including database interactions, 
        running calculations, and managing the graph database.

        `MatGraphDB` serves as the main interface for handling material data and computations. 
        It contains instances of `MaterialDatabaseManager`, `CalculationManager`, and `GraphManager`, 
        providing methods for interacting with the material database, performing calculations, 
        and managing graph-based data. The class also organizes directories for database files, 
        calculations, and graph data.

        Attributes:
        -----------
        db_manager : MaterialDatabaseManager
            Manages interactions with the material database.
        calc_manager : CalculationManager
            Handles calculation operations, including HPC-based tasks.
        graph_manager : GraphManager
            Manages the graph database for material data.
        """
    def __init__(self, main_dir: str,
                 calculation_dirname='calculations',
                 graph_dirname='graph_database',
                 db_dirname='materials', 
                 n_cores=N_CORES,
                 **kwargs):
        """
        Initializes the `MatGraphDB` class by setting up directories, database, calculation manager, and graph manager.

        This constructor sets up the main directory structure and initializes the managers for 
        handling material database operations, running calculations, and managing graph-based data.

        Parameters:
        -----------
        main_dir : str
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

            matgraphdb = MatGraphDB(main_dir='/path/to/main_dir', n_cores=4)
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

    def add_material(self, structure: Structure = None,
                coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                coords_are_cartesian: bool = False,
                species: List[str] = None,
                lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                properties: dict = None,
                include_symmetry: bool = True,
                calculate_funcs: List[Callable]=None,
                dataset_name:str='main',
                save_db: bool = True,
                **kwargs):
        """
        Adds a material to the database with optional symmetry and calculated properties.

        This method generates an entry for a material based on its structure, atomic coordinates, species, 
        and lattice parameters. It also allows for the calculation of additional properties and saves the 
        material to the database.

        Parameters:
        -----------
        structure : Structure, optional
            The atomic structure in Pymatgen Structure format.
        coords : Union[List[Tuple[float, float, float]], np.ndarray], optional
            Atomic coordinates of the material.
        coords_are_cartesian : bool, optional
            If True, indicates that the coordinates are in cartesian format.
        species : List[str], optional
            A list of atomic species present in the structure.
        lattice : Union[List[Tuple[float, float, float]], np.ndarray], optional
            Lattice parameters of the material.
        properties : dict, optional
            Additional properties to include in the material entry.
        include_symmetry : bool, optional
            If True, performs symmetry analysis and includes symmetry information in the entry.
        calculate_funcs : List[Callable], optional
            A list of functions for calculating additional properties of the material.
        dataset_name : str, optional
            The name of the dataset to which the material data will be added (default is 'main').
        save_db : bool, optional
            If True, saves the material entry to the database (default is True).
        **kwargs
            Additional keyword arguments passed to the ParquetDB `create` method.

        Returns:
        --------
        dict
            A dictionary containing the material's data, including calculated properties and additional information.

        Examples:
        ---------
        # Example usage:
        # Add a material to the database
        .. highlight:: python
        .. code-block:: python

            # Adding through coords, species, and lattice
            material_data = matgraphdb.add_material(
                coords=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
                species=["H", "He"],
                lattice=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            )

            # Adding cartesian coordinates
            material_data = matgraphdb.add_material(
                coords=[(0.0, 0.0, 0.0), (2.5, 2.5, 2.5)],
                species=["H", "He"],
                lattice=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                coords_are_cartesian=True,
            )

            # Adding through a structure
            structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            material_data = matgraphdb.add_material(structure=structure)
        """
        all_args=locals()
        all_args.pop('self')
        return self.db_manager.add(**all_args)
    
    def add_materials(self, materials: List[Dict], dataset_name:str='main', **kwargs):
        """
        Adds multiple materials to the database in a single transaction.

        This method processes a list of materials and writes their data to the specified 
        database dataset in a single transaction. Each material should be represented as a 
        dictionary with keys corresponding to the arguments for the `add` method.

        Parameters:
        -----------
        materials : List[Dict]
            A list of dictionaries where each dictionary contains the material data and 
            corresponds to the arguments for the `add` method.
        dataset_name : str, optional
            The name of the dataset to add the data to (default is 'main').
        **kwargs
            Additional keyword arguments passed to the ParquetDB `create` method.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Add a batch of materials to the database
        .. highlight:: python
        .. code-block:: python

            materials = [
                {'coords': [(0.0, 0.0, 0.0)], 'species': ["H"], 'lattice': [(1.0, 0.0, 0.0)]},
                {'coords': [(0.5, 0.5, 0.5)], 'species': ["He"], 'lattice': [(0.0, 1.0, 0.0)]}
            ]
            matgraphdb.add_materials(materials)
        """
        all_args=locals()
        all_args.pop('self')
        return self.db_manager.add_many(materials, **all_args)
   
    def get_materials(self, ids=None, 
            columns:List[str]=None, 
            include_cols:bool=True, 
            filters: List[pc.Expression]=None,
            output_format='dataset',
            dataset_name:str='main', 
            batch_size=None):
        """
        Reads materials from the database by ID or based on specific filters.

        This method retrieves material data from the database, allowing filtering by material IDs, 
        selecting specific columns, and applying additional filters. The output format can be specified 
        as either a dataset or another supported format.

        Parameters:
        -----------
        ids : list, optional
            A list of material IDs to retrieve from the database. If None, all materials will be retrieved.
        columns : List[str], optional
            A list of column names to include in the output. If None, all columns are included.
        include_cols : bool, optional
            If True, only the specified columns are included. If False, all columns except the specified 
            ones are included (default is True).
        filters : List[pc.Expression], optional
            A list of filters to apply to the query, allowing for more fine-grained material selection.
        output_format : str, optional
            The format in which to return the data, such as 'dataset' or another supported format (default is 'dataset').
        dataset_name : str, optional
            The name of the dataset to read the data from (default is 'main').
        batch_size : int, optional
            The size of data batches to be read, useful for handling large datasets (default is None).

        Returns:
        --------
        Depends on `output_format`
            The material data in the specified format (e.g., a dataset or another format supported by the database).

        Examples:
        ---------
        # Example usage:
        # Read specific materials by their IDs and select certain columns
        .. highlight:: python
        .. code-block:: python

            # read materials by IDs, select certain columns, and choose a dataset output format
            materials_dataset =  matgraphdb.get_materials(
                ids=[1, 2, 3],
                columns=['formula', 'density'],
                output_format='table'
            )

            # Convert to pandas DataFrame
            df=materials_dataset.to_pandas()

            # Apply filters to the query
            materials_dataset = matgraphdb.get_materials(
                filters=[pc.field('formula') == 'Fe', pc.field('density') > 7.0],
                output_format='table'
            )

            # Read materials in batches
            materials_generator =  matgraphdb.get_materials(
                batch_size=100,
                output_format='batch_generator'
            )
            for batch_table in materials_generator:
                # Do something with the batch dataset


            # Read materials as an unloaded dataset
            dataset =  matgraphdb.get_materials(
                output_format='dataset'
            )
        """
        all_args=locals()
        all_args.pop('self')
        return self.db_manager.read(**all_args)
    
    def update_materials(self, data: Union[List[dict], dict, pd.DataFrame], 
               dataset_name:str='main', 
               field_type_dict:dict=None):
        """
        Updates material records in the database.

        This method updates existing material entries in the specified dataset using the provided data.
        The data can be a dictionary, a list of dictionaries, or a pandas DataFrame, where each record 
        should include an identifier (e.g., 'id') for the material to update.

        Parameters:
        -----------
        data : Union[List[dict], dict, pd.DataFrame]
            The data to update in the database. Each dictionary or row should contain the 'id' of the record to update.
        dataset_name : str, optional
            The name of the dataset where the materials are stored (default is 'main').
        field_type_dict : dict, optional
            A dictionary mapping field names to new data types, used to modify the schema if needed.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Update material data in the database
        .. highlight:: python
        .. code-block:: python

            update_data = [{'id': 1, 'density': 5.3}, {'id': 2, 'volume': 22.1}]
            matgraphdb.update_materials(data=update_data, dataset_name='main')

            # Update records with new field types
            field_type_dict = {'density': pa.float32(), 'volume': pa.float16()}
            matgraphdb.update_materials(update_data, field_type_dict=field_type_dict)
        """

        all_args=locals()
        all_args.pop('self')
        return self.db_manager.update(**all_args)
    
    def delete_materials(self, ids:List[int], dataset_name:str='main'):
        """
        Deletes material records from the database by their IDs.

        This method removes material entries from the specified dataset based on the provided list of IDs.

        Parameters:
        -----------
        ids : List[int]
            A list of material IDs to delete from the dataset.
        dataset_name : str, optional
            The name of the dataset from which to delete the materials (default is 'main').

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Delete materials by their IDs
        .. highlight:: python
        .. code-block:: python

            matgraphdb.delete_materials(ids=[1, 2, 3], dataset_name='main')
        """

        all_args=locals()
        all_args.pop('self')
        return self.db_manager.delete(**all_args)

    def run_inmemory_calculation(self, calc_func:Callable, save_results:bool=False, 
                                 verbose:bool=False, read_args:dict=None, **kwargs):
        """
        Runs a calculation on the data retrieved from the `MaterialDatabaseManager`.

        This method processes data from the material database using a user-defined calculation function.
        The `calc_func` is expected to accept a dictionary-like object (each row's data) and return a 
        dictionary representing the results of the calculation. Optionally, the results can be saved 
        back to the database. They will save as the key in the return dictionary

        Parameters:
        -----------
        calc_func : Callable
            A function that processes each row of data. This function should accept a dictionary-like 
            object representing the row's data and return a dictionary containing the calculated results 
            for that row.
        save_results : bool, optional
            A flag indicating whether to save the results back to the database after processing. 
            Defaults to False.
        verbose : bool, optional
            A flag indicating whether to print error messages. Defaults to False.
        read_args : dict, optional
            Additional arguments to pass to the `MaterialDatabaseManager.read` method.
        **kwargs
            Additional keyword arguments to pass to the calculation function.

        Returns:
        --------
        list
            A list of result dictionaries returned by the `calc_func` for each row in the database.

        Examples:
        ---------
        # Example usage:
        # Define a calculation function and apply it to in-memory material data
        .. highlight:: python
        .. code-block:: python

            # Define a custom calculation function
            def my_calc_func(row_data, **kwargs):
                # Perform some calculations on row_data
                return {'result': sum(row_data.values())}

            # Run the in-memory calculation function on all material data
            results = matgraphdb.run_inmemory_calculation(my_calc_func, save_results=True, verbose=True)
        """
        all_args=locals()
        all_args.pop('self')
        return self.calc_manager.run_inmemory_calculation(**all_args)
    
    def create_disk_calculation(self, calc_func: Callable, calc_name: str = None, read_args: dict = None, **kwargs):
        """
        Creates a new calculation by applying the provided function to each row in the database.
        The `calc_func` expects a dictionary-like object (each row's data) and a directory path where 
        the calculation results should be stored. The function is responsible for saving the results in 
        this directory, which is specific to each row and named based on the row's unique ID and the 
        calculation name.

        Parameters:
        -----------
        calc_func : Callable
            The function to apply to each material.
            The first argument should be a dictionary-like object (each row's data) 
            and the second argument should be the calculation directory path.
        calc_name : str, optional
            The name of the calculation. If not provided, it defaults to the name of the function.
        read_args : dict, optional
            Additional arguments to pass to the `MaterialDatabaseManager.read` method.
        **kwargs
            Additional arguments to pass to the calculation function.

        Returns:
        --------
        List
            The results of the calculation for each material.

        Examples:
        ---------
        # Example usage:
        # Define a calculation function and apply it to all materials
        .. highlight:: python
        .. code-block:: python


            # Define a custom calculation function
            def my_calc_func(row_data: dict, calc_dir: str, **kwargs):
                # Perform some calculation
                return None

            # Apply the calculation function to all materials
            results = matgraphdb.create_disk_calculation(my_calc_func, 'my_calc')
        """

        all_args=locals()
        all_args.pop('self')
        return self.calc_manager.create_disk_calculation_disk_calculation(**all_args)
    
    def run_func_on_disk_calculations(self, calc_func:Callable, calc_name: str, ids:List[int]=None, **kwargs):
        """
        Runs a specified function on all calculation directories or a subset of directories.

        Parameters:
        -----------
        calc_func : Callable
            The function to run on each calculation directory. 
            This functions first argument should be the calculation directory path.
        calc_name : str
            The name of the calculation, used to locate directories for each material where the calculation is stored.
        ids : list[int], optional
            A list of material IDs. If not provided, the function will run on all material directories.
        **kwargs
            Additional keyword arguments to pass to the `calc_func` and the multiprocessing task.

        Returns:
        --------
        List
            The results of running the function on each calculation directory.

        Examples:
        ---------
        # Example usage:
        # Define a function that processes the calculation directory
        .. highlight:: python
        .. code-block:: python

            def process_directory(calc_dir, **kwargs):
                # Custom logic for processing
                return None

            # Run function on specific material IDs
            results = calc_manager.run_func_on_disk_calculations(process_directory, "calculation_name", ids=[0, 1])

            # Run function on all material directories
            results = calc_manager.run_func_on_disk_calculations(process_directory, "calculation_name")
        """
        all_args=locals()
        all_args.pop('self')
        return self.calc_manager.run_func_on_disk_calculations(**all_args)

    def submit_disk_jobs(self, calc_name:str, ids:List=None, **kwargs):
        """
        Submits SLURM jobs for all materials or a subset of materials by calculation name.

        Parameters:
        -----------
        calc_name : str
            The name of the calculation for which jobs will be submitted.
        ids : list[int], optional
            A list of material IDs to submit jobs for. If not provided, jobs are submitted for all materials.
        **kwargs
            Additional keyword arguments to pass to the job submission function.

        Returns:
        --------
        List
            The results of job submission for each material.

        Examples:
        ---------
        # Example usage:
        # Submit SLURM jobs for all materials
        .. highlight:: python
        .. code-block:: python

            # Submit jobs for all materials for a specific calculation
            results = matgraphdb.submit_disk_jobs("calculation_name")

            # Submit jobs for specific material IDs
            results = matgraphdb.submit_disk_jobs("calculation_name", ids=[0, 1])
        """
        all_args=locals()
        all_args.pop('self')
        return self.calc_manager.submit_disk_jobs(**all_args)
    
    def add_field_from_disk_calculation(self, func:Callable, calc_name: str, ids:List[int]=None, update_args: dict = None, **kwargs):
        """
        Adds calculation data from disk to the database by processing each material's calculation directory and
        updating the database with the results.

        Parameters:
        -----------
        func : Callable
            The function that performs the calculation or processing task on each calculation directory.
            The first argument should be the calculation directory path.
            This function must return a dictionary with field names as keys and values as values.
        calc_name : str
            The name of the calculation, used to locate directories for each material where the calculation is stored.
        ids : list[int], optional
            A list of material IDs to process. If not provided, all material directories in `self.material_dirs` are processed.
        update_args : dict, optional
            Additional arguments to pass to the database update operation.
        **kwargs
            Additional keyword arguments to pass to the `func` and the multiprocessing task.

        Returns:
        --------
        None
            This method does not return a value. It updates the database with the results of the calculations.

        Examples:
        ---------
        # Example usage:
        # Define a function that processes the calculation directory and returns a dictionary of results
        .. highlight:: python
        .. code-block:: python

            def process_directory(calc_dir, **kwargs):
                # Custom logic for processing
                return {"field_name": "value"}

            # Add calculation data to the database
            matgraphdb.add_field_from_disk_calculation(process_directory, "calculation_name", ids=[0, 1], 
                            update_args={"table_name": "main", field_type_dict={"field_name": float}})
        """
        all_args=locals()
        all_args.pop('self')
        return self.calc_manager.add_field_from_disk_calculation(**all_args)
    
if __name__=='__main__':
    mgdb=MatGraphDB(main_dir=os.path.join('data','MatGraphDB'))

    mgdb.get_materials()
        


    




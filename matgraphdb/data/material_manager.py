import json
import os
import logging
from typing import Callable, Union, List, Tuple, Dict
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Composition
from parquetdb import ParquetDB
import spglib

from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections
from matgraphdb.utils import multiprocess_task

logger = logging.getLogger(__name__)


class MaterialDatabaseManager:
    """
    This class is intended to be the Data Access Layer for the Material Database.
    It provides methods for adding, reading, updating, and deleting materials from the database.
    """

    def __init__(self, db_dir: str, n_cores=8):
        """
        Initializes the `MaterialDatabaseManager` by setting the database directory and number of cores.

        This constructor sets up the `MaterialDatabaseManager` by specifying the directory where the database 
        is located and configuring the number of cores to be used. It also initializes the connection to the 
        database and defines the main dataset name.

        Parameters:
        -----------
        db_dir : str
            The directory where the database is stored.
        n_cores : int, optional
            The number of CPU cores to use for operations (default is 8).

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Initialize the MaterialDatabaseManager with a database directory and the default number of cores
        .. highlight:: python
        .. code-block:: python

            from matgraphdb.data.material_manager import MaterialDatabaseManager
            manager = MaterialDatabaseManager(db_dir='/path/to/db')
        """

        self.db_dir=db_dir
        self.n_cores=n_cores
        self.main_dataset_name='main'

        logger.info(f"Initializing MaterialDatabaseManager with database at {db_dir}")
        self.db = ParquetDB(db_dir)

    def add(self, structure: Structure = None,
                coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                coords_are_cartesian: bool = False,
                species: List[str] = None,
                lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                properties: dict = None,
                include_symmetry: bool = True,
                calculate_funcs: List[Callable]=None,
                dataset_name:str='main',
                save_db: bool = True,
                **kwargs
        ):
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
            material_data = manager.add(
                coords=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
                species=["H", "He"],
                lattice=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            )

            # Adding cartesian coordinates
            material_data = manager.add(
                coords=[(0.0, 0.0, 0.0), (2.5, 2.5, 2.5)],
                species=["H", "He"],
                lattice=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                coords_are_cartesian=True,
            )

            # Adding through a structure
            structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            material_data = manager.add(structure=structure)
        """

        # Generating entry data
        entry_data={}

        if properties is None:
            properties={}
        if calculate_funcs is None:
            calculate_funcs=[]
        if include_symmetry:
            calculate_funcs.append(partial(perform_symmetry_analysis, symprec=kwargs.get('symprec',0.1)))
            
        logger.info("Adding a new material.")
        try:

            structure = self._init_structure(structure, coords, coords_are_cartesian, species, lattice)

            if structure is None:
                logger.error("A structure must be provided.")
                raise ValueError("Either a structure must be provided")
            
      
            composition=structure.composition
    
            entry_data['formula']=composition.formula
            entry_data['elements']=list([element.symbol for element in composition.elements])
            
            entry_data['lattice']=structure.lattice.matrix.tolist()
            entry_data['frac_coords']=structure.frac_coords.tolist()
            entry_data['cartesian_coords']=structure.cart_coords.tolist()
            entry_data['atomic_numbers']=structure.atomic_numbers
            entry_data['species']=list([specie.symbol for specie in structure.species])

            entry_data["volume"]=structure.volume
            entry_data["density"]=structure.density
            entry_data["nsites"]=len(structure.sites)
            entry_data["density_atomic"]=entry_data["nsites"]/entry_data["volume"]

            # Calculating additional properties
            if calculate_funcs:
                for func in calculate_funcs:
                    try:
                        func_results=partial(func, **kwargs)(structure)
                        entry_data.update(func_results)
                    except Exception as e:
                        logger.error(f"Error calculating property: {e}")

            # Adding other properties as columns
            entry_data.update(properties)
            
            if save_db:
                self.db.create(entry_data, dataset_name=dataset_name, **kwargs)
                logger.info("Material added successfully.")
            return entry_data
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        
        return entry_data
    
    def add_many(self, materials: List[Dict], dataset_name:str='main', **kwargs):
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
            manager.add_many(materials)
        """

        logger.info(f"Adding {len(materials)} materials to the database.")
        results=multiprocess_task(self._add_many, materials, n_cores=self.n_cores)
        entry_data=[result for result in results if result]
        try:
            self.db.create(entry_data, dataset_name=dataset_name, **kwargs)
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        logger.info("All materials added successfully.")

    def read(self, ids=None, 
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
            materials_dataset = manager.read(
                ids=[1, 2, 3],
                columns=['formula', 'density'],
                output_format='table'
            )

            # Convert to pandas DataFrame
            df=materials_dataset.to_pandas()

            # Apply filters to the query
            materials_dataset = manager.read(
                filters=[pc.field('formula') == 'Fe', pc.field('density') > 7.0],
                output_format='table'
            )

            # Read materials in batches
            materials_generator = manager.read(
                batch_size=100,
                output_format='batch_generator'
            )
            for batch_table in materials_generator:
                # Do something with the batch dataset


            # Read materials as an unloaded dataset
            dataset = manager.read(
                output_format='dataset'
            )
        """

        logger.debug(f"Reading materials.")
        logger.debug(f"ids: {ids}")
        logger.debug(f"dataset_name: {dataset_name}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"include_cols: {include_cols}")
        logger.debug(f"filters: {filters}")
        logger.debug(f"output_format: {output_format}")
        logger.debug(f"batch_size: {batch_size}")

        kwargs=dict(ids=ids, dataset_name=dataset_name, columns=columns, include_cols=include_cols, 
                    filters=filters, output_format=output_format, batch_size=batch_size)
        return self.db.read(**kwargs)
    
    def update(self, data: Union[List[dict], dict, pd.DataFrame], 
               dataset_name='main', 
               field_type_dict=None):
        """
        Updates existing records in the database.

        This method updates records in the specified dataset based on the provided data. Each entry in the data 
        must include an 'id' key that corresponds to the record to be updated. Field types can also be updated 
        if specified in `field_type_dict`.

        Parameters:
        -----------
        data : Union[List[dict], dict, pd.DataFrame]
            The data to update in the database. It can be a dictionary, a list of dictionaries, or a pandas DataFrame. 
            Each dictionary should have an 'id' key for identifying the record to update.
        dataset_name : str, optional
            The name of the dataset where the data will be updated (default is 'main').
        field_type_dict : dict, optional
            A dictionary specifying new field types. The keys are field names, and the values are the desired types.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Update a record in the database
        .. highlight:: python
        .. code-block:: python

            # Update a record in the database
            update_data = {'id': 1, 'density': 5.3, 'volume': 22.1}
            manager.update(update_data)

            # Update multiple records in the database
            update_data = [
                {'id': 1, 'density': 5.3, 'volume': 22.1},
                {'id': 2, 'density': 7.8, 'volume': 33.2}
            ]
            manager.update(update_data)

            # Update records with new field types
            field_type_dict = {'density': pa.float32(), 'volume': pa.float16()}
            manager.update(update_data, field_type_dict=field_type_dict)

        """

        logger.info(f"Updating data")
        self.db.update(data, dataset_name=dataset_name, field_type_dict=field_type_dict)
        logger.info("Data updated successfully.")

    def delete(self, ids:List[int], dataset_name:str='main'):
        """
        Deletes records from the database by ID.

        This method deletes specific records from the database based on the provided list of IDs.

        Parameters:
        -----------
        ids : List[int]
            A list of record IDs to delete from the database.
        dataset_name : str, optional
            The name of the dataset from which to delete the records (default is 'main').

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Delete records by ID
        .. highlight:: python
        .. code-block:: python

            manager.delete(ids=[1, 2, 3])
        """

        logger.info(f"Deleting data {ids}")
        self.db.delete(ids, dataset_name=dataset_name)
        logger.info("Data deleted successfully.")

    def get_schema(self, dataset_name:str ='main'):
        """
        Retrieves the schema of a specified dataset.

        This method returns the schema of the specified dataset in the database, which includes the 
        field names and their data types.

        Parameters:
        -----------
        dataset_name : str, optional
            The name of the dataset for which the schema is to be retrieved (default is 'main').

        Returns:
        --------
        pyarrow.Schema
            The schema of the specified dataset, including field names and types.

        Examples:
        ---------
        # Example usage:
        # Get the schema of the 'main' dataset
        .. highlight:: python
        .. code-block:: python

            schema = manager.get_schema(dataset_name='main')
        """

        return self.db.get_schema(dataset_name=dataset_name)
    
    def update_schema(self, dataset_name:str='main', field_dict=None, schema=None):
        """
        Updates the schema of a specified dataset.

        This method allows updating the schema of a dataset in the database. You can either provide a dictionary 
        specifying new field types or directly pass a new schema.

        Parameters:
        -----------
        dataset_name : str, optional
            The name of the dataset whose schema will be updated (default is 'main').
        field_dict : dict, optional
            A dictionary where the keys are field names and the values are the new field types.
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Update the schema of the 'main' dataset by adding a new field
        .. highlight:: python
        .. code-block:: python

            # Update fields
            field_dict = {'density': pa.float64()}
            manager.update_schema(field_dict=field_dict)

            # Update by passing a new schema. 
            new_schema = pa.schema([
                ('id', pa.int64()),
                ('formula', pa.string()),
                ('density', pa.float64()),
                ('new_field', pa.float64())
            ])
            manager.update_schema(schema=new_schema)

            # The field names must match the existing ones and be in the same order.
            # Also the schema field type must be compatible with the existing entries.
            # It would be useful to get the schema of a dataset before updating it.

            current_schema = manager.get_schema(dataset_name='main')
            density_field = current_schema.get_field_index('density')
            new_schema = current_schema.set(density_field, pa.float32())
            manager.update_schema(schema=new_schema)
        """

        self.db.update_schema(dataset_name=dataset_name, field_dict=field_dict, schema=schema)

    def get_datasets(self):
        """
        Retrieves a list of all datasets in the database.

        This method returns the names of all datasets currently stored in the database.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[str]
            A list of dataset names present in the database.

        Examples:
        ---------
        # Example usage:
        # Get all datasets in the database
        .. highlight:: python
        .. code-block:: python

            datasets = manager.get_datasets()
        """

        return self.db.get_datasets()
    
    def get_metadata(self, dataset_name:str='main'):
        """
        Retrieves the metadata of a specified dataset.

        This method returns the metadata associated with the specified dataset in the database.

        Parameters:
        -----------
        dataset_name : str, optional
            The name of the dataset for which metadata is retrieved (default is 'main').

        Returns:
        --------
        dict
            A dictionary containing the metadata of the specified dataset.

        Examples:
        ---------
        # Example usage:
        # Get the metadata of the 'main' dataset
        .. highlight:: python
        .. code-block:: python

            metadata = manager.get_metadata(dataset_name='main')
        """
        return self.db.get_metadata(dataset_name=dataset_name)
    
    def set_metadata(self, metadata: dict, dataset_name:str):
        """
        Sets the metadata for a specified dataset.

        This method updates the metadata of the specified dataset with the provided dictionary.

        Parameters:
        -----------
        metadata : dict
            A dictionary containing the metadata to be set.
        dataset_name : str
            The name of the dataset whose metadata will be updated.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Set new metadata for a dataset
        .. highlight:: python
        .. code-block:: python

            metadata = {'description': 'Material properties', 'version': 2}
            manager.set_metadata(metadata=metadata, dataset_name='main')
        """

        self.db.set_metadata(metadata=metadata, dataset_name=dataset_name)

    def drop_dataset(self, dataset_name:str='main'):
        """
        Drops a specified dataset from the database.

        This method permanently removes the specified dataset and its contents from the database.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset to be dropped.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Drop the 'main' dataset from the database
        .. highlight:: python
        .. code-block:: python

            manager.drop_dataset(dataset_name='main')
        """

        self.db.drop_dataset(dataset_name=dataset_name)

    def rename_dataset(self, old_dataset_name:str, new_dataset_name:str):
        """
        Renames a dataset in the database.

        This method changes the name of an existing dataset to a new specified name.

        Parameters:
        -----------
        old_dataset_name : str
            The current name of the dataset to be renamed.
        new_dataset_name : str
            The new name for the dataset.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Rename a dataset from 'old_dataset' to 'new_dataset'
        .. highlight:: python
        .. code-block:: python

            manager.rename_dataset(old_dataset_name='old_dataset', new_dataset_name='new_dataset')
        """

        self.db.rename_dataset(old_dataset_name=old_dataset_name, new_dataset_name=new_dataset_name)

    def copy_dataset(self, old_dataset_name:str, new_dataset_name:str, **kwargs):
        """
        Copies a dataset in the database to a new dataset.

        This method creates a copy of an existing dataset under a new name. Additional parameters 
        can be passed to customize the copying process.

        Parameters:
        -----------
        old_dataset_name : str
            The name of the dataset to be copied.
        new_dataset_name : str
            The name for the new copy of the dataset.
        **kwargs
            Additional keyword arguments for customizing the copy process.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Copy a dataset from 'main' to 'main_backup'
        .. highlight:: python
        .. code-block:: python

            manager.copy_dataset(old_dataset_name='main', new_dataset_name='main_backup')
        """

        self.db.copy_dataset(old_dataset_name=old_dataset_name, new_dataset_name=new_dataset_name, **kwargs)

    def optimize_dataset(self, dataset_name:str='main', 
                    max_rows_per_file=10000,
                    min_rows_per_group=0,
                    max_rows_per_group=10000,
                    batch_size=None,
                    **kwargs):
        """
        Optimizes a specified dataset in the database.

        This method optimizes the storage and performance of a dataset by controlling file and group sizes, 
        potentially improving query performance and reducing disk usage.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset to be optimized.
        max_rows_per_file : int
            The maximum number of rows allowed per file.
        min_rows_per_group : int
            The minimum number of rows per group.
        max_rows_per_group : int
            The maximum number of rows per group.
        batch_size : int
            The batch size to be used during optimization.
        **kwargs
            Additional keyword arguments passed to the `pq.write_to_dataset` function.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Optimize the dataset with specific row and group limits
        .. highlight:: python
        .. code-block:: python

            # The smaller parquet files will now only contain 100 rows per file
            manager.optimize_dataset(
                dataset_name='main',
                max_rows_per_file=100,
                min_rows_per_group=0,
                max_rows_per_group=100,
            )
        """

        self.db.optimize_dataset(dataset_name=dataset_name, **kwargs)

    def export_dataset(self, dataset_name:str='main', export_dir:str=None, export_format:str='parquet', **kwargs):
        """Export a dataset in the database.

        Args:
            dataset_name (str): The name of the dataset to be exported.
            export_dir (str, optional): The directory where the exported dataset will be saved.
            export_format (str, optional): The format of the exported dataset. Defaults to 'parquet'.
            **kwargs: Additional keyword arguments to pass to the ParquetDB export_dataset method.

        Returns:
            None
        """        
        self.db.export_dataset(dataset_name=dataset_name, file_path=export_dir, format=export_format, **kwargs)

    def export_partitioned_dataset(self, dataset_name: str, 
                                   export_dir: str, 
                                   partitioning,
                                   partitioning_flavor=None,
                                   batch_size: int = None, 
                                   **kwargs):
        """
        Exports a partitioned dataset to a specified directory.

        This method exports a dataset to the specified directory using the provided partitioning scheme. 
        Additional parameters allow customization of the export process, such as setting the partitioning flavor 
        and controlling the batch size.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset to export.
        export_dir : str
            The directory where the dataset will be exported.
        partitioning : dict
            The partitioning scheme to use for the dataset.
        partitioning_flavor : str, optional
            The partitioning flavor to use (e.g., 'hive' or another supported flavor).
        batch_size : int, optional
            The batch size to use during export.
        **kwargs
            Additional keyword arguments passed to the `pq.write_to_dataset` function.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Export a partitioned dataset to a directory
        .. highlight:: python
        .. code-block:: python

            manager.export_partitioned_dataset(
                dataset_name='main',
                export_dir='/path/to/export',
                partitioning={'nelements': 'int', 'crystal_system': 'str'},
                partitioning_flavor='hive',
                batch_size=1000
            )
        """

        self.db.export_partitioned_dataset(dataset_name=dataset_name, 
                                           file_path=export_dir, 
                                           partitioning=partitioning, 
                                           partitioning_flavor=partitioning_flavor, 
                                           batch_size=batch_size, 
                                           **kwargs)
    
    def _init_structure(self, structure, coords, coords_are_cartesian, species, lattice):
        """
        Initializes a structure object from provided data.

        This method checks whether a structure object is provided directly or if it needs to be built 
        from coordinates, species, and lattice parameters. It returns the structure or raises an error 
        if invalid input is provided.

        Parameters:
        -----------
        structure : Structure, optional
            An existing `Structure` object to use. If not provided, the structure is built from other parameters.
        coords : list or np.ndarray, optional
            Atomic coordinates for the structure.
        coords_are_cartesian : bool, optional
            If True, the coordinates are in Cartesian format. If False, they are fractional.
        species : list, optional
            A list of atomic species.
        lattice : list or np.ndarray, optional
            Lattice parameters for the structure.

        Returns:
        --------
        Structure or None
            A `Structure` object if valid inputs are provided, or None if inputs are incomplete.

        Examples:
        ---------
        # Example usage:
        # Initialize a structure from coordinates, species, and lattice
        .. highlight:: python
        .. code-block:: python

            structure = manager._init_structure(
                coords=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
                species=["H", "He"],
                lattice=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            )
        """

        check_all_params_provided(coords=coords, species=species, lattice=lattice)
        logger.debug("Processing structure input.")
        if structure is not None:
            if not isinstance(structure, Structure):
                logger.error("Structure must be an Structure object.")
                raise TypeError("Structure must be an Structure object")
            logger.debug("Using provided Structure structure.")
            return structure
        elif coords is not None and species is not None and lattice is not None:
            logger.debug("Building Structure structure from provided coordinates, species, and lattice.")
            if coords_are_cartesian:
                return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=True)
            else:
                return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=False)
        else:
            logger.debug("No valid structure information provided.")
            return None

    def _init_composition(self, composition):
        """
        Initializes a composition object from provided data.

        This method processes the input to initialize a `Composition` object, which can be provided as an 
        ASE Atoms object, a string, or a dictionary. The method handles different formats and returns a 
        valid composition or None if the input is invalid.

        Parameters:
        -----------
        composition : Union[Composition, str, dict], optional
            The composition of the material, which can be provided as a string, dictionary, or `Composition` object.

        Returns:
        --------
        Composition or None
            A `Composition` object if the input is valid, or None if the input is incomplete or invalid.

        Examples:
        ---------
        # Example usage:
        # Initialize composition from a string
        .. highlight:: python
        .. code-block:: python

            composition = manager._init_composition("H2O")
        """

        logger.debug("Processing composition input.")
        if isinstance(composition, Composition):
            logger.debug("Composition provided as ASE Atoms object.")
            return composition
        elif isinstance(composition, str):
            composition_str = composition
            logger.debug(f"Composition provided as string: {composition_str}")
            return Composition(composition_str)
        elif isinstance(composition, dict):
            composition_str = ', '.join(f"{k}:{v}" for k, v in composition.items())
            logger.debug(f"Composition provided as dict: {composition_str}")
            return Composition(composition_str)
        else:
            logger.debug("No valid composition information provided.")
            return None
        
    def _add_many(self, material):
        """
        Adds a material entry to the database without saving it immediately.

        This method prepares the material data by disabling automatic database saving and then calls 
        the `add` method to process the material. It is typically used in batch processing scenarios.

        Parameters:
        -----------
        material : dict
            A dictionary containing the material data, passed as arguments to the `add` method.

        Returns:
        --------
        dict
            The processed material data returned by the `add` method.

        Examples:
        ---------
        # Example usage:
        # Add a material entry in batch without saving to the database
        .. highlight:: python
        .. code-block:: python

            material_data = {
                'coords': [(0.0, 0.0, 0.0)],
                'species': ['H'],
                'lattice': [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            }
            processed_material = manager._add_many(material_data)
        """

        material['save_db']=False
        return self.add(**material)


def check_all_params_provided(**kwargs):
    """
    Ensures that all or none of the provided parameters are given.

    This utility function checks whether either all or none of the provided parameters 
    are set. If only some parameters are provided, it raises a `ValueError`, indicating 
    which parameters are missing and which are provided.

    Parameters:
    -----------
    **kwargs : dict
        A dictionary of parameter names and their corresponding values to be checked.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If only some of the parameters are provided and not all.

    Examples:
    ---------
    # Example usage:
    # Check that all required parameters are provided
    .. highlight:: python
    .. code-block:: python

        check_all_params_provided(coords=[(0, 0, 0)], species=None, lattice=None)
    """

    param_names = list(kwargs.keys())
    param_values = list(kwargs.values())

    all_provided = all(value is not None for value in param_values)
    none_provided = all(value is None for value in param_values)
    
    if not (all_provided or none_provided):
        missing = [name for name, value in kwargs.items() if value is None]
        provided = [name for name, value in kwargs.items() if value is not None]
        logger.error(
            f"If any of {', '.join(param_names)} are provided, all must be provided. "
            f"Missing: {', '.join(missing)}. Provided: {', '.join(provided)}."
        )
        raise ValueError(
            f"If any of {', '.join(param_names)} are provided, all must be provided. "
            f"Missing: {', '.join(missing)}. Provided: {', '.join(provided)}."
        )

def convert_coordinates(coords, lattice, coords_are_cartesian=True):
    """
    Converts between Cartesian and fractional coordinates based on lattice vectors.

    This method takes a set of coordinates and converts them either from Cartesian to fractional 
    or from fractional to Cartesian, depending on the provided flag.

    Parameters:
    -----------
    coords : numpy.ndarray
        A 1D or 2D array of coordinates to be converted.
    lattice : numpy.ndarray
        A 3x3 matrix representing the lattice vectors.
    coords_are_cartesian : bool, optional
        A flag indicating whether the input coordinates are in Cartesian format (True) 
        or fractional format (False). Defaults to True.

    Returns:
    --------
    tuple
        A tuple containing fractional coordinates and Cartesian coordinates.

    Examples:
    ---------
    # Example usage:
    # Convert Cartesian coordinates to fractional coordinates
    .. highlight:: python
    .. code-block:: python

        frac_coords, cart_coords = manager.convert_coordinates(
            coords=np.array([[1.0, 2.0, 3.0]]),
            lattice=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
    """

    # Ensure the lattice is a numpy array
    lattice = np.array(lattice)
    
    if coords_are_cartesian:
        # If coordinates are Cartesian, calculate fractional coordinates
        frac_coords = np.linalg.solve(lattice.T, coords.T).T  # cartesian to fractional
        cart_coords = coords
    else:
        # If coordinates are fractional, calculate Cartesian coordinates
        frac_coords = coords
        cart_coords = np.dot(coords, lattice)  # fractional to cartesian

    return frac_coords, cart_coords

def perform_symmetry_analysis(structure: Structure, symprec: float = 0.1):
    """
    Performs symmetry analysis on a structure.

    This method analyzes the symmetry of the provided structure and returns detailed symmetry 
    information, including the crystal system, space group number, point group, and Wyckoff positions.

    Parameters:
    -----------
    structure : Structure
        The atomic structure to be analyzed.
    symprec : float, optional
        The precision for symmetry determination (default is 0.1).

    Returns:
    --------
    dict
        A dictionary containing detailed symmetry information, such as crystal system, space group 
        number, point group, and Wyckoff positions.

    Examples:
    ---------
    # Example usage:
    # Perform symmetry analysis on a structure
    .. highlight:: python
    .. code-block:: python

        symmetry_info = manager.perform_symmetry_analysis(structure, symprec=0.05)
    """

    sym_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    sym_dataset = sym_analyzer.get_symmetry_dataset()

    symmetry_info = {'symmetry':{}}
    symmetry_info['symmetry']["crystal_system"] = sym_analyzer.get_crystal_system()
    symmetry_info['symmetry']["number"] = sym_analyzer.get_space_group_number()
    symmetry_info['symmetry']["point_group"] = sym_analyzer.get_point_group_symbol()
    symmetry_info['symmetry']["symbol"] = sym_analyzer.get_hall()
    symmetry_info['symmetry']["symprec"] = symprec
    symmetry_info['symmetry']["version"] = str(spglib.__version__)
    symmetry_info['symmetry']["wyckoffs"] = sym_dataset['wyckoffs']

    return symmetry_info
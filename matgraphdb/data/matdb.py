import json
import os
import logging
from typing import Callable, Union, List, Tuple, Dict
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Composition
from parquetdb import ParquetDB
from parquetdb.core.parquetdb import NormalizeConfig, LoadConfig
import spglib

from matgraphdb import config
from matgraphdb.utils.mp_utils import multiprocess_task
from matgraphdb.utils.general_utils import set_verbosity

logger = logging.getLogger(__name__)
    
class MatDB:
    """
    This class is intended to be the Data Access Layer for the Material Database.
    It provides methods for adding, reading, updating, and deleting materials from the database.
    """

    def __init__(self,  db_name: str = 'main', db_dir: str = 'MatDB', n_cores=8, verbose=3):
        """
        Initializes the `MatDB` by setting the database directory and number of cores.

        This constructor sets up the `MatDB` by specifying the directory where the database 
        is located and configuring the number of cores to be used. It also initializes the connection to the 
        database and defines the main dataset name.

        Parameters:
        -----------
        db_dir : str
            The directory where the database is stored.
        n_cores : int, optional
            The number of CPU cores to use for operations (default is 8).
        verbose : int, optional
            The verbosity level for logging (default is 3).

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Initialize the MatDB with a database directory and the default number of cores
        .. highlight:: python
        .. code-block:: python

            from matgraphdb.data.material_manager import MatDB
            manager = MatDB(db_dir='/path/to/db')
        """
        set_verbosity(verbose)
        
        self.db_dir=db_dir
        self.n_cores=n_cores
        self.db_name=db_name

        logger.info(f"Initializing MatDB with database at {db_dir}")
        self.db = ParquetDB(self.db_name, dir = self.db_dir)

    def add(self, structure: Structure = None,
                coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                coords_are_cartesian: bool = False,
                species: List[str] = None,
                lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                properties: dict = None,
                include_symmetry: bool = True,
                calculate_funcs: List[Callable]=None,
                save_db: bool = True,
                schema:pa.Schema=None,
                metadata:dict=None,
                normalize_dataset:bool=False,
                normalize_config:NormalizeConfig=NormalizeConfig(),
                verbose: int = 3,
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
        save_db : bool, optional
            If True, saves the material entry to the database (default is True).
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.
        metadata : dict, optional
            A dictionary containing the metadata to be set.
        normalize_dataset : bool, optional
            If True, normalizes the dataset.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from ParquetDB.
        verbose : int, optional
            The verbosity level for logging (default is 3).
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
        set_verbosity(verbose)

        # Generating entry data
        entry_data={}

        if properties is None:
            properties={}
        if calculate_funcs is None:
            calculate_funcs=[]
        if include_symmetry:
            calculate_funcs.append(partial(perform_symmetry_analysis, symprec=kwargs.get('symprec',0.1)))
            
        logger.info("Adding a new material.")
        

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

        df = pd.DataFrame([entry_data])
        
        logger.debug(f'Input dataframe head - \n{df.head(1)}')
        logger.debug(f'Input dataframe shape - {df.shape}')
        try:
            if save_db:
                create_kwargs=dict(data=df, schema=schema, metadata=metadata, normalize_dataset=normalize_dataset, normalize_config=normalize_config)
                self.db.create(**create_kwargs)
                logger.info("Material added successfully.")
            return entry_data
        except Exception as e:
            logger.exception(f"Error adding material: {e}")
        
        return entry_data
    
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
    
    def add_many(self, materials: Union[List[dict]],  
                schema:pa.Schema=None,
                metadata:dict=None,
                normalize_dataset:bool=False,
                normalize_config:NormalizeConfig=NormalizeConfig(),
                verbose: int = 3, **kwargs):
        """
        Adds multiple materials to the database in a single transaction.

        This method processes a list of materials and writes their data to the specified 
        database dataset in a single transaction. Each material should be represented as a 
        dictionary with keys corresponding to the arguments for the `add` method.

        Parameters:
        -----------
        materials : Union[List[dict]]
            A list of dictionaries where each dictionary contains the material data and 
            corresponds to the arguments for the `add` method.
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.
        metadata : dict, optional
            A dictionary containing the metadata to be set.
        normalize_dataset : bool, optional
            If True, normalizes the dataset.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from ParquetDB.
        verbose : int, optional
            The verbosity level for logging (default is 3).
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
        set_verbosity(verbose)
        logger.info(f"Adding {len(materials)} materials to the database.")
        
        add_kwargs=dict(schema=schema, metadata=metadata, normalize_dataset=normalize_dataset, 
                        normalize_config=normalize_config, verbose=verbose)
        
        results=multiprocess_task(self._add_many, materials, n_cores=self.n_cores, **add_kwargs)
        entry_data=[result for result in results if result]
        
        df = pd.DataFrame(entry_data)
        try:
            self.db.create(df, **kwargs)
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        logger.info("All materials added successfully.")
        
    def _add_many(self, material, **kwargs):
        """
        Adds a material entry to the database without saving it immediately.

        This method prepares the material data by disabling automatic database saving and then calls 
        the `add` method to process the material. It is typically used in batch processing scenarios.

        Parameters:
        -----------
        material : dict
            A dictionary containing the material data, passed as arguments to the `add` method.
        **kwargs
            Additional keyword arguments passed to the `add` method.

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
        return self.add(**material, **kwargs)

    def read(self, 
        ids: List[int] = None,
        columns: List[str] = None,
        filters: List[pc.Expression] = None,
        load_format: str = 'table',
        batch_size:int=None,
        include_cols: bool = True,
        rebuild_nested_struct: bool = False,
        rebuild_nested_from_scratch: bool = False,
        load_config:LoadConfig=LoadConfig(),
        normalize_config:NormalizeConfig=NormalizeConfig(),
        verbose: int = 3
        ):
        """
        Reads data from the database.

        Parameters
        ----------
        
        ids : list of int, optional
            A list of IDs to read. If None, all data is read (default is None).
        columns : list of str, optional
            The columns to include in the output. If None, all columns are included (default is None).
        filters : list of pyarrow.compute.Expression, optional
            Filters to apply to the data (default is None).
        load_format : str, optional
            The format of the returned data: 'table' or 'batches' (default is 'table').
        batch_size : int, optional
            The batch size to use for loading data in batches. If None, data is loaded as a whole (default is None).
        include_cols : bool, optional
            If True, includes only the specified columns. If False, excludes the specified columns (default is True).
        rebuild_nested_struct : bool, optional
            If True, rebuilds the nested structure (default is False).
        rebuild_nested_from_scratch : bool, optional
            If True, rebuilds the nested structure from scratch (default is False).
        load_config : LoadConfig, optional
            Configuration for loading data, optimizing performance by managing memory usage.
        normalize_config : NormalizeConfig, optional
            Configuration for the normalization process, optimizing performance by managing row distribution and file structure.

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
                load_format='table'
            )

            # Convert to pandas DataFrame
            df=materials_dataset.to_pandas()

            # Apply filters to the query
            materials_dataset = manager.read(
                filters=[pc.field('formula') == 'Fe', pc.field('density') > 7.0],
                load_format='table'
            )

            # Read materials in batches
            materials_generator = manager.read(
                batch_size=100,
                load_format='batch_generator'
            )
            for batch_table in materials_generator:
                # Do something with the batch dataset


            # Read materials as an unloaded dataset
            dataset = manager.read()
        """
        set_verbosity(verbose)
        
        logger.debug(f"Reading materials.")
        logger.debug(f"ids: {ids}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"include_cols: {include_cols}")
        logger.debug(f"filters: {filters}")
        logger.debug(f"load_format: {load_format}")
        logger.debug(f"batch_size: {batch_size}")

        kwargs=dict(ids=ids, columns=columns, include_cols=include_cols, 
                    filters=filters, load_format=load_format, batch_size=batch_size,
                    rebuild_nested_struct=rebuild_nested_struct, rebuild_nested_from_scratch=rebuild_nested_from_scratch,
                    load_config=load_config, normalize_config=normalize_config)
        return self.db.read(**kwargs)
    
    def update(self, data: Union[List[dict], dict, pd.DataFrame, pa.Table], 
               schema=None, 
               metadata=None,
               normalize_config=NormalizeConfig(), 
               verbose: int = 3):
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
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.
        metadata : dict, optional
            A dictionary containing the metadata to be set.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from ParquetDB.
        verbose : int, optional
            The verbosity level for logging (default is 3).

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

        """
        set_verbosity(verbose)

        logger.info(f"Updating data")
        self.db.update(data, schema=schema, metadata=metadata, normalize_config=normalize_config)
        logger.info("Data updated successfully.")

    def delete(self, ids:List[int]=None, columns:List[str]=None,
               normalize_config: NormalizeConfig = NormalizeConfig(),
               verbose: int = 3):
        """
        Deletes records from the database by ID.

        This method deletes specific records from the database based on the provided list of IDs.

        Parameters:
        -----------
        ids : List[int]
            A list of record IDs to delete from the database.
        columns : List[str], optional
            A list of column names to delete from the database.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from ParquetDB.
        verbose : int, optional
            The verbosity level for logging (default is 3).

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
        set_verbosity(verbose)
        
        logger.info(f"Deleting data {ids}")
        self.db.delete(ids=ids, columns=columns, normalize_config=normalize_config)
        logger.info("Data deleted successfully.")

    def normalize(self, normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Normalizes the dataset.
        """
        self.db.normalize(normalize_config=normalize_config)
    
    def update_schema(self, field_dict:dict=None, schema:pa.Schema=None, normalize_config:NormalizeConfig=NormalizeConfig()):
        """
        Updates the schema of a specified dataset.

        This method allows updating the schema of a dataset in the database. You can either provide a dictionary 
        specifying new field types or directly pass a new schema.

        Parameters:
        -----------
        field_dict : dict, optional
            A dictionary where the keys are field names and the values are the new field types.
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from ParquetDB.

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

            current_schema = manager.get_schema()
            density_field = current_schema.get_field_index('density')
            new_schema = current_schema.set(density_field, pa.float32())
            manager.update_schema(schema=new_schema)
        """

        self.db.update_schema(field_dict=field_dict, schema=schema, normalize_config=normalize_config)

    def get_schema(self):
        """
        Retrieves the schema of a specified dataset.

        This method returns the schema of the specified dataset in the database, which includes the 
        field names and their data types.


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

            schema = manager.get_schema()
        """

        return self.db.get_schema()
    
    def get_metadata(self):
        """
        Retrieves the metadata of a specified dataset.

        This method returns the metadata associated with the specified dataset in the database.


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

            metadata = manager.get_metadata()
        """
        return self.db.get_metadata()
    
    def get_field_names(self):
        """
        Retrieves the field names of a specified dataset.
        """
        return self.db.get_field_names()
    
    def get_metadata(self):
        """
        Retrieves the metadata of a specified dataset.
        """
        return self.db.get_metadata()
    
    def set_metadata(self, metadata: dict):
        """
        Sets the metadata for a specified dataset.

        This method updates the metadata of the specified dataset with the provided dictionary.

        Parameters:
        -----------
        metadata : dict
            A dictionary containing the metadata to be set.

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
            manager.set_metadata(metadata=metadata)
        """

        self.db.set_metadata(metadata=metadata)

    def drop_dataset(self):
        """
        Drops a specified dataset from the database.

        This method permanently removes the specified dataset and its contents from the database.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Drop the 'main' dataset from the database
        .. highlight:: python
        .. code-block:: python

            manager.drop_dataset()
        """

        self.db.drop_dataset()

    def rename_dataset(self, new_name:str):
        """
        Renames a dataset in the database.

        This method changes the name of an existing dataset to a new specified name.

        Parameters:
        -----------
        new_name : str
            The new name for the dataset.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Rename a dataset from 'old_dataset' to 'new_name'
        .. highlight:: python
        .. code-block:: python

            manager.rename_dataset(new_name='new_name')
        """

        self.db.rename_dataset(new_name=new_name)

    def copy_dataset(self, dest_name:str, overwrite:bool=False):
        """
        Copies a dataset in the database to a new dataset.

        This method creates a copy of an existing dataset under a new name. Additional parameters 
        can be passed to customize the copying process.

        Parameters:
        -----------
        dest_name : str
            The new name for the dataset.
        overwrite : bool, optional
            If True, overwrites the destination dataset if it already exists.

        Returns:
        --------
        None

        Examples:
        ---------
        # Example usage:
        # Copy a dataset from 'main' to 'main_backup'
        .. highlight:: python
        .. code-block:: python

            manager.copy_dataset(dest_name='main_backup')
        """

        self.db.copy_dataset( dest_name=dest_name, overwrite=overwrite)


    def export_dataset(self, export_dir:str=None, export_format:str='parquet'):
        """Export a dataset in the database.

        Args:
            export_dir (str, optional): The directory where the exported dataset will be saved.
            export_format (str, optional): The format of the exported dataset. Defaults to 'parquet'.
            **kwargs: Additional keyword arguments to pass to the ParquetDB export_dataset method.

        Returns:
            None
        """        
        self.db.export_dataset(file_path=export_dir, format=export_format)

    def export_partitioned_dataset(self, 
                                   export_dir: str, 
                                   partitioning,
                                   partitioning_flavor=None,
                                   load_config:LoadConfig=LoadConfig(),
                                   load_format:str='parquet',
                                   batch_size: int = None, 
                                   **kwargs):
        """
        Exports a partitioned dataset to a specified directory.

        This method exports a dataset to the specified directory using the provided partitioning scheme. 
        Additional parameters allow customization of the export process, such as setting the partitioning flavor 
        and controlling the batch size.

        Parameters:
        -----------
        export_dir : str
            The directory where the dataset will be exported.
        partitioning : dict
            The partitioning scheme to use for the dataset.
        partitioning_flavor : str, optional
            The partitioning flavor to use (e.g., 'hive' or another supported flavor).
        load_config : LoadConfig, optional
            The load configuration to use for the dataset.
        load_format : str, optional
            The format of the exported dataset. Defaults to 'parquet'.
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
        if batch_size:
            load_config.batch_size=batch_size
        self.db.export_partitioned_dataset(
                                           file_path=export_dir, 
                                           partitioning=partitioning, 
                                           partitioning_flavor=partitioning_flavor, 
                                           load_config=load_config,
                                           load_format=load_format,
                                           **kwargs)
    
    
        
    

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
    symmetry_info['symmetry']["wyckoffs"] = sym_dataset.wyckoffs

    return symmetry_info
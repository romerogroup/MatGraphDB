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

def check_all_params_provided(**kwargs):
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
    Convert between Cartesian and fractional coordinates based on lattice vectors.
    
    Args:
        coords (numpy.ndarray): A 1D or 2D array of coordinates.
        lattice (numpy.ndarray): A 3x3 matrix representing the lattice vectors.
        coords_are_cartesian (bool): Whether the input coordinates are in Cartesian (True) or fractional (False).
    
    Returns:
        frac_coords (numpy.ndarray): Fractional coordinates.
        cart_coords (numpy.ndarray): Cartesian coordinates.
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
    Perform symmetry analysis on a structure.

    Args:
        structure (Structure): The structure to be analyzed.
        symprec (float, optional): The symmetry precision. Defaults to 0.1.

    Returns:
        dict: A dictionary containing the symmetry information.
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


class MaterialDatabaseManager:
    """This class is intended to be the Data Access Layer for the Material Database.
    It provides methods for adding, reading, updating, and deleting materials from the database.
    """

    def __init__(self, db_dir: str, n_cores=8):
        self.db_dir=db_dir
        self.n_cores=n_cores
        self.main_table_name='main'

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
                table_name:str='main',
                save_db: bool = True,
                **kwargs
            ):
        """
        Add a material to the database.

        Args:
            
            coords (Union[List[Tuple[float,float,float]], np.ndarray], optional): Atomic coordinates.
            coords_are_cartesian (bool, optional): If the coordinates are cartesian.
            species (List[str], optional): Atomic species.
            lattice (Union[List[Tuple[float,float,float]], np.ndarray], optional): Lattice parameters.
            structure (Atoms, optional): The atomic structure in ASE format.
            composition (Union[str, dict], optional): The composition of the material.
            properties (dict, optional): Additional properties of the material.
            include_symmetry (bool, optional): If True, include symmetry information in the entry data.
            calculate_funcs (List[Callable], optional): A list of functions to calculate additional properties.
                This must rturn a dictionary with the calculated properties.
                ```python
                def calculate_property(structure: Structure):
                    do_something(structure)
                    return {'property1': value1, 'property2': value2}
                magager.add(calculate_funcs=[calculate_property])
                ```
            table_name (str, optional): The name of the table to add the data to.
            save_db (bool, optional): If True, save the entry data to the database.
            **kwargs:  Additional keyword arguments to pass to the ParquetDB create method.
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
                self.db.create(entry_data, table_name=table_name, **kwargs)
                logger.info("Material added successfully.")
            return entry_data
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        
        return entry_data
    
    def add_many(self, materials: List[Dict], table_name:str='main', **kwargs):
        """
        Write many materials to the database in a single transaction.
        
        Args:
            materials (List[Dict]): A list of dictionaries where the dictionary keys should 
            be the arguments to pass to the add method.
            table_name (str, optional): The name of the table to add the data to.
            **kwargs: Additional keyword arguments to pass to the ParquetDB create method.
            
        """
        logger.info(f"Adding {len(materials)} materials to the database.")
        results=multiprocess_task(self._add_many, materials, n_cores=self.n_cores)
        entry_data=[result for result in results if result]
        try:
            self.db.create(entry_data, table_name=table_name, **kwargs)
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        logger.info("All materials added successfully.")

    def read(self, ids=None, 
            columns:List[str]=None, 
            include_cols:bool=True, 
            filters: List[pc.Expression]=None,
            output_format='table',
            table_name:str='main', 
            batch_size=None):
        """Read materials from the database by ID."""
        logger.debug(f"Reading materials.")
        logger.debug(f"ids: {ids}")
        logger.debug(f"table_name: {table_name}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"include_cols: {include_cols}")
        logger.debug(f"filters: {filters}")
        logger.debug(f"output_format: {output_format}")
        logger.debug(f"batch_size: {batch_size}")

        kwargs=dict(ids=ids, table_name=table_name, columns=columns, include_cols=include_cols, 
                    filters=filters, output_format=output_format, batch_size=batch_size)
        return self.db.read(**kwargs)
    
    def update(self, data: Union[List[dict], dict, pd.DataFrame], 
               table_name='main', 
               field_type_dict=None):
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
        logger.info(f"Updating data")
        self.db.update(data, table_name=table_name, field_type_dict=field_type_dict)
        logger.info("Data updated successfully.")

    def delete(self, ids:List[int], table_name:str='main'):
        logger.info(f"Deleting data {ids}")
        self.db.delete(ids, table_name=table_name)
        logger.info("Data deleted successfully.")

    def get_schema(self, table_name:str ='main'):
        """Get the schema of a table.

        Args:
            table_name (str): The name of the table.

        Returns:
            pyarrow.Schema: The schema of the table.
        """
        return self.db.get_schema(table_name=table_name)
    
    def update_schema(self, table_name:str='main', field_dict=None, schema=None):
        """Update the schema of a table.

        Args:
            table_name (str): The name of the table.
            field_dict (dict, optional): A dictionary where the keys are the field names and the values are the new field types.
            schema (pyarrow.Schema, optional): The new schema to be set.

        Returns:
            None
        """
        self.db.update_schema(table_name=table_name, field_dict=field_dict, schema=schema)

    def get_tables(self):
        """Get a list of all tables in the database."""
        return self.db.get_tables()
    
    def get_metadata(self, table_name:str='main'):
        """Get the metadata of a table.
        
        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata of the table.
        """

        return self.db.get_metadata(table_name=table_name)
    
    def set_metadata(self, metadata: dict, table_name:str):
        """Set the metadata of a table.
        
        Args:
            metadata (dict): The metadata to be set.
            table_name (str): The name of the table.

        Returns:
            None
        """
        self.db.set_metadata(metadata=metadata, table_name=table_name)

    def drop_table(self, table_name:str='main'):
        """Drop a table from the database.

        Args:
            table_name (str): The name of the table to be dropped.

        Returns:
            None
        """
        self.db.drop_table(table_name=table_name)

    def rename_table(self, old_table_name:str, new_table_name:str):
        """Rename a table in the database.

        Args:
            old_table_name (str): The current name of the table.
            new_table_name (str): The new name of the table.

        Returns:
            None
        """
        self.db.rename_table(old_table_name=old_table_name, new_table_name=new_table_name)

    def copy_table(self, old_table_name:str, new_table_name:str, **kwargs):
        """Copy a table in the database.

        Args:
            old_table_name (str): The current name of the table.
            new_table_name (str): The new name of the table.

        Returns:
            None
        """
        self.db.copy_table(old_table_name=old_table_name, new_table_name=new_table_name, **kwargs)

    def optimize_table(self, table_name:str='main', **kwargs):
        """Optimize a table in the database.

        Args:
            table_name (str): The name of the table to be optimized.
            **kwargs: Additional keyword arguments to pass to the ParquetDB optimize_table method.

        Returns:
            None
        """
        self.db.optimize_table(table_name=table_name, **kwargs)

    def export_table(self, table_name:str='main', export_dir:str=None, export_format:str='parquet', **kwargs):
        """Export a table in the database.

        Args:
            table_name (str): The name of the table to be exported.
            export_dir (str, optional): The directory where the exported table will be saved.
            export_format (str, optional): The format of the exported table. Defaults to 'parquet'.
            **kwargs: Additional keyword arguments to pass to the ParquetDB export_table method.

        Returns:
            None
        """        
        self.db.export_table(table_name=table_name, file_path=export_dir, format=export_format, **kwargs)

    def export_partitioned_dataset(self, table_name: str, 
                                   export_dir: str, 
                                   partitioning,
                                   partitioning_flavor=None,
                                   batch_size: int = None, 
                                   **kwargs):
        """
        This method exports a partitioned dataset to a specified file format.

        Args:
            table_name (str): The name of the table to export.
            export_dir (str): The directory to export the data to.
            partitioning (dict): The partitioning to use for the dataset.
            partitioning_flavor (str): The partitioning flavor to use.
            batch_size (int): The batch size.
            **kwargs: Additional keyword arguments to pass to the pq.write_to_dataset function.

        """
        self.db.export_partitioned_dataset(table_name=table_name, 
                                           file_path=export_dir, 
                                           partitioning=partitioning, 
                                           partitioning_flavor=partitioning_flavor, 
                                           batch_size=batch_size, 
                                           **kwargs)
    
    def _init_structure(self, structure, coords, coords_are_cartesian, species, lattice):
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
        material['save_db']=False
        return self.add(**material)

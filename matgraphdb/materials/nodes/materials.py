import json
import logging
import os
from functools import partial
from glob import glob
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import spglib
from parquetdb import ParquetDB
from parquetdb.core.parquetdb import LoadConfig, NormalizeConfig
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.core.nodes import NodeStore, node_generator
from matgraphdb.utils.general_utils import set_verbosity
from matgraphdb.utils.mp_utils import multiprocess_task

logger = logging.getLogger(__name__)


class MaterialStore(NodeStore):
    """
    A class that inherits from NodeStore.
    """

    def create_material(
        self,
        structure: Structure = None,
        coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
        coords_are_cartesian: bool = False,
        species: List[str] = None,
        lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
        properties: dict = None,
        fields_metadata: dict = None,
        schema: pa.Schema = None,
        metadata: dict = None,
        treat_fields_as_ragged: List[str] = [],
        convert_to_fixed_shape: bool = True,
        normalize_dataset: bool = False,
        normalize_config: NormalizeConfig = NormalizeConfig(),
        verbose: int = 3,
        save_db: bool = True,
        **kwargs,
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
        fields_metadata : dict, optional
            A dictionary containing the metadata to be set for the fields.
        schema : pyarrow.Schema, optional
            A new schema to be applied to the dataset.
        metadata : dict, optional
            A dictionary containing the metadata to be set.
        treat_fields_as_ragged : List[str], optional
            A list of fields to be treated as ragged.
        convert_to_fixed_shape : bool, optional
            If True, converts the fields to fixed shape.
        normalize_dataset : bool, optional
            If True, normalizes the dataset.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from Parquet
        verbose : int, optional
            The verbosity level for logging (default is 3).
        save_db : bool, optional
            If True, saves the material to the database.
        **kwargs
            Additional keyword arguments passed to the ParquetDB `create` method.

        Returns:
        --------
        dict
            A dictionary containing the material's data, including calculated properties and additional information.
        """
        set_verbosity(verbose)

        # Generating entry data
        entry_data = {}

        if properties is None:
            properties = {}

        treat_fields_as_ragged.extend(
            ["frac_coords", "cartesian_coords", "atomic_numbers", "species"]
        )

        logger.info("Adding a new material.")

        structure = self._init_structure(
            structure, coords, coords_are_cartesian, species, lattice
        )

        if structure is None:
            logger.error("A structure must be provided.")
            raise ValueError("Either a structure must be provided")

        composition = structure.composition
        entry_data = {}
        entry_data["formula"] = composition.formula
        entry_data["elements"] = list(
            [element.symbol for element in composition.elements]
        )

        entry_data["lattice"] = structure.lattice.matrix.tolist()
        entry_data["frac_coords"] = structure.frac_coords.tolist()
        entry_data["cartesian_coords"] = structure.cart_coords.tolist()
        entry_data["atomic_numbers"] = structure.atomic_numbers
        entry_data["species"] = list([specie.symbol for specie in structure.species])
        entry_data["nelements"] = len(structure.species)
        entry_data["volume"] = structure.volume
        entry_data["density"] = structure.density
        entry_data["nsites"] = len(structure.sites)
        entry_data["density_atomic"] = entry_data["nsites"] / entry_data["volume"]
        entry_data["structure"] = structure.as_dict()

        # Adding other properties as columns
        entry_data.update(properties)

        df = pd.DataFrame([entry_data])

        logger.debug(f"Input dataframe head - \n{df.head(1)}")
        logger.debug(f"Input dataframe shape - {df.shape}")

        try:
            if save_db:
                logger.debug(f"Saving material to database")
                create_kwargs = dict(
                    data=df,
                    schema=schema,
                    metadata=metadata,
                    fields_metadata=fields_metadata,
                    treat_fields_as_ragged=treat_fields_as_ragged,
                    convert_to_fixed_shape=convert_to_fixed_shape,
                    normalize_dataset=normalize_dataset,
                    normalize_config=normalize_config,
                )
                self.create(**create_kwargs)
                logger.info("Material added successfully.")
            else:
                logger.info("Material not saved to database")

        except Exception as e:
            logger.exception(f"Error adding material: {e}")

        return entry_data

    def _init_structure(
        self, structure, coords, coords_are_cartesian, species, lattice
    ):
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
            logger.debug(
                "Building Structure structure from provided coordinates, species, and lattice."
            )
            if coords_are_cartesian:
                return Structure(
                    lattice=lattice,
                    species=species,
                    coords=coords,
                    coords_are_cartesian=True,
                )
            else:
                return Structure(
                    lattice=lattice,
                    species=species,
                    coords=coords,
                    coords_are_cartesian=False,
                )
        else:
            logger.debug("No valid structure information provided.")
            return None

    def create_materials(
        self,
        materials: Union[List[dict], dict, pd.DataFrame, pa.Table, pa.RecordBatch],
        schema: pa.Schema = None,
        metadata: dict = None,
        fields_metadata: dict = None,
        treat_fields_as_ragged: List[str] = [],
        convert_to_fixed_shape: bool = True,
        normalize_dataset: bool = False,
        normalize_config: NormalizeConfig = NormalizeConfig(),
        verbose: int = 3,
        **kwargs,
    ):
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
        fields_metadata : dict, optional
            A dictionary containing the metadata to be set for the fields.
        treat_fields_as_ragged : List[str], optional
            A list of fields to be treated as ragged.
        convert_to_fixed_shape : bool, optional
            If True, converts the fields to fixed shape.
        normalize_dataset : bool, optional
            If True, normalizes the dataset.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from Parquet
        verbose : int, optional
            The verbosity level for logging (default is 3).
        **kwargs
            Additional keyword arguments passed to the ParquetDB `create` method.

        Returns:
        --------
        None
        """
        set_verbosity(verbose)
        logger.info(f"Adding {len(materials)} materials to the database.")

        add_kwargs = dict(
            schema=schema,
            metadata=metadata,
            fields_metadata=fields_metadata,
            normalize_dataset=normalize_dataset,
            normalize_config=normalize_config,
            verbose=verbose,
            treat_fields_as_ragged=treat_fields_as_ragged,
            convert_to_fixed_shape=convert_to_fixed_shape,
        )

        results = multiprocess_task(self._create_material, materials, **add_kwargs)
        entry_data = [result for result in results if result]

        df = pd.DataFrame(entry_data)
        try:
            self.create(df, **kwargs)
        except Exception as e:
            logger.error(f"Error adding material: {e}")
        logger.info("All materials added successfully.")

    def _create_material(self, material, **kwargs):
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
        """

        material["save_db"] = False
        return self.create_material(**material, **kwargs)

    def read_materials(
        self,
        ids: List[int] = None,
        columns: List[str] = None,
        filters: List[pc.Expression] = None,
        load_format: str = "table",
        batch_size: int = None,
        include_cols: bool = True,
        rebuild_nested_struct: bool = False,
        rebuild_nested_from_scratch: bool = False,
        load_config: LoadConfig = LoadConfig(),
        normalize_config: NormalizeConfig = NormalizeConfig(),
    ):
        """
        Reads data from the MaterialStore.

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
        """

        logger.debug(f"Reading materials.")
        logger.debug(f"ids: {ids}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"include_cols: {include_cols}")
        logger.debug(f"filters: {filters}")
        logger.debug(f"load_format: {load_format}")
        logger.debug(f"batch_size: {batch_size}")

        kwargs = dict(
            ids=ids,
            columns=columns,
            include_cols=include_cols,
            filters=filters,
            load_format=load_format,
            batch_size=batch_size,
            rebuild_nested_struct=rebuild_nested_struct,
            rebuild_nested_from_scratch=rebuild_nested_from_scratch,
            load_config=load_config,
            normalize_config=normalize_config,
        )
        return self.read_nodes(**kwargs)

    def update_materials(
        self,
        data: Union[List[dict], dict, pd.DataFrame, pa.Table],
        schema=None,
        metadata=None,
        fields_metadata: dict = None,
        update_keys: List[str] = ["id"],
        treat_fields_as_ragged=None,
        convert_to_fixed_shape: bool = True,
        normalize_config=NormalizeConfig(),
    ):
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
        fields_metadata : dict, optional
            A dictionary containing the metadata to be set for the fields.
        update_keys : List[str], optional
            A list of keys to be updated.
        treat_fields_as_ragged : List[str], optional
            A list of fields to be treated as ragged.
        convert_to_fixed_shape : bool, optional
            If True, converts the fields to fixed shape.
        normalize_config : NormalizeConfig, optional
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from Parquet
        verbose : int, optional
            The verbosity level for logging (default is 3).

        Returns:
        --------
        None
        """

        logger.info(f"Updating data")
        update_kwargs = dict(
            data=data,
            schema=schema,
            metadata=metadata,
            fields_metadata=fields_metadata,
            update_keys=update_keys,
            normalize_config=normalize_config,
            treat_fields_as_ragged=treat_fields_as_ragged,
            convert_to_fixed_shape=convert_to_fixed_shape,
        )

        self.update_nodes(**update_kwargs)
        logger.info("Data updated successfully.")

    def delete_materials(
        self,
        ids: List[int] = None,
        columns: List[str] = None,
        normalize_config: NormalizeConfig = NormalizeConfig(),
        verbose: int = 3,
    ):
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
            The normalize configuration to be applied to the data. This is the NormalizeConfig object from Parquet
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
        self.delete(ids=ids, columns=columns, normalize_config=normalize_config)
        logger.info("Data deleted successfully.")


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


@node_generator
def material_lattices(material_store: NodeStore):
    """
    Creates Lattice nodes if no file exists, otherwise loads them from a file.
    """
    # Retrieve material nodes with lattice properties
    try:
        # material_nodes = NodeStore(material_store_path)
        material_nodes = material_store

        table = material_nodes.read(
            columns=[
                "structure.lattice.a",
                "structure.lattice.b",
                "structure.lattice.c",
                "structure.lattice.alpha",
                "structure.lattice.beta",
                "structure.lattice.gamma",
                "structure.lattice.volume",
                "structure.lattice.pbc",
                "structure.lattice.matrix",
                "id",
                "core.material_id",
            ]
        )

        for i, column in enumerate(table.columns):
            field = table.schema.field(i)
            field_name = field.name
            if "." in field_name:
                field_name = field_name.split(".")[-1]
            if "id" == field_name:
                field_name = "material_node_id"
            new_field = field.with_name(field_name)
            table = table.set_column(i, new_field, column)

    except Exception as e:
        logger.error(f"Error creating lattice nodes: {e}")
        return None

    return table


@node_generator
def material_sites(material_store: NodeStore):
    try:
        material_nodes = material_store
        lattice_names = [
            "structure.lattice.a",
            "structure.lattice.b",
            "structure.lattice.c",
            "structure.lattice.alpha",
            "structure.lattice.beta",
            "structure.lattice.gamma",
            "structure.lattice.volume",
        ]
        id_names = ["id", "core.material_id"]
        tmp_dict = {field: [] for field in id_names}
        tmp_dict.update({field: [] for field in lattice_names})
        table = material_nodes.read(
            columns=["structure.sites", *id_names, *lattice_names]
        )
        # table=material_nodes.read(columns=['structure.sites', *id_names])#, *lattice_names])
        material_sites = table["structure.sites"].combine_chunks()

        flatten_material_sites = pc.list_flatten(material_sites)
        material_sites_length_list = pc.list_value_length(material_sites).to_numpy()

        for i, legnth in enumerate(material_sites_length_list):
            for field_name in tmp_dict.keys():
                column = table[field_name].combine_chunks()
                value = column[i]
                tmp_dict[field_name].extend([value] * legnth)
        table = None

        arrays = flatten_material_sites.flatten()
        names = flatten_material_sites.type.names

        flatten_material_sites = None
        material_sites_length_list = None

        for name, column_values in tmp_dict.items():
            arrays.append(pa.array(column_values))
            names.append(name)

        table = pa.Table.from_arrays(arrays, names=names)

        for i, column in enumerate(table.columns):
            field = table.schema.field(i)
            field_name = field.name
            if "." in field_name:
                field_name = field_name.split(".")[-1]
            if "id" == field_name:
                field_name = "material_node_id"
            new_field = field.with_name(field_name)
            table = table.set_column(i, new_field, column)

    except Exception as e:
        logger.error(f"Error creating site nodes: {e}")
        return None
    return table

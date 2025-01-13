import logging
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.core.nodes import node_generator
from matgraphdb.materials.nodes import *
from matgraphdb.utils.config import PKG_DIR

logger = logging.getLogger(__name__)

BASE_CHEMENV_FILE = os.path.join(
    PKG_DIR, "utils", "chem_utils", "resources", "coordination_geometries.parquet"
)


BASE_ELEMENT_FILE = os.path.join(
    PKG_DIR, "utils", "chem_utils", "resources", "imputed_periodic_table_values.parquet"
)


@node_generator
def elements(base_file=BASE_ELEMENT_FILE):
    """
    Creates Element nodes if no file exists, otherwise loads them from a file.
    """

    logger.info(f"Initializing element nodes from {base_file}")
    # Suppress warnings during node creation
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        file_ext = os.path.splitext(base_file)[-1][1:]
        logger.debug(f"File extension: {file_ext}")
        if file_ext == "parquet":
            df = pd.read_parquet(os.path.join(PKG_DIR, "utils", base_file))
        elif file_ext == "csv":
            df = pd.read_csv(os.path.join(PKG_DIR, "utils", base_file), index_col=0)
        else:
            raise ValueError(f"base_file must be a parquet or csv file")
        logger.debug(f"Read element dataframe shape {df.shape}")

        df["oxidation_states"] = df["oxidation_states"].apply(
            lambda x: x.replace("]", "").replace("[", "")
        )
        df["oxidation_states"] = df["oxidation_states"].apply(
            lambda x: ",".join(x.split())
        )
        df["oxidation_states"] = df["oxidation_states"].apply(
            lambda x: eval("[" + x + "]")
        )
        df["experimental_oxidation_states"] = df["experimental_oxidation_states"].apply(
            lambda x: eval(x)
        )
        df["ionization_energies"] = df["ionization_energies"].apply(lambda x: eval(x))

    except Exception as e:
        logger.error(f"Error reading element CSV file: {e}")
        return None

    return df


@node_generator
def chemenvs(base_file=BASE_CHEMENV_FILE):
    """
    Creates ChemEnv nodes if no file exists, otherwise loads them from a file.
    """

    try:
        file_ext = os.path.splitext(base_file)[-1][1:]
        logger.debug(f"File extension: {file_ext}")
        if file_ext == "parquet":
            df = pd.read_parquet(os.path.join(PKG_DIR, "utils", base_file))
        elif file_ext == "csv":
            df = pd.read_csv(os.path.join(PKG_DIR, "utils", base_file), index_col=0)
        else:
            raise ValueError(f"base_file must be a parquet or csv file")
        logger.debug(f"Read element dataframe shape {df.shape}")

        logger.debug(f"Columns: {df.columns}")
        df.drop(columns=["id"], inplace=True)

    except Exception as e:
        logger.error(f"Error creating chemical environment nodes: {e}")
        return None

    return df


@node_generator
def crystal_systems():
    """
    Creates Crystal System nodes if no file exists, otherwise loads them from a file.
    """
    try:
        crystal_systems = [
            "Triclinic",
            "Monoclinic",
            "Orthorhombic",
            "Tetragonal",
            "Trigonal",
            "Hexagonal",
            "Cubic",
        ]
        crystal_systems_properties = [{"crystal_system": cs} for cs in crystal_systems]
        df = pd.DataFrame(crystal_systems_properties)
    except Exception as e:
        logger.error(f"Error creating crystal system nodes: {e}")
        return None

    return df


@node_generator
def magnetic_states():
    """
    Creates Magnetic State nodes if no file exists, otherwise loads them from a file.
    """
    # Define magnetic states
    try:
        magnetic_states = ["NM", "FM", "FiM", "AFM", "Unknown"]
        magnetic_states_properties = [{"magnetic_state": ms} for ms in magnetic_states]
        df = pd.DataFrame(magnetic_states_properties)
    except Exception as e:
        logger.error(f"Error creating magnetic state nodes: {e}")
        return None
    return df


@node_generator
def oxidation_states():
    """
    Creates Oxidation State nodes if no file exists, otherwise loads them from a file.
    """
    try:
        oxidation_states = np.arange(-9, 10)
        oxidation_states_names = [f"OxidationState{i}" for i in oxidation_states]
        data = {"oxidation_state": oxidation_states_names, "value": oxidation_states}
        df = pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error creating oxidation state nodes: {e}")
        return None
    return df


@node_generator
def space_groups():
    """
    Creates Space Group nodes if no file exists, otherwise loads them from a file.
    """
    # Generate space group numbers from 1 to 230
    try:
        space_groups = [f"spg_{i}" for i in np.arange(1, 231)]
        space_groups_properties = [
            {"spg": int(space_group.split("_")[1])} for space_group in space_groups
        ]

        # Create DataFrame with the space group properties
        df = pd.DataFrame(space_groups_properties)
    except Exception as e:
        logger.error(f"Error creating space group nodes: {e}")
        return None

    return df


@node_generator
def wyckoffs():
    """
    Creates Wyckoff Position nodes if no file exists, otherwise loads them from a file.
    """
    try:
        space_groups = [f"spg_{i}" for i in np.arange(1, 231)]
        # Define Wyckoff letters
        wyckoff_letters = ["a", "b", "c", "d", "e", "f"]

        # Create a list of space group-Wyckoff position combinations
        spg_wyckoffs = [
            f"{spg}_{wyckoff_letter}"
            for wyckoff_letter in wyckoff_letters
            for spg in space_groups
        ]

        # Create a list of dictionaries with 'spg_wyckoff'
        spg_wyckoff_properties = [
            {"spg_wyckoff": spg_wyckoff} for spg_wyckoff in spg_wyckoffs
        ]

        # Create DataFrame with Wyckoff positions
        df = pd.DataFrame(spg_wyckoff_properties)
    except Exception as e:
        logger.error(f"Error creating Wyckoff position nodes: {e}")
        return None

    return df

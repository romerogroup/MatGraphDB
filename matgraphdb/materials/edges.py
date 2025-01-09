import logging
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.core.edges import EdgeStore, edge_generator
from matgraphdb.core.nodes import NodeStore

# from matgraphdb.materials.nodes import *
from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index

logger = logging.getLogger(__name__)


@edge_generator
def element_element_neighborsByGroupPeriod(element_store):

    try:
        connection_name = "neighborsByGroupPeriod"
        table = element_store.read_nodes(
            columns=["atomic_number", "extended_group", "period", "symbol"]
        )
        element_df = table.to_pandas()

        # Getting group-period edge index
        edge_index = get_group_period_edge_index(element_df)

        # Creating the relationships dataframe
        df = pd.DataFrame(edge_index, columns=[f"source_id", f"target_id"])

        # Dropping rows with NaN values and casting to int64
        df = df.dropna().astype(np.int64)

        # Add source and target type columns
        df["source_type"] = element_store.node_type
        df["target_type"] = element_store.node_type
        df["weight"] = 1.0

        table = ParquetDB.construct_table(df)

        reduced_table = element_store.read(
            columns=["symbol", "id", "extended_group", "period"]
        )
        reduced_source_table = reduced_table.rename_columns(
            {
                "symbol": "source_name",
                "extended_group": "source_extended_group",
                "period": "source_period",
            }
        )
        reduced_target_table = reduced_table.rename_columns(
            {
                "symbol": "target_name",
                "extended_group": "target_extended_group",
                "period": "target_period",
            }
        )

        table = pyarrow_utils.join_tables(
            table,
            reduced_source_table,
            left_keys=["source_id"],
            right_keys=["id"],
            join_type="left outer",
        )

        table = pyarrow_utils.join_tables(
            table,
            reduced_target_table,
            left_keys=["target_id"],
            right_keys=["id"],
            join_type="left outer",
        )

        names = pc.binary_join_element_wise(
            pc.cast(table["source_name"], pa.string()),
            pc.cast(table["target_name"], pa.string()),
            f"_{connection_name}_",
        )

        table = table.append_column("name", names)

        logger.debug(
            f"Created element-group-period relationships. Shape: {table.shape}"
        )
    except Exception as e:
        logger.exception(f"Error creating element-group-period relationships: {e}")
        raise e

    return table


@edge_generator
def element_oxiState_canOccur(element_store, oxiState_store):
    try:
        connection_name = "canOccur"

        element_table = element_store.read_nodes(
            columns=["id", "experimental_oxidation_states", "symbol"]
        )
        oxiState_table = oxiState_store.read_nodes(
            columns=["id", "oxidation_state", "value"]
        )

        # element_table=element_table.rename_columns({'id':'source_id'})
        element_table = element_table.append_column(
            "source_type", pa.array([element_store.node_type] * element_table.num_rows)
        )

        # oxiState_table=oxiState_table.rename_columns({'id':'target_id'})
        oxiState_table = oxiState_table.append_column(
            "target_type",
            pa.array([oxiState_store.node_type] * oxiState_table.num_rows),
        )

        element_df = element_table.to_pandas()
        oxiState_df = oxiState_table.to_pandas()
        table_dict = {
            "source_id": [],
            "source_type": [],
            "target_id": [],
            "target_type": [],
            "name": [],
            "weight": [],
        }

        oxiState_id_map = {}
        id_oxidationState_map = {}
        for i, oxiState_row in oxiState_df.iterrows():
            oxiState_id_map[oxiState_row["value"]] = oxiState_row["id"]
            id_oxidationState_map[oxiState_row["id"]] = oxiState_row["oxidation_state"]

        for i, element_row in element_df.iterrows():
            exp_oxidation_states = element_row["experimental_oxidation_states"]
            source_id = element_row["id"]
            source_type = element_store.node_type
            symbol = element_row["symbol"]
            for exp_oxidation_state in exp_oxidation_states:
                target_id = oxiState_id_map[exp_oxidation_state]
                target_type = oxiState_store.node_type
                oxi_state_name = id_oxidationState_map[target_id]

                table_dict["source_id"].append(source_id)
                table_dict["source_type"].append(source_type)
                table_dict["target_id"].append(target_id)
                table_dict["target_type"].append(target_type)
                table_dict["weight"].append(1.0)
                table_dict["name"].append(
                    f"{symbol}_{connection_name}_{oxi_state_name}"
                )

        edge_table = ParquetDB.construct_table(table_dict)

        logger.debug(
            f"Created element-oxiState-canOccur relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(f"Error creating element-oxiState-canOccur relationships: {e}")
        raise e

    return edge_table


@edge_generator
def material_chemenv_containsSite(material_store, chemenv_store):
    try:
        connection_name = "containsSite"

        material_table = material_store.read_nodes(
            columns=[
                "id",
                "core.material_id",
                "chemenv.coordination_environments_multi_weight",
            ]
        )
        chemenv_table = chemenv_store.read_nodes(columns=["id", "mp_symbol"])

        material_table = material_table.rename_columns(
            {"id": "source_id", "core.material_id": "material_name"}
        )
        material_table = material_table.append_column(
            "source_type",
            pa.array([material_store.node_type] * material_table.num_rows),
        )

        chemenv_table = chemenv_table.rename_columns(
            {"id": "target_id", "mp_symbol": "chemenv_name"}
        )
        chemenv_table = chemenv_table.append_column(
            "target_type", pa.array([chemenv_store.node_type] * chemenv_table.num_rows)
        )

        material_df = material_table.to_pandas()
        chemenv_df = chemenv_table.to_pandas()
        chemenv_target_id_map = {
            row["chemenv_name"]: row["target_id"] for _, row in chemenv_df.iterrows()
        }

        table_dict = {
            "source_id": [],
            "source_type": [],
            "target_id": [],
            "target_type": [],
            "name": [],
            "weight": [],
        }

        for _, row in material_df.iterrows():
            coord_envs = row["chemenv.coordination_environments_multi_weight"]
            if coord_envs is None:
                continue

            source_id = row["source_id"]
            material_name = row["material_name"]

            for coord_env in coord_envs:
                try:
                    chemenv_name = coord_env[0]["ce_symbol"]
                    target_id = chemenv_target_id_map[chemenv_name]
                except:
                    continue

                table_dict["source_id"].append(source_id)
                table_dict["source_type"].append(material_store.node_type)
                table_dict["target_id"].append(target_id)
                table_dict["target_type"].append(chemenv_store.node_type)

                name = f"{material_name}_{connection_name}_{chemenv_name}"
                table_dict["name"].append(name)
                table_dict["weight"].append(1.0)

        edge_table = ParquetDB.construct_table(table_dict)

        logger.debug(
            f"Created material-chemenv-containsSite relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(
            f"Error creating material-chemenv-containsSite relationships: {e}"
        )
        raise e

    return edge_table


@edge_generator
def material_crystalSystem_has(material_store, crystal_system_store):
    try:
        connection_name = "has"

        material_table = material_store.read_nodes(
            columns=["id", "core.material_id", "symmetry.crystal_system"]
        )
        crystal_system_table = crystal_system_store.read_nodes(
            columns=["id", "crystal_system"]
        )

        material_table = material_table.rename_columns(
            {"id": "source_id", "symmetry.crystal_system": "crystal_system"}
        )
        material_table = material_table.append_column(
            "source_type",
            pa.array([material_store.node_type] * material_table.num_rows),
        )

        crystal_system_table = crystal_system_table.rename_columns({"id": "target_id"})
        crystal_system_table = crystal_system_table.append_column(
            "target_type",
            pa.array([crystal_system_store.node_type] * crystal_system_table.num_rows),
        )

        edge_table = pyarrow_utils.join_tables(
            material_table,
            crystal_system_table,
            left_keys=["crystal_system"],
            right_keys=["crystal_system"],
            join_type="left outer",
        )

        edge_table = edge_table.append_column(
            "weight", pa.array([1.0] * edge_table.num_rows)
        )

        names = pc.binary_join_element_wise(
            pc.cast(edge_table["core.material_id"], pa.string()),
            pc.cast(edge_table["crystal_system"], pa.string()),
            f"_{connection_name}_",
        )

        edge_table = edge_table.append_column("name", names)

        logger.debug(
            f"Created material-crystalSystem-has relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(
            f"Error creating material-crystalSystem-has relationships: {e}"
        )
        raise e

    return edge_table


@edge_generator
def material_element_has(material_store, element_store):
    try:
        connection_name = "has"

        material_table = material_store.read_nodes(
            columns=["id", "core.material_id", "core.elements"]
        )
        element_table = element_store.read_nodes(columns=["id", "symbol"])

        material_table = material_table.rename_columns(
            {"id": "source_id", "core.material_id": "material_name"}
        )
        material_table = material_table.append_column(
            "source_type", pa.array(["material"] * material_table.num_rows)
        )

        element_table = element_table.rename_columns({"id": "target_id"})
        element_table = element_table.append_column(
            "target_type", pa.array(["elements"] * element_table.num_rows)
        )

        material_df = material_table.to_pandas()
        element_df = element_table.to_pandas()
        element_target_id_map = {
            row["symbol"]: row["target_id"] for _, row in element_df.iterrows()
        }

        table_dict = {
            "source_id": [],
            "source_type": [],
            "target_id": [],
            "target_type": [],
            "name": [],
            "weight": [],
        }

        for _, row in material_df.iterrows():
            elements = row["core.elements"]
            source_id = row["source_id"]
            material_name = row["material_name"]
            if elements is None:
                continue

            # Append the material name for each element in the species list
            for element in elements:

                target_id = element_target_id_map[element]
                table_dict["source_id"].append(source_id)
                table_dict["source_type"].append(material_store.node_type)
                table_dict["target_id"].append(target_id)
                table_dict["target_type"].append(element_store.node_type)

                name = f"{material_name}_{connection_name}_{element}"
                table_dict["name"].append(name)
                table_dict["weight"].append(1.0)

        edge_table = ParquetDB.construct_table(table_dict)

        logger.debug(
            f"Created material-element-has relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(f"Error creating material-element-has relationships: {e}")
        raise e

    return edge_table


@edge_generator
def material_lattice_has(material_store, lattice_store):
    try:
        connection_name = "has"

        material_table = material_store.read_nodes(columns=["id", "core.material_id"])
        lattice_table = lattice_store.read_nodes(columns=["material_node_id"])

        material_table = material_table.rename_columns(
            {"id": "source_id", "core.material_id": "material_id"}
        )
        material_table = material_table.append_column(
            "source_type",
            pa.array([material_store.node_type] * material_table.num_rows),
        )

        lattice_table = lattice_table.append_column(
            "target_id", lattice_table["material_node_id"].combine_chunks()
        )
        lattice_table = lattice_table.append_column(
            "target_type", pa.array([lattice_store.node_type] * lattice_table.num_rows)
        )

        edge_table = pyarrow_utils.join_tables(
            material_table,
            lattice_table,
            left_keys=["source_id"],
            right_keys=["material_node_id"],
            join_type="left outer",
        )

        edge_table = edge_table.append_column(
            "weight", pa.array([1.0] * edge_table.num_rows)
        )

        logger.debug(
            f"Created material-lattice-has relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(f"Error creating material-lattice-has relationships: {e}")
        raise e

    return edge_table


@edge_generator
def material_spg_has(material_store, spg_store):
    try:
        connection_name = "has"

        material_table = material_store.read_nodes(
            columns=["id", "core.material_id", "symmetry.number"]
        )
        spg_table = spg_store.read_nodes(columns=["id", "spg"])

        material_table = material_table.rename_columns(
            {"id": "source_id", "symmetry.number": "spg"}
        )
        material_table = material_table.append_column(
            "source_type",
            pa.array([material_store.node_type] * material_table.num_rows),
        )

        spg_table = spg_table.rename_columns({"id": "target_id"})
        spg_table = spg_table.append_column(
            "target_type", pa.array([spg_store.node_type] * spg_table.num_rows)
        )

        edge_table = pyarrow_utils.join_tables(
            material_table,
            spg_table,
            left_keys=["spg"],
            right_keys=["spg"],
            join_type="left outer",
        )

        edge_table = edge_table.append_column(
            "weight", pa.array([1.0] * edge_table.num_rows)
        )

        names = pc.binary_join_element_wise(
            pc.cast(edge_table["core.material_id"], pa.string()),
            pc.cast(edge_table["spg"], pa.string()),
            f"_{connection_name}_SpaceGroup",
        )

        edge_table = edge_table.append_column("name", names)

        logger.debug(
            f"Created material-spg-has relationships. Shape: {edge_table.shape}"
        )
    except Exception as e:
        logger.exception(f"Error creating material-spg-has relationships: {e}")
        raise e

    return edge_table


@edge_generator
def element_chemenv_canOccur(element_store, chemenv_store, material_store):
    try:

        material_table = material_store.read_nodes(
            columns=[
                "id",
                "core.material_id",
                "core.elements",
                "chemenv.coordination_environments_multi_weight",
            ]
        )

        chemenv_table = chemenv_store.read_nodes(columns=["id", "mp_symbol"])
        element_table = element_store.read_nodes(columns=["id", "symbol"])

        chemenv_table = chemenv_table.rename_columns({"mp_symbol": "name"})
        chemenv_table = chemenv_table.append_column(
            "target_type", pa.array([chemenv_store.node_type] * chemenv_table.num_rows)
        )

        element_table = element_table.rename_columns({"symbol": "name"})
        element_table = element_table.append_column(
            "source_type", pa.array([element_store.node_type] * element_table.num_rows)
        )

        material_df = material_table.to_pandas()
        chemenv_df = chemenv_table.to_pandas()
        element_df = element_table.to_pandas()

        chemenv_target_id_map = {
            row["name"]: row["id"] for _, row in chemenv_df.iterrows()
        }
        element_target_id_map = {
            row["name"]: row["id"] for _, row in element_df.iterrows()
        }

        table_dict = {
            "source_id": [],
            "source_type": [],
            "target_id": [],
            "target_type": [],
            "name": [],
        }

        for _, row in material_df.iterrows():
            coord_envs = row["chemenv.coordination_environments_multi_weight"]

            if coord_envs is None:
                continue

            elements = row["core.elements"]

            for i, coord_env in enumerate(coord_envs):
                try:
                    chemenv_name = coord_env[0]["ce_symbol"]
                    element_name = elements[i]

                    source_id = element_target_id_map[element_name]
                    target_id = chemenv_target_id_map[chemenv_name]
                except:
                    continue

                table_dict["source_id"].append(source_id)
                table_dict["source_type"].append(element_store.node_type)
                table_dict["target_id"].append(target_id)
                table_dict["target_type"].append(chemenv_store.node_type)

                name = f"{element_name}_canOccur_{chemenv_name}"
                table_dict["name"].append(name)

        edge_table = ParquetDB.construct_table(table_dict)

        logger.debug(
            f"Created element-chemenv-canOccur relationships. Shape: {edge_table.shape}"
        )

    except Exception as e:
        logger.exception(f"Error creating element-chemenv-canOccur relationships: {e}")
        raise e

    return edge_table

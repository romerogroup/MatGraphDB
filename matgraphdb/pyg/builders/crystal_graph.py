import logging

import numpy as np
import torch
from torch_geometric.data import Data

from matgraphdb import MatGraphDB

logger = logging.getLogger(__name__)


class CrystalGraphBuilder:
    """
    A class to build PyG graphs from a materials database (mdb).
    """

    def __init__(
        self,
        mdb: MatGraphDB,
        target="elasticity.g_vrh",
        bond_connections_key="bonding.electric_consistent.bond_connections",
        bond_orders_key="bonding.electric_consistent.bond_orders",
        element_feature_keys=None,
        material_feature_keys=None,
        material_filters=None,
        apply_log_to_target=False,
    ):
        """
        Initializes the CrystalGraphBuilder with default parameters and the materials database.

        Args:
            mdb: Your materials database object (assumed to have node_stores).
            target (str): The name of the target property (label) to predict.
            bond_connections_key (str): Column/key for bond connections.
            bond_orders_key (str): Column/key for bond orders.
            element_feature_keys (List[str]): Which element-level features to use.
            material_feature_keys (List[str]): Which material-level features to use.
            filters: Any optional filters to apply when reading from the database.
            apply_log_to_target (bool): If True, apply log10 transform to the target.
        """
        logger.info("Initializing CrystalGraphBuilder")

        if element_feature_keys is None:
            element_feature_keys = [
                "atomic_mass",
                "radius_covalent",
                "radius_vanderwaals",
                "heat_specific",
            ]
            logger.debug(f"Using default element features: {element_feature_keys}")
        else:
            logger.debug(f"Using provided element features: {element_feature_keys}")

        if material_feature_keys is None:
            material_feature_keys = []
            logger.debug("No material features specified")
        else:
            logger.debug(f"Using material features: {material_feature_keys}")

        self.mdb = mdb
        self.target = target
        self.bond_connections_key = bond_connections_key
        self.bond_orders_key = bond_orders_key
        self.element_feature_keys = element_feature_keys
        self.material_feature_keys = material_feature_keys
        self.material_filters = material_filters
        self.apply_log_to_target = apply_log_to_target

        logger.info(f"Target property: {target}")
        logger.info(f"Apply log transform: {apply_log_to_target}")
        if material_filters:
            logger.info(f"Using filters: {material_filters}")

        # Will be populated once you call build_element_feature_map()
        self.element_feature_map = {}

    def build_element_feature_map(self):
        """
        Reads element data from the database and constructs a feature map (symbol -> feature vector).
        """
        logger.info("Building element feature map")
        element_store = self.mdb.node_stores["elements"]
        element_df = element_store.read_nodes(
            columns=[
                "symbol",
                *self.element_feature_keys,
            ],
        ).to_pandas(split_blocks=True, self_destruct=True)

        self.element_feature_map = {
            row["symbol"]: np.array([row[feat] for feat in self.element_feature_keys])
            for _, row in element_df.iterrows()
        }
        logger.debug(
            f"Created feature map for {len(self.element_feature_map)} elements"
        )

    def fetch_material_table(self):
        """
        Reads the material table from the database with the required columns.
        Applies filters if provided, and drops rows with null values.
        """
        logger.info("Fetching material data from database")
        material_store = self.mdb.node_stores["materials"]
        table = material_store.read_nodes(
            columns=[
                self.bond_connections_key,
                self.bond_orders_key,
                "structure.sites",
                "id",
                "core.material_id",
                self.target,
                *self.material_feature_keys,
            ],
            filters=self.material_filters,
        ).drop_null()
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        logger.info(f"Retrieved {len(df)} materials after filtering")
        return df

    def build(self):
        """
        Main method that:
         1) Builds element feature map
         2) Fetches material table
         3) Optionally applies log transform to target
         4) Creates PyG Data objects
         5) Returns a list of Data objects
        """
        logger.info("Starting graph building process")

        # 1. Build the element feature map
        self.build_element_feature_map()

        # 2. Fetch material data
        df = self.fetch_material_table()

        # 3. Optionally apply log transform
        if self.apply_log_to_target:
            logger.info("Applying log10 transform to target values")
            df[self.target] = np.log10(df[self.target])

        # 4. Loop over rows to create Data objects
        logger.info("Creating PyG Data objects")
        data_list = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.debug(f"Processing material {idx}/{len(df)}")
            data = self.create_pyg_graph(row)
            if data is not None:
                data_list.append(data)

        logger.info(f"Successfully created {len(data_list)} graphs")
        return data_list

    def create_pyg_graph(self, row):
        """
        Creates a PyTorch Geometric Data object from a DataFrame row,
        which contains bond connections, bond orders, sites, and the target property.
        """
        try:
            bond_connections = row[self.bond_connections_key]
            bond_orders = row[self.bond_orders_key]
            sites = row["structure.sites"]
            material_id = row["core.material_id"]
            parquetdb_id = row["id"]
            target_value = row[self.target]
        except KeyError as e:
            logger.error(
                f"Missing key while processing material {row.get('core.material_id', 'unknown')}: {e}"
            )
            return None

        x = []
        edge_index = []
        edge_attr = []
        pos = []

        try:
            for i_site, site in enumerate(sites):
                site_bond_orders = bond_orders[i_site]
                site_connections = bond_connections[i_site]

                # Position
                pos.append(site["xyz"])
                # Node feature
                element = site["label"]
                if element not in self.element_feature_map:
                    logger.warning(
                        f"Unknown element {element} in material {material_id}"
                    )
                x.append(
                    self.element_feature_map.get(
                        element, np.zeros(len(self.element_feature_keys))
                    )
                )

                # Edges
                for j_index, j_site in enumerate(site_connections):
                    site_bond_order = site_bond_orders[j_index]
                    edge_index.append([int(i_site), int(j_site)])
                    edge_attr.append(site_bond_order)
        except Exception as e:
            logger.error(f"Error creating graph for material {material_id}: {e}")
            return None

        # Convert to torch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_attr = torch.tensor([edge_attr], dtype=torch.float32)
        pos = torch.tensor(np.array(pos), dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([target_value], dtype=torch.float32)

        # Build the PyG Data object
        data = Data()
        data.x = x
        data.edge_index = edge_index.t().contiguous()
        data.edge_attr = edge_attr.t().contiguous()
        data.pos = pos
        data.y = y
        data.material_id = material_id
        data.id = parquetdb_id

        logger.debug(
            f"Created graph for material {material_id} with {len(x)} nodes and {len(edge_index)} edges"
        )
        return data

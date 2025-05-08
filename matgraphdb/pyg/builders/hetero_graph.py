import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from parquetdb import ParquetGraphDB
from parquetdb.utils import pyarrow_utils
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class HeteroGraphBuilder:
    """
    A class to build PyTorch Geometric graphs from MatGraphDB data.

    This provides a clean interface to convert MatGraphDB nodes and relationships
    into PyTorch Geometric HeteroData objects for machine learning.
    """

    def __init__(self, graph_db: ParquetGraphDB):
        """
        Initialize the graph builder.

        Parameters
        ----------
        graph_db : MatGraphDB
            The material graph database to build graphs from
        """
        self.graph_db = graph_db
        self.hetero_data = HeteroData()
        self._homo_data = None
        self.node_id_mappings = {}

    @property
    def node_types(self):
        return self.hetero_data.node_types

    @property
    def edge_types(self):
        return self.hetero_data.edge_types

    def _process_columns(self, table, encoders: Dict = None):
        if encoders is None:
            encoders = {}
        arrays = []
        for column_name in table.column_names:
            column = table[column_name].combine_chunks()
            column_type = column.type
            if pyarrow_utils.is_fixed_shape_tensor(column_type):
                column_values = column.to_numpy_ndarray()
            elif (
                pa.types.is_floating(column_type)
                or pa.types.is_integer(column_type)
                or pa.types.is_boolean(column_type)
                or pa.types.is_string(column_type)
            ):
                column_values = column.to_numpy()
            elif pa.types.is_list(column_type) or pa.types.is_struct(column_type):
                column_values = column.to_pandas(split_blocks=True, self_destruct=True)
            else:
                raise NotImplementedError(f"Unsupported column type: {column_type}")

            if column_name in encoders:
                x = encoders[column_name](column_values)
            else:
                x = torch.tensor(column_values, dtype=torch.float32)

            arrays.append(x)

        return torch.stack(arrays, dim=1)

    def add_node_type(
        self,
        node_type: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        embedding_vectors: bool = False,
        label_column: Optional[str] = None,
        drop_null: bool = True,
        encoders: Optional[Dict] = None,
        read_kwargs: Optional[Dict] = None,
    ):
        """
        Add a node type to the graph with specified features and targets.

        Parameters
        ----------
        node_type : str
            The type of node to add (e.g., 'materials', 'elements')
        columns : List[str], optional
            Columns to use as node features
        filters : Dict, optional
            Filters to apply when selecting nodes
        encoders : Dict, optional
            Custom encoders for specific columns
        """
        logger.info(f"Adding {node_type} nodes to graph")

        ids, torch_tensor, feature_names, labels = self._process_node_type(
            node_type=node_type,
            columns=columns,
            filters=filters,
            encoders=encoders,
            label_column=label_column,
            read_kwargs=read_kwargs,
            drop_null=drop_null,
        )

        logger.info(f"ids: {ids.shape}")

        self.hetero_data[node_type].node_ids = torch.tensor(ids, dtype=torch.int64)
        if labels is not None:
            self.hetero_data[node_type].labels = labels

        if torch_tensor is not None:
            logger.info(f"torch_tensor: {torch_tensor.shape}")
            logger.info(f"feature_names: {feature_names}")

            self.hetero_data[node_type].x = torch_tensor
            logger.info(f"x: {self.hetero_data[node_type].x.shape}")
            self.hetero_data[node_type].feature_names = feature_names

        if embedding_vectors:
            num_nodes = len(self.hetero_data[node_type].node_ids)
            self.hetero_data[node_type].x = torch.eye(num_nodes)

        self.hetero_data[node_type].num_nodes = len(torch.tensor(ids))

    def add_target_node_property(
        self,
        node_type: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        label_column: Optional[str] = None,
        encoders: Optional[Dict] = None,
        read_kwargs: Optional[Dict] = None,
        drop_null: bool = True,
    ):
        if node_type not in self.hetero_data.node_types:
            raise ValueError(f"Node type {node_type} has not been added to the graph")

        ids, torch_tensor, feature_names, labels = self._process_node_type(
            node_type=node_type,
            columns=columns,
            filters=filters,
            encoders=encoders,
            label_column=label_column,
            read_kwargs=read_kwargs,
            drop_null=drop_null,
        )

        logger.info(f"ids: {ids.shape}")
        logger.info(f"torch_tensor: {torch_tensor.shape}")
        # logger.info(f"feature_names: {feature_names}")

        target_feature_ids = torch.tensor(ids, dtype=torch.int64)

        all_feature_ids = self.hetero_data[node_type].node_ids.clone().detach()
        # all_feature_ids = torch.tensor(
        #     self.hetero_data[node_type].node_id, dtype=torch.int64
        # )
        target_feature_mask = torch.isin(
            target_feature_ids, all_feature_ids, assume_unique=True
        )

        logger.info(f"target_feature_ids: {target_feature_ids.shape}")
        logger.info(f"all_feature_ids: {all_feature_ids.shape}")
        logger.info(f"target_feature_mask: {target_feature_mask.shape}")

        self.hetero_data[node_type].target_feature_mask = target_feature_mask
        if labels is not None:
            self.hetero_data[node_type].labels = labels
        self.hetero_data[node_type].y_index = target_feature_ids
        self.hetero_data[node_type].y = torch_tensor[target_feature_mask]
        logger.info(f"y: {self.hetero_data[node_type].y.shape}")

        self.hetero_data[node_type].y_label_name = feature_names
        self.hetero_data[node_type].target_feature_names = feature_names

        if len(torch_tensor.shape) > 1:
            self.hetero_data[node_type].out_channels = torch_tensor.shape[1]
        else:
            self.hetero_data[node_type].out_channels = 1

    def _process_node_type(
        self,
        node_type: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        drop_null: bool = True,
        label_column: Optional[str] = None,
        encoders: Optional[Dict] = None,
        read_kwargs: Optional[Dict] = None,
    ):

        if read_kwargs is None:
            read_kwargs = {}

        if columns is None:
            all_columns = []
        else:
            all_columns = columns.copy()

        if "id" not in all_columns:
            all_columns.append("id")

        if label_column is not None:
            all_columns.append(label_column)

        # Read nodes from database
        table = self.graph_db.read_nodes(
            node_type=node_type,
            columns=all_columns,
            filters=filters,
            **read_kwargs,
        )
        if drop_null:
            table = table.drop_null()

        # Add node features
        identification_table = table.select(["id"])
        ids = identification_table["id"].combine_chunks().to_numpy()

        if label_column is not None:
            labels = table[label_column].combine_chunks().to_pylist()
        else:
            labels = None

        torch_tensor = None
        feature_names = None
        if columns:
            feature_table = table.select(columns)
            torch_tensor = self._process_columns(feature_table, encoders)
            feature_names = columns

        return ids, torch_tensor, feature_names, labels

    def add_edge_type(
        self,
        edge_type: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        encoders: Optional[Dict] = None,
        read_kwargs: Optional[Dict] = None,
        drop_null: bool = True,
    ):
        """
        Add edges between nodes with optional features and targets.

        Parameters
        ----------
        edge_type : str
            The type of edge to add (e.g., 'MATERIAL-HAS-ELEMENT')
        columns : List[str], optional
            Columns to use as edge features
        filters : Dict, optional
            Filters to apply when selecting edges
        encoders : Dict, optional
            Custom encoders for specific columns
        """
        if read_kwargs is None:
            read_kwargs = {}

        identification_columns = [
            "id",
            "source_id",
            "source_type",
            "target_id",
            "target_type",
            "edge_type",
        ]

        if columns is None:
            all_columns = []
        else:
            all_columns = columns.copy()

        for column in identification_columns:
            if column not in all_columns:
                all_columns.append(column)

        # Read edges from database
        table = self.graph_db.read_edges(
            edge_type,
            columns=all_columns,
            filters=filters,
            **read_kwargs,
        )
        if drop_null:
            table = table.drop_null()

        # identification_table = table.select(["source_id", source "target_id"])

        source_node_types = pc.unique(table["source_type"]).to_pylist()
        target_node_types = pc.unique(table["target_type"]).to_pylist()

        for source_node_type in source_node_types:
            for target_node_type in target_node_types:
                if source_node_type not in self.hetero_data.node_types:
                    raise ValueError(
                        f"Node type {source_node_type} has not been added to the graph"
                    )
                if target_node_type not in self.hetero_data.node_types:
                    raise ValueError(
                        f"Node type {target_node_type} has not been added to the graph"
                    )

                edge_filter_expression = (
                    pc.field("source_type") == source_node_type
                ) & (pc.field("target_type") == target_node_type)
                edge_table = table.filter(edge_filter_expression)

                # If no edges are found, skip this source-target pair
                if edge_table.num_rows == 0:
                    continue

                source_ids = edge_table["source_id"].combine_chunks().to_numpy()
                target_ids = edge_table["target_id"].combine_chunks().to_numpy()
                edge_name = edge_table["edge_type"].combine_chunks()[0].as_py()

                edge_index = torch.from_numpy(np.array([source_ids, target_ids]))
                logger.info(f"edge_index: {edge_index.shape}")

                self.hetero_data[
                    source_node_type, edge_name, target_node_type
                ].edge_index = edge_index

                if columns:
                    feature_table = edge_table.select(columns)
                    torch_tensor = self._process_columns(feature_table, encoders)
                    logger.info(f"torch_tensor: {torch_tensor.shape}")
                    self.hetero_data[
                        source_node_type, edge_name, target_node_type
                    ].edge_attr = torch_tensor

                    logger.info(
                        f"edge_attr: {self.hetero_data[source_node_type, edge_name, target_node_type].edge_attr.shape}"
                    )

                    self.hetero_data[
                        source_node_type, edge_name, target_node_type
                    ].property_names = columns

    def add_target_edge_property(
        self,
        edge_type: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        drop_null: bool = True,
        encoders: Optional[Dict] = None,
        read_kwargs: Optional[Dict] = None,
    ):
        """
        Add edges between nodes with optional features and targets.

        Parameters
        ----------
        edge_type : str
            The type of edge to add (e.g., 'MATERIAL-HAS-ELEMENT')
        columns : List[str], optional
            Columns to use as edge features
        filters : Dict, optional
            Filters to apply when selecting edges
        encoders : Dict, optional
            Custom encoders for specific columns
        """
        if read_kwargs is None:
            read_kwargs = {}

        identification_columns = [
            "id",
            "source_id",
            "source_type",
            "target_id",
            "target_type",
            "edge_type",
        ]

        if columns is None:
            all_columns = []
        else:
            all_columns = columns.copy()

        for column in identification_columns:
            if column not in all_columns:
                all_columns.append(column)

        # Read edges from database
        table = self.graph_db.read_edges(
            edge_type,
            columns=all_columns,
            filters=filters,
            **read_kwargs,
        )
        if drop_null:
            table = table.drop_null()

        source_node_types = pc.unique(table["source_type"]).to_pylist()
        target_node_types = pc.unique(table["target_type"]).to_pylist()

        for source_node_type in source_node_types:
            for target_node_type in target_node_types:
                if source_node_type not in self.hetero_data.node_types:
                    raise ValueError(
                        f"Node type {source_node_type} has not been added to the graph"
                    )
                if target_node_type not in self.hetero_data.node_types:
                    raise ValueError(
                        f"Node type {target_node_type} has not been added to the graph"
                    )

                edge_filter_expression = (
                    pc.field("source_type") == source_node_type
                ) & (pc.field("target_type") == target_node_type)
                edge_table = table.filter(edge_filter_expression)

                # If no edges are found, skip this source-target pair
                if edge_table.num_rows == 0:
                    continue

                feature_table = edge_table.select(columns)
                torch_tensor = self._process_columns(feature_table, encoders)

                logger.info(f"torch_tensor: {torch_tensor.shape}")

                source_ids = edge_table["source_id"].combine_chunks().to_numpy()
                target_ids = edge_table["target_id"].combine_chunks().to_numpy()
                edge_name = edge_table["edge_type"].combine_chunks()[0].as_py()

                relation_type = (source_node_type, edge_name, target_node_type)
                if relation_type not in self.hetero_data.edge_types:
                    raise ValueError(
                        f"Relation type {relation_type} has not been added to the graph"
                    )

                target_edge_index = torch.tensor(np.array([source_ids, target_ids]))
                logger.info(f"target_edge_index: {target_edge_index.shape}")

                all_feature_index = self.hetero_data[
                    source_node_type, edge_name, target_node_type
                ].edge_index
                logger.info(f"all_feature_index: {all_feature_index.shape}")

                target_feature_mask = torch.isin(
                    target_edge_index, all_feature_index, assume_unique=True
                ).all(dim=0)

                self.hetero_data[
                    source_node_type, edge_name, target_node_type
                ].target_feature_mask = target_feature_mask
                logger.info(
                    f"target_feature_mask: {self.hetero_data[source_node_type, edge_name, target_node_type].target_feature_mask.shape}"
                )

                self.hetero_data[source_node_type, edge_name, target_node_type].y = (
                    torch_tensor[target_feature_mask]
                )
                logger.info(
                    f"y: {self.hetero_data[source_node_type, edge_name, target_node_type].y.shape}"
                )

                self.hetero_data[
                    source_node_type, edge_name, target_node_type
                ].y_label_name = columns
                logger.info(
                    f"y_label_name: {self.hetero_data[source_node_type, edge_name, target_node_type].y_label_name}"
                )

                if len(torch_tensor.shape) > 1:
                    self.hetero_data[
                        source_node_type, edge_name, target_node_type
                    ].out_channels = torch_tensor.shape[1]
                else:
                    self.hetero_data[
                        source_node_type, edge_name, target_node_type
                    ].out_channels = 1

    @property
    def homo_data(self):
        """Convert heterogeneous graph to homogeneous if all node features match."""
        if self._homo_data is None:
            self._homo_data = self.hetero_data.to_homogeneous()
        return self._homo_data

    def save(self, path: str):
        """Save the graph to a file."""
        torch.save(self.hetero_data, path)

    @classmethod
    def load(cls, graph_db: ParquetGraphDB, path: str):
        """Load a saved graph."""
        builder = cls(graph_db)
        builder.hetero_data = torch.load(path)
        return builder

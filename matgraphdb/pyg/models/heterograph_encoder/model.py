import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric import nn as pyg_nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

from matgraphdb.pyg.models.base import MultiLayerPercetronLayer


class GNNEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels=None,
        num_conv_layers=1,
        num_ffw_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.num_conv_layers = num_conv_layers
        self.num_ffw_layers = num_ffw_layers
        self.out_channels = out_channels

        # Create ModuleLists to store layers
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.ffw_lins = torch.nn.ModuleList()
        self.ffw_batch_norms = torch.nn.ModuleList()

        self.input_conv = pyg_nn.SAGEConv((-1, -1), hidden_channels, aggr="mean")
        for _ in range(num_conv_layers):
            self.convs.append(pyg_nn.SAGEConv((-1, -1), hidden_channels, aggr="mean"))
            self.lins.append(pyg_nn.Linear(-1, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_ffw_layers):
            self.ffw_lins.append(pyg_nn.Linear(-1, hidden_channels))
            self.ffw_batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Last layer
        if self.out_channels is not None:
            self.out_conv = pyg_nn.SAGEConv((-1, -1), out_channels, aggr="mean")
            self.proj = pyg.nn.Linear(-1, out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index).relu()
            # x = self.activation(x)
            # x = self.lins[i](x).relu()
            # x = self.activation(x)
            # x = self.batch_norms[i](x)
            x = self.dropout(x)

        for i in range(self.num_ffw_layers):
            x = self.ffw_lins[i](x).relu()
            # x = self.ffw_batch_norms[i](x)
            x = self.dropout(x)

        if self.out_channels is not None:
            x = self.proj(x)

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, src_node_name, tgt_node_name, num_layers=1, dropout=0.0):
        super().__init__()
        self.src_node_name = src_node_name
        self.tgt_node_name = tgt_node_name
        self.mlp = MultiLayerPercetronLayer(
            input_dim=2 * hidden_channels, num_layers=num_layers, dropout=dropout
        )
        self.mlp_out = torch.nn.Linear(2 * hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[self.src_node_name][row], z_dict[self.tgt_node_name][col]], dim=-1)

        z = self.mlp(z)
        z = self.mlp_out(z)
        return z.view(-1).sigmoid()


class MaterialEdgePredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        data=None,
        decoder_kwargs=None,
        encoder_kwargs=None,
        use_shallow_embedding_for_materials=False,
        use_projections=False,
        use_embeddings=False,
    ):
        super().__init__()
        if not use_projections and not use_embeddings:
            raise ValueError("Either use_projections or use_embeddings must be True")

        metadata = data.metadata()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.use_shallow_embedding_for_materials = use_shallow_embedding_for_materials
        self.use_projections = use_projections
        self.use_embeddings = use_embeddings
        if use_embeddings:
            node_embeddings = {}
            for node_type in self.node_types:
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    continue
                node_embeddings[node_type] = torch.nn.Embedding(
                    data[node_type].num_nodes, hidden_channels
                )
            self.node_embeddings = torch.nn.ModuleDict(node_embeddings)
        if use_projections:
            node_projections = {}
            for node_type in self.node_types:
                node_projections[node_type] = torch.nn.Linear(
                    data[node_type].x.shape[1], hidden_channels
                )
            self.node_projections = torch.nn.ModuleDict(node_projections)

        if decoder_kwargs is None:
            decoder_kwargs = {}
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, **encoder_kwargs)
        if data is not None:
            self.encoder = pyg.nn.to_hetero(self.encoder, data.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels, **decoder_kwargs)

    def process_input_nodes(self, x_dict, node_ids=None):
        z_dict = {}

        for node_type in self.node_types:
            if self.use_projections and self.use_embeddings:
                z_dict[node_type] = self.node_projections[node_type](x_dict[node_type])
                if node_ids is None:
                    raise ValueError(
                        "node_ids must be provided if use_embeddings is True"
                    )
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    continue
                z_dict[node_type] += self.node_embeddings[node_type](
                    node_ids[node_type]
                )
            elif self.use_projections and not self.use_embeddings:
                z_dict[node_type] = self.node_projections[node_type](x_dict[node_type])
            elif not self.use_projections and self.use_embeddings:
                if node_ids is None:
                    raise ValueError(
                        "node_ids must be provided if use_embeddings is True"
                    )
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    raise ValueError(
                        "use_shallow_embedding_for_materials must be False if use_embeddings is True and use_projections is False"
                    )
                z_dict[node_type] = self.node_embeddings[node_type](node_ids[node_type])
            else:
                raise ValueError(
                    "Either use_projections or use_embeddings must be True"
                )
        return z_dict

    def forward(self, x_dict, edge_index_dict, edge_label_index, node_ids=None):
        z_dict = self.process_input_nodes(x_dict, node_ids)
        z_dict = self.encoder(z_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def encode(self, x_dict, edge_index_dict, node_ids=None):
        z_dict = self.process_input_nodes(x_dict, node_ids)
        return self.encoder(z_dict, edge_index_dict)

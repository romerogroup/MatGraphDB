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
    def __init__(self, hidden_channels, 
                 num_layers=1, 
                 dropout=0.0,
                 src_node_type='materials',
                 dst_node_type='elements',
                 ):
        super().__init__()
        self.src_node_type = src_node_type
        self.dst_node_type = dst_node_type

        self.mlp = MultiLayerPercetronLayer(
            input_dim=2 * hidden_channels, num_layers=num_layers, dropout=dropout
        )
        self.mlp_out = torch.nn.Linear(2 * hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[self.src_node_type][row], z_dict[self.dst_node_type][col]], dim=-1)

        z = self.mlp(z)
        z = self.mlp_out(z)
        return z.view(-1).sigmoid()


class HeteroEncoder(torch.nn.Module):
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
                if data[node_type].x is not None:
                    node_projections[node_type] = torch.nn.Linear(
                        data[node_type].x.shape[1], hidden_channels
                    )
            self.node_projections = torch.nn.ModuleDict(node_projections)

        if decoder_kwargs is None:
            decoder_kwargs = {}
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, **encoder_kwargs)
        # if data is not None:
        #     self.encoder = pyg.nn.to_hetero(self.encoder, data.metadata(), aggr="sum")
        # self.decoder = EdgeDecoder(hidden_channels, **decoder_kwargs)

    def process_input_nodes(self, hetero_data):
        
        
        for node_type in self.node_types:
            x=None
            if hasattr(hetero_data[node_type], "x"):
                x=hetero_data[node_type].x
            
    
            node_ids=hetero_data[node_type].node_ids
            if self.use_projections and self.use_embeddings:
                if x is not None:
                    hetero_data[node_type].z = self.node_projections[node_type](x)
                
                if node_ids is None:
                    raise ValueError(
                        "node_ids must be provided if use_embeddings is True"
                    )
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    continue
                if hasattr(hetero_data[node_type], "z"):
                    hetero_data[node_type].z += self.node_embeddings[node_type](node_ids)
                else:
                    hetero_data[node_type].z = self.node_embeddings[node_type](node_ids)
                
            elif self.use_projections and not self.use_embeddings:
                hetero_data[node_type].z = self.node_projections[node_type](x)
                
                
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
                hetero_data[node_type].z = self.node_embeddings[node_type](node_ids)
            else:
                raise ValueError(
                    "Either use_projections or use_embeddings must be True"
                )
                
            
        return hetero_data

    def forward(self, hetero_data):
        hetero_data = self.process_input_nodes(hetero_data)
        data = hetero_data.to_homogeneous()
        z = self.encoder(data.z, data.edge_index)
        data.z=z
        return data
    
    def encode(self, hetero_data):
        hetero_data = self.process_input_nodes(hetero_data)
        data = hetero_data.to_homogeneous()
        z = self.encoder(data.z, data.edge_index)
        return z


    
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric import nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn import to_hetero


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



class PropInit(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_partitions=None,
        k_steps=2,
        heterodata=None,
        use_projections=False,
        use_embeddings=False,
        use_shallow_embedding_for_materials=False,
        use_projections_for_materials=False,
    ):

        super().__init__()

        self.n_partitions = n_partitions
        self.hidden_channels = hidden_channels
        self.k_steps = k_steps
        self.use_projections = use_projections
        self.use_embeddings = use_embeddings
        self.use_shallow_embedding_for_materials = use_shallow_embedding_for_materials
        self.use_projections_for_materials = use_projections_for_materials

        self.node_types = heterodata.metadata()[0] 
        self.edge_types = heterodata.metadata()[1]  
        self.n_node_types = len(self.node_types)
        self.n_edge_types = len(self.edge_types)
        
        self.node_type_emb = torch.nn.Embedding(self.n_node_types, hidden_channels)
        
        partition_embs = {}
        for partition_i in range(self.n_partitions):
            partition_embs[str(partition_i)] = torch.nn.Embedding(self.n_node_types, hidden_channels)
        self.partition_embs = torch.nn.ModuleDict(partition_embs)
        
        
        # Build node embeddings or projections per node type.
        if use_embeddings:
            node_embeddings = {}
            for node_type in self.node_types:
                # If you want to avoid learning embeddings for materials using shallow embeddings:
                if not self.use_shallow_embedding_for_materials and node_type == "materials":
                    continue
                node_embeddings[node_type] = torch.nn.Embedding(
                    heterodata[node_type].num_nodes, hidden_channels
                )
            self.node_embeddings = torch.nn.ModuleDict(node_embeddings)
        if use_projections:
            node_projections = {}
            for node_type in self.node_types:
                if not self.use_projections_for_materials and node_type == "materials":
                    continue
                if hasattr(heterodata[node_type], "x"):
                    node_projections[node_type] = torch.nn.Linear(
                        heterodata[node_type].x.shape[1], hidden_channels
                    )
            self.node_projections = torch.nn.ModuleDict(node_projections)
        
        self.encoder = GNNEncoder(hidden_channels=hidden_channels,
                                 num_conv_layers=self.k_steps,
                                 num_ffw_layers=3)
        self.encoder = pyg_nn.to_hetero(self.encoder, metadata=heterodata.metadata(), aggr="sum")
        self.out_layer = torch.nn.Linear(hidden_channels, 1)

    def process_shallow_embedding_per_node_type(self, hetero_data, node_type):
        x = torch.zeros(hetero_data[node_type].num_nodes, self.hidden_channels)
        
        if not self.use_shallow_embedding_for_materials and node_type == "materials":
            return x
            
        if self.use_embeddings:
            x += self.node_embeddings[node_type](hetero_data[node_type].node_ids)
        return x
    
    def process_projections_per_node_type(self, hetero_data, node_type):
        x = torch.zeros(hetero_data[node_type].num_nodes, self.hidden_channels)
        
        if not self.use_projections_for_materials and node_type == "materials":
            return x
        
        if self.use_projections and hasattr(hetero_data[node_type], "x"):
            x += self.node_projections[node_type](hetero_data[node_type].x)
        return x
    
    
    def forward(self, data_batch):
        z_dict = {}
        for i_node_type, node_type in enumerate(self.node_types):
            
            node_data = data_batch[node_type]
            # Ensure node_type_id is a tensor of shape [num_nodes]
            node_type_ids = node_data.node_type_id  
            # Compute the node type embeddings for all nodes at once.
            node_type_embs = self.node_type_emb(node_type_ids)  # Shape: [num_nodes, hidden_channels]
            
            # Convert partition information to a tensor (if not already)
            if not torch.is_tensor(node_data.partition):
                partition_ids = torch.tensor(node_data.partition, device=node_type_ids.device)
            else:
                partition_ids = node_data.partition

            # Compute the partition embeddings for all nodes in all partitions at once.
            # This creates a tensor of shape [n_partitions, num_nodes, hidden_channels].
            partition_embs_all = torch.stack([
                self.partition_embs[str(part)](node_type_ids)
                for part in range(self.n_partitions)
            ], dim=0)
            
            # For each node, select the partition embedding corresponding to its partition id.
            # The advanced indexing below picks, for each node j, the embedding from partition_ids[j].
            num_nodes = node_type_ids.size(0)
            partition_embs = partition_embs_all[partition_ids.long(), torch.arange(num_nodes)]
            
            shallow_embs = self.process_shallow_embedding_per_node_type(data_batch, node_type)
            projection_embs = self.process_projections_per_node_type(data_batch, node_type)
            # Combine the node type and partition embeddings.
            z_dict[node_type] = node_type_embs + partition_embs + shallow_embs + projection_embs


        z_dict = self.encoder(z_dict, data_batch.edge_index_dict)
        return z_dict
    
class Model(torch.nn.Module):
    def __init__(self, out_channels, out_node_type, hidden_channels, n_partitions, k_steps, heterodata, use_projections, use_embeddings, use_shallow_embedding_for_materials, use_projections_for_materials):
        super().__init__()
        self.propinit = PropInit(hidden_channels=hidden_channels, 
                                 n_partitions=n_partitions,
                                 k_steps=k_steps, 
                                 heterodata=heterodata, 
                                 use_projections=use_projections, 
                                 use_embeddings=use_embeddings, 
                                 use_shallow_embedding_for_materials=use_shallow_embedding_for_materials,
                                 use_projections_for_materials=use_projections_for_materials)
        self.out_layer = torch.nn.Linear(hidden_channels, out_channels)
        self.out_node_type = out_node_type
        
    def forward(self, data_batch):
        z_dict = self.propinit(data_batch)
        return self.out_layer(z_dict[self.out_node_type])
    
    def get_embeddings(self, data_batch):
        z_dict = self.propinit(data_batch)
        return z_dict[self.out_node_type]
        
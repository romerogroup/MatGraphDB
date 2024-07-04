# Creating a GraphSAGE model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv,to_hetero
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

# Define the model
class StackedSAGELayers(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,dropout=0.2 ):
        super(StackedSAGELayers, self).__init__()
        self.dropout=dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 2)
        ])
        

    def forward(self, x: torch.Tensor, edge_index:  torch.Tensor,training=False):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout,training=training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=self.dropout,training=training)
        
        return x



class UnsupervisedHeteroSAGEModel(nn.Module):
    def __init__(self, data,
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 device='cuda:0'):
        super(UnsupervisedHeteroSAGEModel, self).__init__()
        self.embs={}
        self.data_lins={}
        for node_type in data.node_types:
            self.embs[node_type]=nn.Embedding(data[node_type].num_nodes, 
                                                    hidden_channels,device=device)
            if data[node_type].num_node_features != 0:
                self.data_lins[node_type]=nn.Linear(data[node_type].num_node_features,
                                                        hidden_channels,device=device)



        self.graph_sage = StackedSAGELayers(hidden_channels, 
                                            hidden_channels, 
                                            num_layers)
        # Convert SAGE to heterogeneous variant:
        self.graph_sage = to_hetero(self.graph_sage, metadata=data.metadata())
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        x_dict={}
        for node_type in data.node_types:
            if data[node_type].num_node_features != 0:
                x_dict[node_type]= self.data_lins[node_type](data[node_type].x) + self.embs[node_type](data[node_type].node_id)
            else:
                x_dict[node_type]=self.embs[node_type](data[node_type].node_id)

        x_dict=self.graph_sage(x_dict, data.edge_index_dict)
        return x_dict
    

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch_geometric.transforms as T 

    from matgraphdb.mlcore.datasets import MaterialGraphDataset

    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                        use_weights=False,
                                        use_node_properties=True,
                                        #,properties=['group','atomic_number']
                                        )
    print(graph_dataset.data)
    # print(dir(graph_dataset.data))


    rev_edge_types=[]
    edge_types=[]
    for edge_type in graph_dataset.data.edge_types:
        rel_type=edge_type[1]
        if 'rev' in rel_type:
            rev_edge_types.append(edge_type)
        else:
            edge_types.append(edge_type)
    print(edge_types)
    print(rev_edge_types)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types, 
    )
    train_data, val_data, test_data = transform(graph_dataset.data)
    # print("Train Data")
    # print("-"*200)
    # print(train_data)
    # print("Val Data")
    # print("-"*200)
    # print(val_data)
    # print("Test Data")
    # print("-"*200)
    # print(test_data)

    # print(test_data.node_stores)
    # print(test_data.node_attrs)
    # print(test_data.num_node_features)
    # print(test_data.edge_index_dict)

    device=  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # Define the model
    test_data.to(device)
    model = UnsupervisedHeteroSAGEModel(test_data,
                            hidden_channels=128, 
                            out_channels=1, 
                            num_layers=1,device=device)
    model.to(device)
    # test_data.to(device)
    model(test_data)
    # # Define the loss function
    # loss_fn = nn.MSELoss()
    # # Define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # # compute activations for train subset
    # data = test_data.to(device)
    # test_data.to(device)

    # out = model(data)

# Creating a GraphSAGE model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv,to_hetero
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader


from matgraphdb.graph_kit.pyg.models import MultiLayerPerceptron
# Define the model
class StackedSAGELayers(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.2 ):
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




class SupervisedHeteroSAGEModel(nn.Module):
    def __init__(self, data,
                 hidden_channels:int, 
                 out_channels:int,
                 pred_node_type:str,
                 num_layers,
                 device='cuda:0'):
        super(SupervisedHeteroSAGEModel, self).__init__()

        self.embs = nn.ModuleDict()
        self.data_lins = nn.ModuleDict()
        self.node_type=pred_node_type
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            num_features = data[node_type].num_node_features
            self.embs[node_type]=nn.Embedding(num_nodes,hidden_channels,device=device)
            if num_features != 0:
                self.data_lins[node_type]=nn.Linear(num_features, hidden_channels,device=device)

        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
        # Initialize and convert GraphSAGE to heterogeneous
        self.graph_sage = StackedSAGELayers(hidden_channels,hidden_channels,num_layers)
        self.graph_sage = to_hetero(self.graph_sage, metadata=data.metadata())

    def forward(self, data):
        x_dict={}
        for node_type, emb_layer in self.embs.items():
            # Handling nodes based on feature availability
            if node_type in self.data_lins:
                x_dict[node_type] = self.data_lins[node_type](data[node_type].x) + emb_layer(data[node_type].node_id)
            else:
                x_dict[node_type] = emb_layer(data[node_type].node_id)

        x_dict=self.graph_sage(x_dict, data.edge_index_dict)

        out=self.output_layer(x_dict[self.node_type])
        return out
    



def get_node_dataloaders(data,shuffle=False):
    input_loaders={}
    for node_item in data.node_items():
        node_type=node_item[0]
        node_dict=node_item[1]
        node_ids=node_dict['node_id']
        input_nodes=(node_type, node_ids)
        test_loader = NeighborLoader(
            data,
            # Sample 15 neighbors for each node and each edge type for 2 iterations:
            num_neighbors=[15] * 2,
            replace=False,
            subgraph_type="bidirectional",
            disjoint=False,
            weight_attr = None,
            transform=None,
            transform_sampler_output = None,
            
            input_nodes=input_nodes,
            shuffle=shuffle,
            batch_size=128,
        )
        input_loaders[node_type]=test_loader

    return input_loaders

def train(model, optimizer, dataloader_dict, loss_fn=nn.CrossEntropyLoss()):
    model.train()
    # optimizer.zero_grad()
    # node_train_loss = 0.0
    
                
                
            
    # train_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    # batch_train_loss = batch_train_loss / num_batches

    # print(f"Loss: {batch_train_loss}")
    # node_train_loss += batch_train_loss
    # return batch_train_loss

def evaluate(model, dataloader_dict):
    model.eval()
    with torch.no_grad():
        for node_type,dataloader in dataloader_dict.items():
            for data in dataloader:
                data.to(device)
                z_dict = model(data)
                z_dict = model(data)
        loss = negative_sampling_hetero_loss(z_dict, data.edge_index_dict)
    return loss.item()

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch_geometric.transforms as T 
    from torch_geometric.sampler import NegativeSampling
    from matgraphdb.mlcore.datasets import MaterialGraphDataset
    from matgraphdb.mlcore.loss import negative_sampling_hetero_loss,positive_sampling_hetero_random_walk_loss
    from torch_geometric.loader import NeighborLoader




    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                        use_weights=True,
                                        use_node_properties=True,
                                        properties=['atomic_number','group','row','atomic_mass']
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

    # transform = T.RandomNodeSplit(split="random")  # Or another appropriate method

    train_data, val_data, test_data = transform(graph_dataset.data)
    print(dir(test_data))
    # training_graph, _ = InMemoryDataset.collate(train_data_list)
    # test_graph, _ = InMemoryDataset.collate(test_data_list)

    # print(test_data['material', 'has', 'element'].edge_label_index)
    # print(test_data['material', 'has', 'element'].edge_label)
    # print(test_data['material', 'has', 'element'].edge_label.shape)
    # print(test_data['material', 'has', 'chemenv'].edge_label)
    # print(test_data['material', 'has', 'chemenv'].edge_label.shape)


    # print(test_data['element', 'electric_connects', 'element'].edge_label)
    # print(test_data['element', 'electric_connects', 'element'].edge_label.shape)
    # for x in test_data['material', 'has', 'chemenv'].edge_label:
    #     print(x)
    # print(test_data.edge_index)
    # edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    # edge_label = train_data["user", "rates", "movie"].edge_la

    # train_loader = LinkNeighborLoader(
    # data=train_data,
    # num_neighbors=[20, 10],
    # neg_sampling_ratio=2.0,
    # edge_label_index=(("user", "rates", "movie"), edge_label_index),
    # edge_label=edge_label,
    # batch_size=128,
    # shuffle=True,
    # )
    

    # print("Train Data")
    # print("-"*200)
    # print(train_data)
    # print("Val Data")
    # print("-"*200)
    # print(val_data)
    # print("Test Data")
    # print("-"*200)
    # print(test_data)


    device=  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # Define the model
    model = SupervisedHeteroSAGEModel(test_data,
                            hidden_channels=128, 
                            out_channels=1, 
                            num_layers=1,
                            device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader_dict=get_node_dataloaders(train_data,shuffle=True)
    val_loader_dict=get_node_dataloaders(val_data,shuffle=False)
    test_loader_dict=get_node_dataloaders(test_data,shuffle=False)



    batch=next(iter(test_loader_dict['material']))
    batch.to(device)
    z_dict=model(batch)
    # print(batch)
    # print(z_dict)   

    loss_fn=nn.CrossEntropyLoss()

    # batch=next(iter(test_loader_dict['element']))
    # print(batch)

    # batch=next(iter(test_loader_dict['chemenv']))
    # print(batch)



    # print(batch['material'].node_id)
    # print(model(batch))
    # Train and evaluate model
    # for epoch in range(1):
        # train_loss = train(model, optimizer, train_loader_dict, loss_fn=loss_fn)
        # test_loss = train(model, optimizer, test_loader_dict, loss_fn=loss_fn)
    #     val_loss = evaluate(model, val_loader_dict)

    #     print(f'Epoch {epoch+1}: Train Loss: {train_loss}, Validation Loss: {val_loss}')


    # Evaluate on test data
    # test_loss = evaluate(model, test_data)
    # print(f'Test Loss: {test_loss}')










    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()

    #     #
    #     x_dict = model(data)
    #     loss = negative_sampling_hetero_loss(x_dict, data.edge_index_dict)
    #     loss.backward()
    #     optimizer.step()

    # for epoch in range(1, 6):
    #     total_loss = total_examples = 0
    #     for sampled_data in tqdm.tqdm(train_loader):
    #         optimizer.zero_grad()
    #         sampled_data.to(device)
    #         pred = model(sampled_data)
    #         ground_truth = sampled_data["user", "rates", "movie"].edge_label
    #         loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += float(loss) * pred.numel()
    #         total_examples += pred.numel()
    #     print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
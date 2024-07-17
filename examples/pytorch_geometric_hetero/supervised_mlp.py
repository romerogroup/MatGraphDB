# Creating a GraphSAGE model

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import  SAGEConv, to_hetero


from matgraphdb.mlcore.datasets import MaterialGraphDataset
from matgraphdb.mlcore.metrics import ClassificationMetrics,RegressionMetrics


def get_node_dataloaders(data,node_type,shuffle=False):
    data[node_type].node_id
    input_nodes=('material',data[node_type].node_id)
    loader = NeighborLoader(
                graph_dataset.data,
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
    return loader


def split_data_on_node_type(data,target_node_type,train_proportion=0.8,test_proportion=0.1, val_proportion=0.1):
    assert train_proportion + test_proportion + val_proportion == 1.0
    for node_type in data.node_types:
        train_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        test_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        val_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)

        num_nodes_for_type=data[node_type].num_nodes
        if node_type==target_node_type:
            # Determine indices for training, testing, and validation
            indices = torch.randperm(num_nodes_for_type)

            num_train = int(train_proportion * num_nodes_for_type)
            num_val = int(test_proportion * num_nodes_for_type)
            num_test = num_nodes_for_type - num_train - num_val

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train + num_val]] = True
            test_mask[indices[num_train + num_val:]] = True
        else:
            train_mask[:num_nodes_for_type]=True

        data[node_type].train_mask=train_mask
        data[node_type].test_mask=test_mask
        data[node_type].val_mask=val_mask
    return data


NODE_TYPE='material'
# 
# TARGET_PROPERTY='energy_above_hull'
# TARGET_PROPERTY='formation_energy_per_atom'
# TARGET_PROPERTY='energy_per_atom'
# TARGET_PROPERTY='band_gap'
TARGET_PROPERTY='k_vrh'
# TARGET_PROPERTY='density'
# TARGET_PROPERTY='density_atomic'

# TARGET_PROPERTY='crystal_system'
# TARGET_PROPERTY='point_group'
# TARGET_PROPERTY='nelements'
# TARGET_PROPERTY='elements'

# CONNECTION_TYPE='GEOMETRIC_ELECTRIC_CONNECTS'
# CONNECTION_TYPE='GEOMETRIC_CONNECTS'
CONNECTION_TYPE='ELECTRIC_CONNECTS'

# Training params
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1
VAL_PROPORTION = 0.1
LEARNING_RATE = 0.001
N_EPCOHS = 1000

# model params
NUM_LAYERS = 3
HIDDEN_CHANNELS = 32
EVAL_INTERVAL = 10

USE_EARLY_STOPPING=False
EARLY_STOPPING_PATIENCE = 100


USE_LEARNING_RATE_SCHEDULER=False
STEP_SIZE=500
GAMMA=0.1


node_filtering={
    'material':{
        'k_vrh':(0,200),
        },
    }


node_properties={
'element':
    {
    'properties' :[
            'atomic_number',
            'group',
            'row',
            'atomic_mass'
            ],
    'scale': {
            # 'robust_scale': True,
            # 'standardize': True,
            # 'normalize': True
        }
    },
'material':
        {   
    'properties':[
        # 'composition',
        # 'space_group',
        # 'nelements',
        # 'nsites',
        # 'crystal_system',
        # 'band_gap',
        # 'formation_energy_per_atom',
        # 'energy_per_atom',
        # 'is_stable',
        # 'volume',
        # 'density',
        # 'density_atomic',

        # 'sine_coulomb_matrix',
        # 'element_fraction',
        'element_property',
        # 'xrd_pattern',
        ],
    'scale': {
            # 'robust_scale': True,
            # 'standardize': True,
            # 'normalize': True
        }
        }
    }

edge_properties={
    'weight':
        {
        'properties':[
            'weight'
            ],
        'scale': {
            # 'robust_scale': True,
            # 'standardize': True,
            'normalize': True
        }
    }
    }


if CONNECTION_TYPE=='GEOMETRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.gc_element_chemenv(
                                            node_properties=node_properties,
                                            node_filtering=node_filtering,
                                            edge_properties=edge_properties,
                                            node_target_property=TARGET_PROPERTY,
                                            edge_target_property=None,
                                            )
elif CONNECTION_TYPE=='ELECTRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                            node_properties=node_properties,
                                            node_filtering=node_filtering,
                                            edge_properties=edge_properties,
                                            node_target_property=TARGET_PROPERTY,
                                            edge_target_property=None,
                                            )
elif CONNECTION_TYPE=='GEOMETRIC_ELECTRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.gec_element_chemenv(
                                            node_properties=node_properties,
                                            node_filtering=node_filtering,
                                            edge_properties=edge_properties,
                                            node_target_property=TARGET_PROPERTY,
                                            edge_target_property=None,
                                            )

data=graph_dataset.data
OUT_CHANNELS=data[NODE_TYPE].out_channels
IN_CHANNELS=data[NODE_TYPE].x.shape[1]
print(f"IN_CHANNELS: {IN_CHANNELS}")
print(f"OUT_CHANNELS: {OUT_CHANNELS}")
print(f"Num_layers: {NUM_LAYERS}")
print(f"Hidden_channels: {HIDDEN_CHANNELS}")    

device =  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
data=split_data_on_node_type(data,
                            target_node_type=NODE_TYPE,
                            train_proportion=TRAIN_PROPORTION,
                            test_proportion=TEST_PROPORTION,
                            val_proportion=VAL_PROPORTION)
from torch.utils.data import Dataset, DataLoader
class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # Generating some dummy data: pairs of (input, label)
        self.data = data  # size examples, 10 features each
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the total number of data samples
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
train_mask= data['material'].train_mask
test_mask= data['material'].test_mask
val_mask= data['material'].val_mask

train_data = data['material'].x[train_mask]
test_data = data['material'].x[test_mask]
val_data = data['material'].x[val_mask]

train_labels = data['material'].y[train_mask]
test_labels = data['material'].y[test_mask]
val_labels = data['material'].y[val_mask]


train_dataset = SimpleDataset(train_data, train_labels)
test_dataset = SimpleDataset(test_data, test_labels)
val_dataset = SimpleDataset(val_data, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        dropout=0.2
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )
        self.ln=nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.net(self.ln(x))
        # return self.net(x)

class InputLayer(nn.Module):
    def __init__(self,input_dim,n_embd):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj=nn.Linear(input_dim, n_embd)

    def forward(self, x):
        out=self.flatten(x)
        return self.proj(out)

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, n_embd):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        
        self.input_layer=InputLayer(input_dim,n_embd)
        self.layers = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_layers)])

        self.ln_f=nn.LayerNorm(n_embd)
        self.output_layer=nn.Linear(n_embd,self.output_dim)
        
    
    def forward(self, x):
        out=self.input_layer(x)
        for layer in self.layers:
            out = out + layer(out)
        out=self.ln_f(out)
        out=self.output_layer(out)
        return out
    

model = MultiLayerPerceptron(input_dim=IN_CHANNELS,
                                output_dim=OUT_CHANNELS,
                                num_layers=NUM_LAYERS,
                                n_embd=HIDDEN_CHANNELS)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if OUT_CHANNELS==1:
    loss_fn=nn.MSELoss()
else:
    loss_fn=nn.CrossEntropyLoss()



def train_step(model,dataloader,device,optimizer,loss_fn):
        """
        Trains the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to train on.

        Returns:
            float: The average loss per batch on the training data.
        """
        model.train()
        num_batches = len(dataloader)
        batch_train_loss = 0.0
        for i_batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)

            train_loss = loss_fn(logits, y)
            batch_train_loss += train_loss.item()

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
        batch_train_loss = batch_train_loss / num_batches
        return batch_train_loss

def test_step(model,dataloader):
        """
        Tests the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to test on.

        Returns:
            float: The average loss per batch on the test data.
        """
        num_batches = len(dataloader)
        model.eval()
        batch_test_loss = 0.0
        with torch.no_grad():
            for i_batch,(X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                logits = model(X)
                batch_test_loss += loss_fn(logits, y)
                   
        batch_test_loss /= num_batches
        return batch_test_loss




for epoch in range(N_EPCOHS):
    train_loss = train_step(model, train_dataloader, device, optimizer,loss_fn=loss_fn)
    test_loss = test_step(model, test_dataloader)

    print(f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    # if epoch%EVAL_INTERVAL==0:
    #     losses,metrics = evaluate(model, data)

    #     # print(f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Val Loss: {losses[0]:.4f}, Test Loss: {losses[1]:.4f}")

    #     metrics_str=""
    #     metrics_str+=f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {losses[1]:.4f}"
    #     for split,metrics_dict in metrics.items():
    #         metrics_str+=" | "
    #         for i,(key,value) in enumerate(metrics_dict.items()):
    #             if i==0:
    #                 metrics_str+=f" {split}-{key}: {value[0]:.2f}"
    #             else:
    #                 metrics_str+=f", {split}-{key}: {value[0]:.2f}"
    #     print(metrics_str)
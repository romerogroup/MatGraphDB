import os
import copy
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error
from torch.nn import functional as F

from matgraphdb.pyg_dataset import PolyhedraDataset
from trash.callbacks import EarlyStopping

large_width = 400
np.set_printoptions(linewidth=large_width)

class PolyhedronModel(nn.Module):
    """This is the main Polyhedron Model. 

    Parameters
    ----------
        n_node_features : int
        The number of node features
    n_edge_features : int
        The number of edge features, by default 2
    n_gc_layers : int, optional
        The number of graph convolution layers, by default 1
    global_pooling_method : str, optional
        The global pooling method to be used, by default 'add'
    """

    def __init__(self,
                n_node_features:int,
                n_edge_features:int, 
                n_gc_layers:int=1, 
                n_hidden_layers:List[int]=[5],
                global_pooling_method:str='add'):
        """This is the main Polyhedron Model. 

        Parameters
        ----------
         n_node_features : int
            The number of node features
        n_edge_features : int
            The number of edge features, by default 2
        n_gc_layers : int, optional
            The number of graph convolution layers, by default 1
        global_pooling_method : str, optional
            The global pooling method to be used, by default 'add'
        """
        super().__init__()

            
        layers=[]
        for i_gc_layer in range(n_gc_layers):
            if i_gc_layer == 0:
                vals = " x, edge_index, edge_attr -> x0 "
            else:
                vals = " x" + repr(i_gc_layer - 1) + " , edge_index, edge_attr -> x" + repr(i_gc_layer)

            layers.append((pyg_nn.CGConv(n_node_features, dim=n_edge_features,aggr = 'add'),vals))

        self.relu = nn.ReLU()
        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , layers)

        

        self.out_layer= nn.Linear( n_node_features,  1)
        

        if global_pooling_method == 'add':
            self.global_pooling_layer = global_add_pool
        elif global_pooling_method == 'mean':
            self.global_pooling_layer = global_mean_pool
        elif global_pooling_method == 'max':
            self.global_pooling_layer = global_max_pool


    def forward(self, data_batch, targets=None):
        """The forward pass of of the network

        Parameters
        ----------
        x : pygeometic.Data
            The pygeometric data object
        targets : float, optional
            The target value to use to calculate the loss, by default None

        Returns
        -------
        _type_
            _description_
        """
        
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)
        # out = self.relu(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch) # out -> (n_graphs, n_hidden_layers[0])
        out = self.out_layer(out) # out -> (n_graphs, 1)

        # Loss handling
        if targets is None:
            loss = None
            mape_loss = None
        else:
            loss_fn = torch.nn.MSELoss()
            mape_loss = mean_absolute_percentage_error(torch.squeeze(out, dim=1), targets)
            loss = loss_fn(torch.squeeze(out, dim=1), targets)

        return out,  loss, mape_loss
    
    def encode(self, data_batch):
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)
        out = self.relu(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)
        return out

def weight_init(m):
    """
    Initializes the weights of a module using Xavier initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


###################################################################
# Parameters
###################################################################

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# hyperparameters

# Training params
n_epochs = 200
early_stopping_patience = 5
learning_rate = 1e-2
batch_size = 20
single_batch = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# polyhedron model parameters
n_gc_layers = 2
n_hidden_layers=[8,8]
global_pooling_method = 'add'

###################################################################
# Start of the the training run
###################################################################

train_dir = f"{project_dir}{os.sep}datasets{os.sep}three_body_energy{os.sep}material_polyhedra{os.sep}face_nodes{os.sep}train"
test_dir = f"{project_dir}{os.sep}datasets{os.sep}three_body_energy{os.sep}material_polyhedra{os.sep}face_nodes{os.sep}test"

train_dataset = PolyhedraDataset(database_dir=train_dir, device=device, y_val='y')
n_node_features = train_dataset[0].x.shape[1]
n_edge_features = train_dataset[0].edge_attr.shape[1]



node_max = torch.zeros(n_node_features, device = device)
node_min = torch.zeros(n_node_features, device = device)
edge_max = torch.zeros(n_edge_features, device = device)
edge_min = torch.zeros(n_edge_features, device = device)
for data in train_dataset:

    # Finding node max and min
    current_node_max = data.x.max(axis=0)[0]
    node_max = torch.vstack( [node_max,current_node_max] )
    node_max = node_max.max(axis=0)[0]

    current_node_min = data.x.min(axis=0)[0]
    node_min = torch.vstack( [node_min,current_node_min] )
    node_min = node_min.min(axis=0)[0]

    # Finding edge max and min
    current_edge_max = data.edge_attr.max(axis=0)[0]
    edge_max = torch.vstack( [edge_max,current_edge_max] )
    edge_max = edge_max.max(axis=0)[0]

    current_edge_min = data.edge_attr.min(axis=0)[0]
    edge_min = torch.vstack( [edge_min,current_edge_min] )
    edge_min = edge_min.min(axis=0)[0]

del train_dataset

def min_max_scaler(data):
    data.x = ( data.x - node_min ) / (node_min - node_max)
    data.edge_attr = ( data.edge_attr - edge_min ) / (edge_min - edge_max)

    data.edge_attr=data.edge_attr.nan_to_num()
    data.x=data.x.nan_to_num()
    return data

train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, y_val='y', transform = min_max_scaler)
test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, y_val='y', transform = min_max_scaler)

y_train_vals = []
n_graphs = len(train_dataset)
for data in train_dataset:
    y_train_vals.append(data.y)

y_train_vals = torch.tensor(y_train_vals).to(device)
avg_y_val = torch.mean(y_train_vals, axis=0)
std_y_val = torch.std(y_train_vals, axis=0)
print(f"Train average y_val: {avg_y_val}")
print(f"Train std y_val: {std_y_val}")

y_test_vals = []
n_graphs = len(test_dataset)
for data in test_dataset:
    y_test_vals.append(data.y)

y_test_vals = torch.tensor(y_test_vals).to(device)
avg_y_val = torch.mean(y_test_vals, axis=0)
std_y_val = torch.std(y_test_vals, axis=0)
print(f"Train average y_val: {avg_y_val}")
print(f"Train std y_val: {std_y_val}")


n_train = len(train_dataset)
n_test = len(test_dataset)


# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)


model = PolyhedronModel(n_node_features=n_node_features, 
                                n_edge_features=n_edge_features, 
                                n_gc_layers=n_gc_layers,
                                n_hidden_layers=n_hidden_layers,
                                global_pooling_method=global_pooling_method)

model.apply(weight_init)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
es = EarlyStopping(patience = early_stopping_patience)


###################################################################
# Train and Test epoch steps
###################################################################

def train(single_batch:bool=False):
    batch_train_loss = 0.0
    batch_train_mape = 0.0
    for i,sample in enumerate(train_loader):
        if single_batch:
            if i == 0:
                optimizer.zero_grad()
                out, train_loss, mape_loss = model(sample , targets = sample.y)
                train_loss.backward()
                optimizer.step()
                batch_train_loss += train_loss.item()
                batch_train_mape += mape_loss.item()

        else:
            optimizer.zero_grad()
            out, train_loss, mape_loss = model(sample , targets = sample.y)
            train_loss.backward()
            optimizer.step()
            batch_train_loss += train_loss.item()
            batch_train_mape += mape_loss.item()
            batch_train_loss = batch_train_loss / (i+1)
            batch_train_mape = batch_train_mape / (i+1)


    
    return batch_train_loss, batch_train_mape

@torch.no_grad()
def test():
    batch_test_loss = 0.0
    batch_test_mape = 0.0
    for i,sample in enumerate(test_loader):
        out, test_loss, mape_test_loss = model(sample , targets = sample.y)
        batch_test_loss += test_loss.item()
        batch_test_mape += mape_test_loss.item()
    batch_test_loss = batch_test_loss / (i+1)
    batch_test_mape = batch_test_mape / (i+1)

    return batch_test_loss, batch_test_mape


###################################################################
# Training Loop
###################################################################
n_epoch_0 = 0
model.train()
for epoch in range(n_epochs):
    n_epoch = n_epoch_0 + epoch

    batch_train_loss, batch_train_mape = train(single_batch=single_batch)
    batch_test_loss, batch_test_mape = test()

    if es(model=model, val_loss=batch_test_loss,mape_val_loss=batch_test_mape):
        print("Early stopping")
        print('_______________________')
        print(f'Stopping : {epoch - es.counter}')
        print(f'mae_val : {es.best_loss**0.5}')
        print(f'mape_val : {es.best_mape_loss}')
        break
    if n_epoch % 1 == 0:
        print(f"{n_epoch},{batch_train_loss:.5f},{batch_test_loss:.5f}, {100*batch_test_mape:.3f}")

# Train average y_val: 102.41304016113281
# Train std y_val: 34.732421875
# Train average y_val: 85.29576873779297
# Train std y_val: 105.39012145996094

###############################################################################################
################################################################################################
def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x/np.linalg.norm(x) - y/np.linalg.norm(y))


def compare_polyhedra(loader, model):
    data = []
    expected_values = []
    columns = {
        'expected_value':[],
        'prediction_value':[],
        'percent_error':[],
        'label':[],
        'n_nodes':[],
        }
    
    n_nodes = []
    for sample in loader:
        predictions = model(sample)
        for real, pred, encoding,pos in zip(sample.y,predictions[0],model.encode(sample),sample.node_stores[0]['pos']):
        #     print('______________________________________________________')
            print(f"Prediction : {pred.item()} | Expected : {real.item()} | Percent error : { 100*abs(real.item() - pred.item()) / real.item() }")
            columns['prediction_value'].append(round(pred.item(),3))
            columns['expected_value'].append(round(real.item(),3))
            columns['percent_error'].append(round(100* abs(real.item() - pred.item()) / real.item(),3))
            columns['label'].append(sample.label[0])
            columns['n_nodes'].append(sample.num_nodes)
            
            # print(f"Encodings : {encoding.tolist()}")
            n_node = len(pos)
            n_nodes.append(n_node)
            expected_values.append(real.item())
            data.append(np.array(encoding.tolist()) )


    print(n_nodes)
    print(expected_values)
    n_nodes_before_sort = np.array(n_nodes)
    data = np.array(data)
    print(data)
    # polyhedra = [(data[0],'tetra'),(data[1],'cube'),(data[2],'oct'),(data[3],'dod'),(data[4],'rotated_tetra'),
    #             (data[5],'verts_mp567387_Ti_dod'),(data[6],'verts_mp4019_Ti_cube'),(data[7],'verts_mp3397_Ti_tetra'),
    #             (data[8],'verts_mp15502_Ba_cube'),(data[9],'verts_mp15502_Ti_oct')]
    
    polyhedra = [(data[0],'tetra'),(data[2],'cube'),(data[4],'oct'),(data[6],'dod'),(data[8],'rotated_tetra'),
                (data[9],'verts_mp567387_Ti_dod'),(data[1],'verts_mp4019_Ti_cube'),(data[3],'verts_mp3397_Ti_tetra'),
                (data[5],'verts_mp15502_Ba_cube'),(data[7],'verts_mp15502_Ti_oct')]
    
    polyhedra = [(data[0],'tetra',n_nodes[0]),(data[2],'cube',n_nodes[2]),(data[4],'oct',n_nodes[4]),
                 (data[6],'dod',n_nodes[6]),(data[8],'rotated_tetra',n_nodes[8]),(data[9],'dod-like',n_nodes[9]),
                 (data[1],'cube-like',n_nodes[1]),(data[3],'tetra-like',n_nodes[3]),
                (data[5],'cube-like',n_nodes[5]),(data[7],'oct-like',n_nodes[7])]
    poly_names= []
    distance_similarity_mat = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    cosine_similarity_mat = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    for i,poly_a in enumerate(polyhedra):
        for j,poly_b in enumerate(polyhedra):
            # print('_______________________________________')
            # print(f'Poly_a - {poly_a[1]} | Poly_b - {poly_b[1]}')
            # print(f'Cosine : {cosine_similarity(x=poly_a[0],y=poly_b[0])}')
            # print(f'Distance : {distance_similarity(x=poly_a[0],y=poly_b[0])}')
            distance_similarity_mat[i,j] = distance_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            cosine_similarity_mat[i,j] = cosine_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            

    print(polyhedra[0][1],polyhedra[1][1],polyhedra[2][1],polyhedra[3][1],polyhedra[4][1],
        polyhedra[5][1],polyhedra[6][1],polyhedra[7][1],polyhedra[8][1],polyhedra[9][1],)
    
    # print(distance_similarty,decimals=3)
    # for x in distance_similarty:
    #     print(x)
    print("--------------------------")
    print("Distance Similarity Matrix")
    print("--------------------------")
    print(distance_similarity_mat)
    print("--------------------------")
    print("Cosine Similarity Matrix")
    print("--------------------------")
    print(cosine_similarity_mat)

    n_nodes = [poly[2] for poly in polyhedra]
    names = [poly[1] for poly in polyhedra]
    encodings = [poly[0] for poly in polyhedra]
    df = pd.DataFrame(encodings, index = names)
    df['n_nodes']  = n_nodes
    df.to_csv(f'{project_dir}{os.sep}reports{os.sep}encodings.csv')

    df = pd.DataFrame(cosine_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{project_dir}{os.sep}reports{os.sep}cosine_similarity.csv')

    df = pd.DataFrame(distance_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{project_dir}{os.sep}reports{os.sep}distance_similarity.csv')

    df = pd.DataFrame(columns)
    # df['n_nodes']  = n_nodes_before_sort
    df.to_csv(f'{project_dir}{os.sep}reports{os.sep}energy_test.csv')
    return None

compare_polyhedra(loader=test_loader, model=model)
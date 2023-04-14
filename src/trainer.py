import os
import copy
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error
from torch.nn import functional as F

from poly_graphs_lib.pyg_json_dataset import PolyhedraDataset
from poly_graphs_lib.callbacks import EarlyStopping
from poly_graphs_lib.poly_regression_model import PolyhedronModel
from poly_graphs_lib.poly_residual_regression_model import PolyhedronResidualModel


large_width = 400
np.set_printoptions(linewidth=large_width)

def weight_init(m):
    """
    Initializes the weights of a module using Xavier initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


###################################################################
# Parameters
###################################################################

torch.manual_seed(0)

project_dir = os.path.dirname(os.path.dirname(__file__))
print(project_dir)
# hyperparameters

# Training params
n_epochs = 200
early_stopping_patience = 5
learning_rate = 1e-3
batch_size = 20
single_batch = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model parameters
n_gc_layers = 4
layers_1 = [4*9]
layers_2 = [4*9]
dropout= None
apply_layer_norms=True
global_pooling_method = 'mean'


# data parameters
feature_set = 'face_feature_set_2'
train_dir = f"{project_dir}{os.sep}datasets{os.sep}processed{os.sep}three_body_energy{os.sep}train"
test_dir = f"{project_dir}{os.sep}datasets{os.sep}processed{os.sep}three_body_energy{os.sep}test"
reports_dir = f"{project_dir}{os.sep}reports{os.sep}{feature_set}{os.sep}test_no_dropout"


os.makedirs(reports_dir,exist_ok=True)

###################################################################
# Start of the the training run
###################################################################


###################################################################
# Initializing Model
###################################################################


train_dataset = PolyhedraDataset(database_dir=train_dir, device=device, feature_set=feature_set)
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

# train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, feature_set=feature_set, transform = min_max_scaler)
# test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, feature_set=feature_set, transform = min_max_scaler)

train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, feature_set=feature_set)#, transform = min_max_scaler)
test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, feature_set=feature_set)#, transform = min_max_scaler)

y_train_vals = []
n_graphs = len(train_dataset)
for data in train_dataset:
    y_train_vals.append(data.y)

y_train_vals = torch.tensor(y_train_vals).to(device)
avg_y_val_train = torch.mean(y_train_vals, axis=0)
std_y_val = torch.std(y_train_vals, axis=0)
print(f"Train average y_val: {avg_y_val_train}")
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

###################################################################
# Initializing Model
###################################################################

# model = PolyhedronModel(n_node_features=n_node_features, 
#                                 n_edge_features=n_edge_features, 
#                                 n_gc_layers=n_gc_layers,
#                                 layers_1=layers_1,
#                                 layers_2=layers_2,
#                                 global_pooling_method=global_pooling_method)

model = PolyhedronResidualModel(n_node_features=n_node_features, 
                                n_edge_features=n_edge_features, 
                                n_gc_layers=n_gc_layers,
                                layers_1=layers_1,
                                layers_2=layers_2,
                                dropout=dropout,
                                apply_layer_norms=apply_layer_norms,
                                global_pooling_method=global_pooling_method,
                                target_mean=avg_y_val_train)
print(str(model))
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: " + str(pytorch_total_params))

print("Number of training samples: " + str(n_train) )
print("Number of training samples: " + str(n_test) )
# model.apply(weight_init)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
es = EarlyStopping(patience = early_stopping_patience)
es = None
metrics_dict = {
                "train_mse":[],
                "test_mse":[],
                "train_mape":[],
                "test_mape":[],
                "trainable_params":pytorch_total_params,
                "model":str(model)
                }

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
    model.eval()
    batch_test_loss = 0.0
    batch_test_mape = 0.0
    for i,sample in enumerate(test_loader):
        out, test_loss, mape_test_loss = model(sample , targets = sample.y)
        batch_test_loss += test_loss.item()
        batch_test_mape += mape_test_loss.item()
    batch_test_loss = batch_test_loss / (i+1)
    batch_test_mape = batch_test_mape / (i+1)
    model.train()
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

    metrics_dict['train_mse'].append(batch_train_loss)
    metrics_dict['test_mse'].append(batch_test_loss)

    metrics_dict['train_mape'].append(batch_train_mape)
    metrics_dict['test_mape'].append(batch_test_mape)

    if es is not None:
        if es(model=model, val_loss=batch_test_loss,mape_val_loss=batch_test_mape):
            print("Early stopping")
            print('_______________________')
            print(f'Stopping : {epoch - es.counter}')
            print(f'mae_val : {es.best_loss**0.5}')
            print(f'mape_val : {es.best_mape_loss}')
            break

    if n_epoch % 1 == 0:
        print(f"{n_epoch}, {batch_train_loss:.5f}, {batch_test_loss:.5f}, {100*batch_test_mape:.3f}")

with open(f'{reports_dir}{os.sep}metrics.json','w') as outfile:
    json.dump(metrics_dict, outfile,indent=4)

# batch = next(iter(train_loader))

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
    expected_values = []
    columns = {
        'expected_value':[],
        'prediction_value':[],
        'percent_error':[],
        'label':[],
        'n_nodes':[],
        }
    polyhedra_encodings = []
    n_nodes = []
    model.eval()
    for sample in loader:
        predictions = model(sample)
        print(sample.label)
        # print(sample.x)
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
            polyhedra_encodings.append((np.array(encoding.tolist()),sample.label[0] , sample.num_nodes ))

    distance_similarity_mat = np.zeros(shape = (len(polyhedra_encodings),len(polyhedra_encodings)))
    cosine_similarity_mat = np.zeros(shape = (len(polyhedra_encodings),len(polyhedra_encodings)))
    for i,poly_a in enumerate(polyhedra_encodings):
        for j,poly_b in enumerate(polyhedra_encodings):
            # print('_______________________________________')
            # print(f'Poly_a - {poly_a[1]} | Poly_b - {poly_b[1]}')
            # print(f'Cosine : {cosine_similarity(x=poly_a[0],y=poly_b[0])}')
            # print(f'Distance : {distance_similarity(x=poly_a[0],y=poly_b[0])}')
            distance_similarity_mat[i,j] = distance_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            cosine_similarity_mat[i,j] = cosine_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            
    print('________________________________________________________________')
    print(polyhedra_encodings[0][1],polyhedra_encodings[1][1],polyhedra_encodings[2][1],polyhedra_encodings[3][1],polyhedra_encodings[4][1],
        polyhedra_encodings[5][1],polyhedra_encodings[6][1],polyhedra_encodings[7][1],polyhedra_encodings[8][1])

    print("--------------------------")
    print("Distance Similarity Matrix")
    print("--------------------------")
    print(distance_similarity_mat)
    print("--------------------------")
    print("Cosine Similarity Matrix")
    print("--------------------------")
    print(cosine_similarity_mat)

    n_nodes = [poly[2] for poly in polyhedra_encodings]
    names = [poly[1] for poly in polyhedra_encodings]
    encodings = [poly[0] for poly in polyhedra_encodings]
    df = pd.DataFrame(encodings, index = names)
    df['n_nodes']  = n_nodes
    df.to_csv(f'{reports_dir}{os.sep}encodings.csv')

    df = pd.DataFrame(cosine_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{reports_dir}{os.sep}cosine_similarity.csv')

    df = pd.DataFrame(distance_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{reports_dir}{os.sep}distance_similarity.csv')

    df = pd.DataFrame(columns)
    # df['n_nodes']  = n_nodes_before_sort
    df.to_csv(f'{reports_dir}{os.sep}energy_test.csv')
    return None

compare_polyhedra(loader=test_loader, model=model)
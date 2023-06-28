import os
import copy
from typing import List
import json
import yaml
import pickle
from datetime import datetime
import time

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

from poly_graphs_lib.models.poly.pyg_json_dataset import PolyhedraDataset
from poly_graphs_lib.callbacks import EarlyStopping
from poly_graphs_lib.models.poly.poly_regression_model import PolyhedronModel
from poly_graphs_lib.models.poly.poly_residual_regression_model import PolyhedronResidualModel


large_width = 400
np.set_printoptions(linewidth=large_width)

###################################################################
# Import configurations
###################################################################

torch.manual_seed(0)

project_dir = os.path.dirname(os.path.dirname(__file__))
print(project_dir)
# hyperparameters

input_file = os.path.join(project_dir,'sample.yml')
with open( input_file, 'r' ) as input_stream:
    SETTINGS = yaml.load(input_stream, Loader=yaml.Loader)


###################################################################
# Initializing Model
###################################################################


train_dataset = PolyhedraDataset(database_dir=SETTINGS['train_dir'],device=SETTINGS['device'], feature_set=SETTINGS['feature_set'])
n_node_features = train_dataset[0].x.shape[1]

n_edge_features = train_dataset[0].edge_attr.shape[1]



# node_max = torch.zeros(n_node_features, device = device)
# node_min = torch.zeros(n_node_features, device = device)
# edge_max = torch.zeros(n_edge_features, device = device)
# edge_min = torch.zeros(n_edge_features, device = device)
# for i,data in enumerate(train_dataset):
#     data.to(device)
#     # Finding node max and min
#     current_node_max = data.x.max(axis=0)[0]
#     node_max = torch.vstack( [node_max,current_node_max] )
#     node_max = node_max.max(axis=0)[0]

#     current_node_min = data.x.min(axis=0)[0]
#     node_min = torch.vstack( [node_min,current_node_min] )
#     node_min = node_min.min(axis=0)[0]

#     # Finding edge max and min
#     current_edge_max = data.edge_attr.max(axis=0)[0]
#     edge_max = torch.vstack( [edge_max,current_edge_max] )
#     edge_max = edge_max.max(axis=0)[0]

#     current_edge_min = data.edge_attr.min(axis=0)[0]
#     edge_min = torch.vstack( [edge_min,current_edge_min] )
#     edge_min = edge_min.min(axis=0)[0]

# del train_dataset

# def min_max_scaler(data):
#     data.x = ( data.x - node_min ) / (node_min - node_max)
#     data.edge_attr = ( data.edge_attr - edge_min ) / (edge_min - edge_max)

#     data.edge_attr=data.edge_attr.nan_to_num()
#     data.x=data.x.nan_to_num()
#     return data

# train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, feature_set=feature_set, transform = min_max_scaler)
# test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, feature_set=feature_set, transform = min_max_scaler)

train_dataset = PolyhedraDataset(database_dir=SETTINGS['train_dir'],
                                 device=SETTINGS['device'], 
                                 n_max_entries=SETTINGS['n_train_max_entries'],
                                 feature_set=SETTINGS['feature_set'])
test_dataset = PolyhedraDataset(database_dir=SETTINGS['test_dir'],
                                device=SETTINGS['device'], 
                                n_max_entries=SETTINGS['n_test_max_entries'],
                                feature_set=SETTINGS['feature_set'])
n_train = len(train_dataset)
n_test = len(test_dataset)

print("Using cuda : ", torch.cuda.is_available())
print("Number of gpus available : ", torch.cuda.device_count())
print("Number of Training samples: " + str(n_train))
print("Number of Test samples: " + str(n_test))
print("Number of node features : ", n_node_features)
print("Number of edge features : ", n_edge_features)


# y_train_vals = []
# n_graphs = len(train_dataset)
# for data in train_dataset:
#     data.to(SETTINGS['device'])
#     y_train_vals.append(data.y)

# y_train_vals = torch.tensor(y_train_vals).to(SETTINGS['device'])
# avg_y_val_train = torch.mean(y_train_vals, axis=0)
# std_y_val = torch.std(y_train_vals, axis=0)
# print(f"Train average y_val: {avg_y_val_train}")
# print(f"Train std y_val: {std_y_val}")

# y_test_vals = []
# n_graphs = len(test_dataset)
# for data in test_dataset:
#     data.to(SETTINGS['device'])
#     y_test_vals.append(data.y)

# y_test_vals = torch.tensor(y_test_vals).to(SETTINGS['device'])
# avg_y_val = torch.mean(y_test_vals, axis=0)
# std_y_val = torch.std(y_test_vals, axis=0)
# print(f"Test average y_val: {avg_y_val}")
# print(f"Test std y_val: {std_y_val}")

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=SETTINGS['batch_size'],shuffle=True)#, num_workers=SETTINGS['num_workers'])
test_loader = DataLoader(test_dataset, batch_size=SETTINGS['batch_size'],shuffle=False)#, num_workers=SETTINGS['num_workers'])
print(next(iter(train_loader)))
for batch in train_loader:
    print(batch)

for batch in test_loader:
    print(batch)
###################################################################
# Initializing Model
###################################################################

model = PolyhedronResidualModel(n_node_features=n_node_features, 
                                    n_edge_features=n_edge_features, 
                                    n_gc_layers=SETTINGS['n_gc_layers'],
                                    layers_1=SETTINGS['layers_1'],
                                    layers_2=SETTINGS['layers_2'],
                                    dropout=SETTINGS['dropout'],
                                    apply_layer_norms=SETTINGS['apply_layer_norms'],
                                    global_pooling_method=SETTINGS['global_pooling_method'],
                                    target_mean=None)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters: " + str(pytorch_total_params))


m = model.to(SETTINGS['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=float(SETTINGS['learning_rate']))
es = EarlyStopping(patience = SETTINGS['early_stopping_patience'])
loss_fn = torch.nn.MSELoss()


metrics_dict = {
                "epochs":np.arange(SETTINGS['epochs']),
                "train_mse":[],
                "test_mse":[],
                "trainable_params":pytorch_total_params,
                }

###################################################################
# Train and Test epoch steps
###################################################################

def train(single_batch:bool=False):
    batch_train_loss = 0.0
    for i,sample in enumerate(train_loader):
        sample.to(SETTINGS['device'])
        if single_batch:
            if i == 0:
                optimizer.zero_grad()
                targets = sample.y
                out = model(sample)

                train_loss = loss_fn(torch.squeeze(out, dim=1), targets)
                train_loss.backward()
                optimizer.step()
                batch_train_loss += train_loss.item()

        else:
            optimizer.zero_grad()
            targets = sample.y
            out = model(sample)
            
            train_loss = loss_fn(torch.squeeze(out, dim=1), targets)
            train_loss.backward()
            optimizer.step()
            batch_train_loss += train_loss.item()

    batch_train_loss = batch_train_loss / (i+1)
    return batch_train_loss

@torch.no_grad()
def test():
    model.eval()
    batch_test_loss = 0.0
    for i,sample in enumerate(test_loader):
        sample.to(SETTINGS['device'])
        targets = sample.y
        out = model(sample)
        test_loss = loss_fn(torch.squeeze(out, dim=1), targets)
        batch_test_loss += test_loss.item()
    batch_test_loss = batch_test_loss / (i+1)
    model.train()
    return batch_test_loss


###################################################################
# Training Loop
###################################################################
start_time= time.time()
n_epoch_0 = 0
model.train()
for epoch in range(SETTINGS['epochs']):
    n_epoch = n_epoch_0 + epoch

    batch_train_loss = train(single_batch=SETTINGS['single_batch'])
    batch_test_loss= test()

    metrics_dict['train_mse'].append(batch_train_loss)
    metrics_dict['test_mse'].append(batch_test_loss)

    if es is not None:
        if es(model=model, val_loss=batch_test_loss):
            print("Early stopping")
            print('_______________________')
            print(f'Stopping : {epoch - es.counter}')
            print(f'mae_val : {es.best_loss**0.5}')
            break

    if n_epoch % 1 == 0:
        print(f"{n_epoch}, {batch_train_loss:.5f}, {batch_test_loss:.5f}")


###############################################################################################
################################################################################################
def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x/np.linalg.norm(x) - y/np.linalg.norm(y))


def compare_polyhedra(run_dir, loader, model):
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
        sample.to(SETTINGS['device'])
        predictions = model(sample)
        print(sample.label)
        # print(sample.x)
        for real, pred, encoding,pos in zip(sample.y,predictions[0],model.encode_2(sample),sample.node_stores[0]['pos']):
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
    poly_type_str = ''
    for i in range(len(polyhedra_encodings)):
        poly_type_str += polyhedra_encodings[i][1]
        if i != len(polyhedra_encodings) - 1:
            poly_type_str += ' , '

    print(poly_type_str)

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
    df.to_csv(f'{run_dir}{os.sep}encodings.csv')

    df = pd.DataFrame(cosine_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{run_dir}{os.sep}cosine_similarity.csv')

    df = pd.DataFrame(distance_similarity_mat, columns = names, index = names)
    df['n_nodes']  = n_nodes
    df.loc['n_nodes'] = np.append(n_nodes, np.array([0]),axis=0)
    df.to_csv(f'{run_dir}{os.sep}distance_similarity.csv')

    df = pd.DataFrame(columns)
    # df['n_nodes']  = n_nodes_before_sort
    df.to_csv(f'{run_dir}{os.sep}energy_test.csv')
    return None



# Saving model metrics and parameters

os.makedirs(SETTINGS['save_dir'], exist_ok=True)
for n in range(1, 9999):
    p = SETTINGS['save_dir'] + os.sep + f'train{n}'  # increment path
    if not os.path.exists(p):  #
        break

run_dir = p
weights_dir = run_dir + os.sep + 'weights'
os.makedirs(weights_dir)


with open(f'{run_dir}{os.sep}metrics.json','w') as outfile:
    json.dump(metrics_dict, outfile,indent=4)
compare_polyhedra(run_dir = run_dir,loader=test_loader, model=model)


args_file = os.path.join(run_dir,'sample.yml')
with open( args_file, 'w' ) as file:
    yaml.dump(SETTINGS, file)

txt_file = os.path.join(run_dir,'training.txt')
training_time = time.time() - start_time
with open(txt_file,'w') as file:
    file.write(f"Training time : {training_time:.2f}s")

metrics = pd.DataFrame(metrics_dict)

metrics_file = os.path.join(run_dir,'results.csv')
metrics.to_csv(metrics_file, index=False)



ckpt_last = {
            'epoch': SETTINGS['epochs'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_args': SETTINGS,  # save as dict
            'date': datetime.now().isoformat()}

torch.save(ckpt_last, weights_dir +os.sep+ f'last.pt', pickle_module=pickle)

ckpt_last = {
            'epoch': SETTINGS['epochs'],
            'model': es.best_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_args': SETTINGS,  # save as dict
            'date': datetime.now().isoformat()}

torch.save(ckpt_last, weights_dir +os.sep+ f'best.pt', pickle_module=pickle)


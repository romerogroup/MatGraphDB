import os
import copy
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import mlflow

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error
from torchmetrics import MeanAbsolutePercentageError

from poly_graphs_lib.poly_dataset import PolyhedraDataset
from poly_graphs_lib.poly_graph_model import PolyhedronModel

def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x/np.linalg.norm(x) -y/np.linalg.norm(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Training params
    batch_size = 2
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    feature_set_index = 3
    n_gc_layers=2
    y_val = 'energy_per_verts'
    global_pooling_method = 'add'
    dataset = 'material_random_polyhedra'

    load_checkpoint = project_dir + os.sep+ 'models' + os.sep + 'model_checkpoint_1000.pth'


    ######################################################################################
    # Testing model
    ######################################################################################
    test_dir = f"{project_dir}{os.sep}data{os.sep}{dataset}{os.sep}feature_set_{feature_set_index}{os.sep}test"

    val_dataset = PolyhedraDataset(database_dir=test_dir,device=device, y_val='moi')
    n_node_features = val_dataset[0].x.shape[1]
    n_edge_features = val_dataset[0].edge_attr.shape[1]
    del val_dataset

    test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, y_val=y_val)

    # Creating data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    loaded_checkpoint = torch.load(load_checkpoint)
        
    model = PolyhedronModel(n_node_features=n_node_features, 
                            n_edge_features=n_edge_features, 
                            n_gc_layers=n_gc_layers,
                            global_pooling_method=global_pooling_method)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
    n_epoch = loaded_checkpoint['epoch']
    train_loss = loaded_checkpoint['train_loss']
    val_loss = loaded_checkpoint['val_loss']
    print(repr(n_epoch) + ",  " + repr(train_loss) + ",  " + repr(val_loss))
    data = []
    for sample in test_loader:
        predictions = model(sample)
        # print(predictions)
        # print(predictions.shape)
        for real, pred, encoding in zip(sample.y,predictions[0],model.generate_encoding(sample)):
        #     print('______________________________________________________')
            print(f"Prediction : {pred.item()} | Expected : {real.item()} | Percent error : { abs(real.item() - pred.item()) / real.item() }")

            # print(f"Encodings : {encoding.tolist()}")
            data.append(np.array(encoding.tolist()))


    print(np.array(data))

    polyhedra = [(data[0],'tetra'),(data[1],'cube'),(data[2],'oct'),(data[3],'dod'),(data[4],'rotated_tetra'),
                (data[5],'verts_mp567387_Ti_dod'),(data[6],'verts_mp4019_Ti_cube'),(data[7],'verts_mp3397_Ti_tetra'),(data[8],'verts_mp15502_Ba_cube'),(data[9],'verts_mp15502_Ti_oct')]
    distance_similarty = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    cosine_similarty = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    for i,poly_a in enumerate(polyhedra):
        for j,poly_b in enumerate(polyhedra):
            # print('_______________________________________')
            # print(f'Poly_a - {poly_a[1]} | Poly_b - {poly_b[1]}')
            # print(f'Cosine : {cosine_similarity(x=poly_a[0],y=poly_b[0])}')
            # print(f'Distance : {distance_similarity(x=poly_a[0],y=poly_b[0])}')
            distance_similarty[i,j] = distance_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            cosine_similarty[i,j] = cosine_similarity(x=poly_a[0],y=poly_b[0]).round(3)
    print(polyhedra[0][1],polyhedra[1][1],polyhedra[2][1],polyhedra[3][1],polyhedra[4][1],
        polyhedra[5][1],polyhedra[6][1],polyhedra[7][1],polyhedra[8][1],polyhedra[9][1],)
    
    # print(distance_similarty,decimals=3)
    for x in distance_similarty:
        print(x)

    print(cosine_similarty)

    print(count_parameters(model=model))

if __name__ == '__main__':
    main()
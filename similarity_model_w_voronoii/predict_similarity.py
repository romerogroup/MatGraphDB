import os
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import numpy as np

from model_similarity import PolyhedronModel
from polyhedra_dataset import PolyhedraDataset

def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x-y)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# hyperparameters
# directories where data is
parent_dir = os.path.dirname(__file__)
train_dir = f"{parent_dir}{os.sep}train"
test_dir = f"{parent_dir}{os.sep}test"
val_dir = f"{parent_dir}{os.sep}val"



# Saving and Loading model settings
# model_name = 'model_checkpoint_4.pth'
# model_name = 'model_checkpoint_4_connected-energy.pth'
model_name = 'model_checkpoint_2.pth'
# ----------------------
# Saving from scratch
# load_checkpoint = None
# save_model = f"{parent_dir}{os.sep}{model_name}"
# ----------------------
# Loading model
load_checkpoint = f"{parent_dir}{os.sep}{model_name}"
save_model = None

# Training params
train_model = False
n_epochs = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Other params
batch_size = 5
dropout = 0.2

n_node_features = 2
n_edge_features = 2
# ------------

torch.manual_seed(1337)

# Loading directories as Datatsets
test_dataset = PolyhedraDataset(database_dir=test_dir,device=device)

n_test = len(test_dataset)

# Creating data loaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)



def main():

    # Either loading a model or starting from scratch
    if load_checkpoint:
        loaded_checkpoint = torch.load(load_checkpoint)
        
        model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features])
        m = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optim_state'])
        n_epoch = loaded_checkpoint['epoch']
        train_loss = loaded_checkpoint['train_loss']
        val_loss = loaded_checkpoint['val_loss']
        print(repr(n_epoch) + ",  " + repr(train_loss) + ",  " + repr(val_loss))
        data = []
        for sample in test_loader:
            print(model.generate_polyhedron_enocoding(sample[0]))
            print(model.generate_polyhedron_enocoding(sample[1]))
            predictions = model.predict(sample)
            for real, pred,encoding in zip(sample[2],predictions,model.generate_polyhedron_enocoding(sample[0])):
                print('______________________________________________________')
                print(f"Prediction : {pred.item()} | Expected : {real.item()}")

                print(f"Encodings : {encoding.tolist()}")
                data.append(np.array(encoding.tolist()))


        print(np.array(data))
        # tetra = np.array([1.3789399862289429, 0.0, 0.0, 0.0])
        # rotated_tetra = np.array([5.904356956481934, 0.45628172159194946, 0.0, 0.0])
        # dod = np.array([2.440209150314331, 0.0, 0.0, 0.0])
        # cube =  np.array([1.7790727615356445, 0.0, 0.0, 0.0])
        # oct =  np.array([0.5938437581062317, 0.0, 0.0, 0.0])

        # polyhedra = [(tetra,'tetra'),(cube,'cube'),(oct,'oct'),(dod,'dod'),(rotated_tetra,'rotated_tetra')]
        # polyhedra = [(data[0],'tetra'),(data[1],'cube'),(data[2],'oct'),(data[3],'dod'),(data[4],'rotated_tetra')]
        # distance_similarty = np.zeros(shape = (len(polyhedra),len(polyhedra)))
        # cosine_similarty = np.zeros(shape = (len(polyhedra),len(polyhedra)))
        # for i,poly_a in enumerate(polyhedra):
        #     for j,poly_b in enumerate(polyhedra):
        #         print('_______________________________________')
        #         print(f'Poly_a - {poly_a[1]} | Poly_b - {poly_b[1]}')
        #         print(f'Cosine : {cosine_similarity(x=poly_a[0],y=poly_b[0])}')
        #         print(f'Distance : {distance_similarity(x=poly_a[0],y=poly_b[0])}')
        #         distance_similarty[i,j] = distance_similarity(x=poly_a[0],y=poly_b[0]).round(3)
        #         cosine_similarty[i,j] = cosine_similarity(x=poly_a[0],y=poly_b[0]).round(3)
        # print(polyhedra[0][1],polyhedra[1][1],polyhedra[2][1],polyhedra[3][1],polyhedra[4][1])
        # print(distance_similarty)

        # print(cosine_similarty)

        print(count_parameters(model=model))


    else:
        model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features])
        m = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if __name__=='__main__':
    main()


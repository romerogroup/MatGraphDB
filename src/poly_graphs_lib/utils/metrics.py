import os

import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
from poly_graphs_lib.utils.plotting import plot_similarity_matrix,plot_training_curves
from poly_graphs_lib.utils.math import distance_similarity, cosine_similarity

def compare_polyhedra(run_dir, dataset, model, device):

    loader = DataLoader(dataset, batch_size=1,shuffle=False)
    expected_values = []
    columns = {
        'expected_value':[],
        'prediction_value':[],
        'percent_error':[],
        'label':[],
        }
    polyhedra_encodings = []
    n_nodes = []
    model.eval()
    for sample in loader:
        sample.to(device)
        predictions = model(sample)
        for real, pred, encoding in zip(sample.y,predictions[0],model.encode_2(sample)):
        #     print('______________________________________________________')
            print(f"Prediction : {pred.item()} | Expected : {real.item()} | Percent error : { 100*abs(real.item() - pred.item()) / real.item() }")
            columns['prediction_value'].append(round(pred.item(),3))
            columns['expected_value'].append(round(real.item(),3))
            columns['percent_error'].append(round(100* abs(real.item() - pred.item()) / real.item(),3))
            columns['label'].append(sample.label[0])
            expected_values.append(real.item())
            polyhedra_encodings.append((np.array(encoding.tolist()),sample.label[0] ))

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

    names = [poly[1] for poly in polyhedra_encodings]
    encodings = [poly[0] for poly in polyhedra_encodings]
    df = pd.DataFrame(encodings, index = names)

    df.to_csv(f'{run_dir}{os.sep}encodings.csv')

    df = pd.DataFrame(cosine_similarity_mat, columns = names, index = names)
    df.to_csv(f'{run_dir}{os.sep}cosine_similarity.csv')

    df = pd.DataFrame(distance_similarity_mat, columns = names, index = names)
    df.to_csv(f'{run_dir}{os.sep}distance_similarity.csv')

    df = pd.DataFrame(columns)
    df.to_csv(f'{run_dir}{os.sep}energy_test.csv')

    plot_similarity_matrix( similarity_matrix=distance_similarity_mat, labels=names, 
                           add_values=True, 
                           filename=os.path.join(run_dir,'distance_similarity.png'))
    plot_similarity_matrix( similarity_matrix=cosine_similarity_mat, labels=names, 
                           add_values=True, 
                           filename=os.path.join(run_dir,'cosine_similarity.png'))
    


def compare_polyhedra_old(run_dir, dataset, model, device):

    loader = DataLoader(dataset, batch_size=1,shuffle=False)
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
        sample.to(device)
        predictions = model(sample)
        # print(sample.label)
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

    plot_similarity_matrix( similarity_matrix=distance_similarity_mat, labels=names, 
                           add_values=True, 
                           filename=os.path.join(run_dir,'distance_similarity.png'))
    plot_similarity_matrix( similarity_matrix=cosine_similarity_mat, labels=names, 
                           add_values=True, 
                           filename=os.path.join(run_dir,'cosine_similarity.png'))
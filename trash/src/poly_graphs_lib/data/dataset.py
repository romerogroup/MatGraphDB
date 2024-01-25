import json
import os
import shutil
import logging
from glob import glob
import multiprocessing as mp

import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from .featurization import PolyFeaturizer
import torch


# Create a logger
logging.basicConfig(filename='featurization.log', level=logging.INFO)

class PolyDataset(InMemoryDataset):
    def __init__(self, 
                 save_root, 
                 node_features=None,
                 edge_index=None, 
                 edge_features=None, 
                 y_feature=None,
                 raw_root=None,
                 pos=None, 
                 transform=None, 
                 pre_transform=None):
        
        self.raw_root=raw_root
        self.node_features = node_features
        self.edge_index = edge_index
        self.pos = pos
        self.edge_features = edge_features
        self.y_feature = y_feature
        # self.raw_files=glob(self.raw_root + "/*.json")
        super(PolyDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        
        
    @property
    def raw_file_names(self):
        os.makedirs(self.raw_dir,exist_ok=True)
        return os.listdir(self.raw_dir)
    
        # return os.listdir(self.raw_root)

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        if os.path.exists(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        shutil.copytree(self.raw_root, self.raw_dir)

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
        # for raw_path in self.raw_files:
            with open(raw_path, 'r') as f:
                try:
                    poly_dict = json.load(f)
                    vertices = poly_dict['vertices']
                    featurizer = PolyFeaturizer(vertices)

                    # Compute the features
                    edge_feature_values={}
                    node_feature_values={}
                    y_values={}
                    for feature in self.node_features:
                        node_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        node_feature_values.update({feature['name']: getattr(featurizer, feature['name'])(**node_params)})
                    for feature in self.edge_features:
                        edge_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        edge_feature_values.update({feature['name']: getattr(featurizer, feature['name'])(**edge_params)})
                    for feature in self.y_feature:
                        y_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        y_values.update({feature['name']: getattr(featurizer, feature['name'])(**y_params)})

                    edge_index_values = getattr(featurizer, self.edge_index[0]['name'])()

                    if self.pos:
                        self.pos=pos = getattr(featurizer, self.pos)()

                    # Find label
                    label=raw_path.split(os.sep)[-1].split('.')[0]

                    x=torch.cat([torch.from_numpy(node_feature_values[feature]) for feature in node_feature_values], dim=-1)
                    edge_attr=torch.cat([torch.from_numpy(edge_feature_values[feature]) for feature in edge_feature_values], dim=-1)
                    edge_index=torch.from_numpy(edge_index_values).t().contiguous()
                    y=torch.tensor([y_values[feature] for feature in y_values][0],dtype=torch.float)
                    # Create a Data object.
                    data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,pos=self.pos,label=label)

                    data_list.append(data)
                except Exception as e:
                    print(f"Featurization failed for {raw_path} with error {str(e)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MPPolyDataset(InMemoryDataset):
    def __init__(self, 
                 save_root, 
                 node_features=None,
                 edge_index=None, 
                 edge_features=None, 
                 y_feature=None,
                 raw_root=None,
                 pos=None, 
                 transform=None,
                 n_cores=None,
                 pre_transform=None):
        self.n_cores=n_cores
        self.raw_root=raw_root
        self.node_features = node_features
        self.edge_index = edge_index
        self.pos = pos
        self.edge_features = edge_features
        self.y_feature = y_feature
        # self.raw_files=glob(self.raw_root + "/*.json")
        super(MPPolyDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        
        
    @property
    def raw_file_names(self):
        os.makedirs(self.raw_dir,exist_ok=True)
        return os.listdir(self.raw_dir)
    
        # return os.listdir(self.raw_root)

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        if os.path.exists(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        shutil.copytree(self.raw_root, self.raw_dir)

    def process_file(self, raw_path):
            try:
                with open(raw_path, 'r') as f:
                    poly_dict = json.load(f)
                    vertices = poly_dict['vertices']
                    featurizer = PolyFeaturizer(vertices)

                    # Compute the features
                    edge_feature_values={}
                    node_feature_values={}
                    y_values={}
                    for feature in self.node_features:
                        node_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        node_feature_values.update({feature['name']: getattr(featurizer, feature['name'])(**node_params)})
                    for feature in self.edge_features:
                        edge_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        edge_feature_values.update({feature['name']: getattr(featurizer, feature['name'])(**edge_params)})
                    for feature in self.y_feature:
                        y_params={param: eval(value) if isinstance(value,str) else value for param, value in feature['params'].items() }
                        y_values.update({feature['name']: getattr(featurizer, feature['name'])(**y_params)})

                    edge_index_values = getattr(featurizer, self.edge_index[0]['name'])()

                    if self.pos:
                        self.pos=pos = getattr(featurizer, self.pos)()

                    # Find label
                    label=raw_path.split(os.sep)[-1].split('.')[0]
                    x=torch.cat([torch.from_numpy(node_feature_values[feature]) for feature in node_feature_values], dim=-1)
                    edge_attr=torch.cat([torch.from_numpy(edge_feature_values[feature]) for feature in edge_feature_values], dim=-1)
                    edge_index=torch.from_numpy(edge_index_values).t().contiguous()
                    y=torch.tensor([y_values[feature] for feature in y_values][0],dtype=torch.float)
                    # Create a Data object.

                    x = x.float()
                    edge_attr = edge_attr.float()
                    edge_index = edge_index.long()

                    data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,pos=self.pos,label=label)
                    return data
            except Exception as e:
                print(f"Featurization failed for {raw_path} with error {str(e)}")
                return None

    def process(self):
        with mp.Pool(self.n_cores) as pool:
            data_list = list(filter(None, pool.map(self.process_file, self.raw_paths)))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# # # Usage:

# dataset = PolyDataset(root='data', node_features=['face_centers'], edge_features=['edges'], y_feature='output_label')
# print(len(dataset))
# print(dataset[0])

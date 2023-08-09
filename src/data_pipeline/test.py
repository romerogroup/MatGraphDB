import os

from poly_graphs_lib.data.generate_datasets import generate_datasets
from poly_graphs_lib.data.dataset import PolyDataset
from poly_graphs_lib.utils import ROOT, PROJECT_DIR

dataset = PolyDataset(save_root=os.path.join(PROJECT_DIR,'data','processed','datasets','plutonic_2'))
print(dataset[0])
print(dataset[0].y)
print(dataset[0].label)
print(dataset[0].edge_attr)
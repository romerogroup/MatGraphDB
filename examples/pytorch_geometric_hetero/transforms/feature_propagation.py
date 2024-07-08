

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import FeaturePropagation

from matgraphdb.mlcore.datasets import MaterialGraphDataset
from torch_geometric.data import HeteroData, Data

from torch_geometric.utils import to_undirected

"""https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.FeaturePropagation.html#torch_geometric.transforms.FeaturePropagation"""



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
CONNECTION_TYPE='GEOMETRIC_CONNECTS'
# CONNECTION_TYPE='ELECTRIC_CONNECTS'

# Training params
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1
VAL_PROPORTION = 0.1
LEARNING_RATE = 0.001
N_EPCOHS = 1000

# model params

NUM_LAYERS = 2
HIDDEN_CHANNELS = 128
EVAL_INTERVAL = 10

properties_per_node_type={
    'element':[
        'atomic_number',
        'X',
        'atomic_radius',
        'group',
        'row',
        'atomic_mass'
        ],
    'material':[
        'composition',
        # 'space_group',
        # 'nelements',
        # 'nsites',
        # 'crystal_system',
        # 'density',
        # 'density_atomic',
        # 'volume',
        # 'g_vrh',
        ],
}
properties=[]
for node_type in properties_per_node_type.keys():
    properties.extend(properties_per_node_type[node_type])

if CONNECTION_TYPE=='GEOMETRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.gc_element_chemenv(
                                            use_weights=True,
                                            use_node_properties=True,
                                            properties=properties,
                                            target_property=TARGET_PROPERTY
                                            )
elif CONNECTION_TYPE=='ELECTRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                            use_weights=True,
                                            use_node_properties=True,
                                            properties=properties,
                                            target_property=TARGET_PROPERTY
                                            )
elif CONNECTION_TYPE=='GEOMETRIC_ELECTRIC_CONNECTS':
    graph_dataset=MaterialGraphDataset.gec_element_chemenv(
                                            use_weights=True,
                                            use_node_properties=True,
                                            properties=properties,
                                            target_property=TARGET_PROPERTY
                                            )
data=graph_dataset.data

print(data['element'].property_names)

data['element'].x = data['element'].x.float()

edge_type=('element',CONNECTION_TYPE.lower(),'element')

print(edge_type)
homo_graph = Data(x=data['element'].x, 
                         edge_index=data[edge_type].edge_index)
print(homo_graph.edge_index.shape)

undirected_edge_index = to_undirected(homo_graph.edge_index)

homo_graph.edge_index = undirected_edge_index
print(homo_graph.edge_index.shape)




transform = FeaturePropagation(missing_mask=torch.isnan(homo_graph.x))
homo_graph_transformed = transform(homo_graph)


print(homo_graph_transformed.x)

feature_propagated_path='examples/pytorch_geometric_hetero/transforms/geometric_featue_propagated_element.csv'
# Export graph to csv

df= pd.DataFrame(homo_graph_transformed.x.numpy(),columns=data['element'].property_names)
df.to_csv(feature_propagated_path)



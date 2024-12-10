import os
import random

import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import  SAGEConv, GCNConv, GraphConv, to_hetero, GraphSAGE,to_hetero_with_bases

from matgraphdb.graph_kit.pyg.metrics import ClassificationMetrics,RegressionMetrics


from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager


random.seed(42)
np.random.seed(42)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42) 

DEVICE =  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

generator=DataGenerator()
manager=GraphManager()

node_names=manager.list_nodes()
relationship_names=manager.list_relationships()

print('-'*100)
print('Nodes Types')
print('-'*100)
for i,node_name in enumerate(node_names):
    print(i,node_name)

print('-'*100)
print('Relationships')
print('-'*100)
for i,relationship_name in enumerate(relationship_names):
    print(i,relationship_name)
print('-'*100)



node_files=manager.get_node_filepaths()
relationship_files=manager.get_relationship_filepaths()

print('-'*100)
print('Nodes Types')
print('-'*100)
for i,node_file in enumerate(node_files):
    print(i,node_file)

print('-'*100)
print('Relationships')
print('-'*100)
for i,relationship_file in enumerate(relationship_files):
    print(i,relationship_file)
print('-'*100)




node_properties={
        'CHEMENV':[
            'coordination',
        ],
        'ELEMENT':[
            # 'abundance_universe',
            # 'abundance_solar',
            # 'abundance_meteor',
            # 'abundance_crust',
            # 'abundance_ocean',
            # 'abundance_human',
            'atomic_mass',
            'atomic_number',
            'block',
            'boiling_point',
            'critical_pressure',
            'critical_temperature',
            'density_stp',
            'electron_affinity',
            'electronegativity_pauling',
            'extended_group',
            'heat_specific',
            'heat_vaporization',
            'heat_fusion',
            'heat_molar',
            'magnetic_susceptibility_mass',
            'magnetic_susceptibility_molar',
            'magnetic_susceptibility_volume',
            'melting_point',
            'molar_volume',
            'neutron_cross_section',
            'neutron_mass_absorption',
            'period',
            'radius_calculated',
            'radius_empirical',
            'radius_covalent',
            'radius_vanderwaals',
            'refractive_index',
            'speed_of_sound',
            'conductivity_electric',
            'electrical_resistivity',
            'modulus_bulk',
            'modulus_shear',
            'modulus_young',
            'poisson_ratio',
            'coefficient_of_linear_thermal_expansion',
            'hardness_vickers',
            'hardness_brinell',
            'hardness_mohs',
            'superconduction_temperature',
            # 'is_actinoid',
            # 'is_alkali',
            # 'is_alkaline',
            # 'is_chalcogen',
            # 'is_halogen',
            # 'is_lanthanoid',
            # 'is_metal',
            # 'is_metalloid',
            # 'is_noble_gas',
            # 'is_post_transition_metal',
            # 'is_quadrupolar',
            # 'is_rare_earth_metal',
            # 'experimental_oxidation_states',
        ],
        'MATERIAL':[
            'nsites',
            'nelements',
            'volume',
            'density',
            'density_atomic',
            'crystal_system',
            'a',
            'b',
            'c',
            'alpha',
            'beta',
            'gamma',
            'unit_cell_volume',
            # 'energy_per_atom',
            # 'formation_energy_per_atom',
            # 'energy_above_hull',
            # 'band_gap',
            # 'cbm',
            # 'vbm',
            # 'efermi',
            # 'is_stable',
            # 'is_gap_direct',
            # 'is_metal',
            # 'is_magnetic',
            # 'ordering',
            # 'total_magnetization',
            # 'total_magnetization_normalized_vol',
            # 'num_magnetic_sites',
            # 'num_unique_magnetic_sites',
            # 'e_total',
            # 'e_ionic',
            # 'e_electronic',
            # 'sine_coulomb_matrix',
            # 'element_fraction',
            # 'element_property',
            # 'xrd_pattern',
            # 'uncorrected_energy_per_atom',
            # 'equilibrium_reaction_energy_per_atom',
            # 'n',
            # 'e_ij_max',
            # 'weighted_surface_energy_EV_PER_ANG2',
            # 'weighted_surface_energy',
            # 'weighted_work_function',
            # 'surface_anisotropy',
            # 'shape_factor',
            # 'elasticity-k_vrh',
            # 'elasticity-k_reuss',
            # 'elasticity-k_voigt',
            # 'elasticity-g_vrh',
            # 'elasticity-g_reuss',
            # 'elasticity-g_voigt',
            # 'elasticity-sound_velocity_transverse',
            # 'elasticity-sound_velocity_longitudinal',
            # 'elasticity-sound_velocity_total',
            # 'elasticity-sound_velocity_acoustic',
            # 'elasticity-sound_velocity_optical',
            # 'elasticity-thermal_conductivity_clarke',
            # 'elasticity-thermal_conductivity_cahill',
            # 'elasticity-young_modulus',
            # 'elasticity-universal_anisotropy',
            # 'elasticity-homogeneous_poisson',
            # 'elasticity-debye_temperature',
            # 'elasticity-state',
        ]
    }


relationship_properties={
    'ELEMENT-CAN_OCCUR-CHEMENV':[
        'weight',
        ],
    'ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT':[
        'weight',
        ],
    'ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT':[
        'weight',
        ],
    'CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV':[
         'weight',
      ],
    'MATERIAL-HAS-CHEMENV':[
         'weight',
    ],
    'MATERIAL-HAS-ELEMENT':[
        'weight',
    ],
}


generator.add_node_type(node_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/nodes/CHEMENV.parquet',
                        feature_columns=node_properties['CHEMENV'],
                        target_columns=[])
generator.add_node_type(node_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/nodes/ELEMENT.parquet',
                        feature_columns=node_properties['ELEMENT'],
                        target_columns=[])
generator.add_node_type(node_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/nodes/MATERIAL.parquet',
                        feature_columns=node_properties['MATERIAL'],
                        target_columns=['elasticity-k_vrh'],
                        filter={'elasticity-k_vrh':(0,300)})
print(generator.hetero_data)



generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV.parquet',
                        feature_columns=relationship_properties['CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT.parquet',
                        feature_columns=relationship_properties['ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

# generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT.parquet',
#                         feature_columns=relationship_properties['ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT'],
#                         # target_columns=['weight'],
#                         # custom_encoders={}, 
#                         # node_filter={},
#                         undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-CHEMENV.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-CHEMENV'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-ELEMENT.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-ELEMENT'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/ELEMENT-CAN_OCCUR-CHEMENV.parquet',
                        feature_columns=relationship_properties['ELEMENT-CAN_OCCUR-CHEMENV'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)
print(generator.hetero_data)


print(dir(generator.hetero_data))

print(generator.hetero_data.edge_types)

print(generator.hetero_data.metadata())



class HeteroInputLayer(nn.Module):
    def __init__(self, data, n_embd:int, 
                device='cuda:0'):
        super().__init__()

        self.embs = nn.ModuleDict()
        self.data_lins = nn.ModuleDict()

        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            num_features = data[node_type].num_node_features

            self.embs[node_type]=nn.Embedding(num_nodes,n_embd,device=device)
            if num_features != 0:
                self.data_lins[node_type]=nn.Linear(num_features, n_embd,device=device)

    def forward(self, data):
        x_dict={}
        edge_index_dict={}
        edge_attr_dict={}
        for node_type, emb_layer in self.embs.items():
            # Handling nodes based on feature availability
            if node_type in self.data_lins:
                x_dict[node_type] = self.data_lins[node_type](data[node_type].x) + emb_layer(data[node_type].node_id)
            else:
                x_dict[node_type] = emb_layer(data[node_type].node_id)

            # edge_index_dict[node_type] = data[node_type].edge_index
            # edge_attr_dict[node_type] = data[node_type].edge_attr

        return x_dict


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, device=DEVICE),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd,device=DEVICE),
            nn.Dropout(dropout),
        )
        self.ln=nn.LayerNorm(n_embd, device=DEVICE)

    def forward(self, x):
        return self.net(self.ln(x))


class HeteroConvModel(nn.Module):
    def __init__(self, data, n_embd:int, 
                out_channels:int,
                prediction_node_type:str,
                n_conv_layers=1,
                aggr='sum',
                dropout=0.0,
                conv_params={
                    'dropout': 0.0,
                    'act': 'relu',
                    'act_first': True},
                device='cuda:0'):
        super(HeteroConvModel, self).__init__()
        self.prediction_node_type=prediction_node_type
        self.input_layer=HeteroInputLayer(data, n_embd, device=device)

        self.fwd1_dict = nn.ModuleDict()
        for node_type in data.node_types:
           self.fwd1_dict[node_type]=FeedFoward(n_embd, dropout=dropout)

        # Initialize and convert GraphSAGE to heterogeneous
        self.graph_conv= GraphSAGE( n_embd, n_embd, n_conv_layers, **conv_params)
        self.stacked_conv = to_hetero(self.graph_conv, metadata=data.metadata())
        # self.stacked_conv = to_hetero_with_bases(model, metadata, bases=3)

        self.fwd2_dict = nn.ModuleDict()
        for node_type in data.node_types:
           self.fwd2_dict[node_type]=FeedFoward(n_embd, dropout=dropout)

        self.output_layer = nn.Linear(n_embd, out_channels)

    def forward(self, data):
        x_dict = self.input_layer(data)
        for node_type in data.node_types:
            x_dict[node_type]=self.fwd1_dict[node_type](x_dict[node_type])
        x_dict=self.stacked_conv(x_dict, data.edge_index_dict)
        for node_type in data.node_types:
            x_dict[node_type]=self.fwd2_dict[node_type](x_dict[node_type])

        out=self.output_layer(x_dict[self.prediction_node_type])
        return out
    


NODE_TYPE='MATERIAL'
def split_data_on_node_type(data,node_type,train_proportion=0.8,test_proportion=0.1, val_proportion=0.1):
    assert train_proportion + test_proportion + val_proportion == 1.0
    for node_type in data.node_types:
        train_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        test_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)
        val_mask=torch.zeros(data[node_type].num_nodes,dtype=torch.bool)

        num_nodes_for_type=data[node_type].num_nodes
        if node_type==NODE_TYPE:
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


# Training params
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1
VAL_PROPORTION = 0.1
LEARNING_RATE = 0.001
N_EPCOHS = 400

N_EMBD=128
OUT_CHANNELS = 1
NUM_LAYERS=2
EVAL_INTERVAL = 10

hetero_data=generator.hetero_data


hetero_data=split_data_on_node_type(hetero_data,
                            node_type=NODE_TYPE,
                            train_proportion=TRAIN_PROPORTION,
                            test_proportion=TEST_PROPORTION,
                            val_proportion=VAL_PROPORTION)



model=HeteroConvModel(hetero_data, 
                n_embd=N_EMBD, 
                out_channels=OUT_CHANNELS,
                prediction_node_type=NODE_TYPE,
                n_conv_layers=NUM_LAYERS,
                aggr='add',
                dropout=0.3,
                conv_params={
                    'dropout': 0.3,
                    'act': 'relu',
                    'act_first': True},
                device=DEVICE)
model.to(device=DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
if OUT_CHANNELS==1:
    loss_fn=nn.MSELoss()
else:
    loss_fn=nn.CrossEntropyLoss()



def train(model, optimizer, data, loss_fn):
    model.train()
    total_loss = 0
    total_examples = 0

    data = data.to(DEVICE)
    optimizer.zero_grad()

    out = model(data)
    mask=data[NODE_TYPE].train_mask
    pred=out[mask]
    ground_truth=data[NODE_TYPE].y[mask]

    loss = loss_fn(pred, ground_truth)

    loss.backward()
    optimizer.step()
    total_loss += float(loss)
    total_examples += pred.numel()

    return total_loss 


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        
        losses=[]
        metrics={"val":{},"test":{}}
        data = data.to(DEVICE)
        optimizer.zero_grad()

        logits = model(data)
        out_channel=logits.shape[1]
        for key in metrics.keys():
            if out_channel==1:
                metrics[key]['mape']=[]
                metrics[key]['mae']=[]
            else:
                metrics[key]['accuracy']=[]
                metrics[key]['precision']=[]
                metrics[key]['recall']=[]
                metrics[key]['f1']=[]

        for split in ['val_mask', 'test_mask']:
            mask=data[NODE_TYPE][split]
            masked_logits=logits[mask]
            ground_truth=data[NODE_TYPE].y[mask]

            loss = loss_fn(masked_logits, ground_truth)

            out_channel=logits.shape[1]
            split_name=split.split('_')[0]
            if out_channel==1:
                mape=RegressionMetrics.mean_absolute_percentage_error(y_pred=masked_logits,y_true=ground_truth)
                mae=RegressionMetrics.mean_absolute_error(y_pred=masked_logits,y_true=ground_truth)
                metrics[split_name]['mape'].append(mape.item())
                metrics[split_name]['mae'].append(mae.item())
            else:
                probabilities = torch.sigmoid(masked_logits)
                # Converting masked_logits from (batch_size, out_channels) to (batch_size,)
                pred=probabilities.argmax(1)

                
                # accuracy=ClassificationMetrics.accuracy(y_pred=pred,y_true=ground_truth)
                # metrics[split_name]['accuracy'].append(accuracy.item()*100)

                cm=ClassificationMetrics.confusion_matrix(y_pred=pred,y_true=ground_truth,num_classes=out_channel)

                weights=cm.sum(dim=1)/cm.sum(dim=1).sum()

                accuracy=ClassificationMetrics.multi_class_accuracy(confusion_matrix=cm)
                avg_accuracy=(weights * accuracy).sum()
                metrics[split_name]['accuracy'].append(avg_accuracy*100)

                precision=ClassificationMetrics.multiclass_precision(confusion_matrix=cm)
                avg_precision= (weights * precision).sum()
                metrics[split_name]['precision'].append(avg_precision*100)

                recall=ClassificationMetrics.multiclass_recall(confusion_matrix=cm)
                avg_recall=(weights * recall).sum()
                metrics[split_name]['recall'].append(avg_recall*100)

                f1=ClassificationMetrics.multiclass_f1_score(confusion_matrix=cm)
                avg_f1=(weights * f1).sum()
                metrics[split_name]['f1'].append(avg_f1*100)

            losses.append(loss.item())

    return losses,metrics



for epoch in range(N_EPCOHS):
    train_loss = train(model, optimizer, hetero_data, loss_fn=loss_fn)
    if epoch%EVAL_INTERVAL==0:
        losses,metrics = evaluate(model, hetero_data)

        # print(f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Val Loss: {losses[0]:.4f}, Test Loss: {losses[1]:.4f}")

        metrics_str=""
        metrics_str+=f"Epoch: {epoch:03d},Train Loss: {train_loss:.4f}, Test Loss: {losses[1]:.4f}"
        for split,metrics_dict in metrics.items():
            metrics_str+=" | "
            for i,(key,value) in enumerate(metrics_dict.items()):
                if i==0:
                    metrics_str+=f" {split}-{key}: {value[0]:.2f}"
                else:
                    metrics_str+=f", {split}-{key}: {value[0]:.2f}"
        print(metrics_str)



def parity_plot(data, model, node_type=NODE_TYPE):
    from sklearn.metrics import mean_squared_error, r2_score
    model.eval()
    out = model(data)
    mask = data[node_type].test_mask
    pred = out[mask].to('cpu').detach().numpy()
    ground_truth = data[node_type].y[mask].to('cpu').detach().numpy()

    # Calculate metrics
    mse = mean_squared_error(ground_truth, pred)
    r2 = r2_score(ground_truth, pred)

    # Create the scatter plot
    plt.scatter(ground_truth, pred, s=10)
    plt.plot([0, 300], [0, 300], color='red', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    # Add MSE and R-squared to the plot
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {mse**0.5:.4f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top')

    plt.show()


parity_plot(hetero_data, model=model, node_type=NODE_TYPE)
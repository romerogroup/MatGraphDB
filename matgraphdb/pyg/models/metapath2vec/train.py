import json
import os
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from torch_geometric import nn as pyg_nn

from matgraphdb.materials.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.data import HeteroGraphBuilder
from matgraphdb.pyg.models.metapath2vec.metrics import plot_pca
import matplotlib.patches as mpatches
from sklearn import linear_model
from matgraphdb.utils.colors import DEFAULT_COLORS, DEFAULT_CMAP
from matgraphdb.utils.config import config
from torch_geometric.nn import MetaPath2Vec
########################################################################################################################
import umap

LOGGER = logging.getLogger(__name__)

print(LOGGER)
# # Set up logging to print to console in debug mode
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )

# # Add file-specific logger configuration
# LOGGER.setLevel(logging.DEBUG)
# # Remove any existing handlers to avoid duplicate logging
# LOGGER.handlers = []
# LOGGER.addHandler(logging.StreamHandler())





def to_log(x):
    return torch.tensor(np.log10(x), dtype=torch.float32)
    
DATA_CONFIG = OmegaConf.create(
    {
    "dataset_dir": os.path.join("data", "datasets", "MPNearHull"),
    "nodes" :
        {"materials": {"columns": ["core.density_atomic"], 'drop_null': True},
         "elements": {"columns": ["atomic_mass", "radius_covalent", "radius_vanderwaals"], 'drop_null':True, 'label_column': 'symbol'},
         "space_groups": {'drop_null': True, 'label_column': 'spg'},
         "crystal_systems": {'drop_null': True, 'label_column': 'crystal_system'}
        },
    "edges" :
        {
        "element_element_neighborsByGroupPeriod": {},
        "material_element_has": {},
        "material_spg_has": {},
        "material_crystalSystem_has": {}
        },
    "target":{
        "materials": {"columns": ["elasticity.g_vrh"], 'drop_null': True, 
                      'filters': "[pc.field('elasticity.g_vrh') > 0, pc.field('elasticity.g_vrh') < 400]",
                      'encoders': "{'elasticity.g_vrh': to_log}"}
        }
    }
)

METAPATH2VEC_CONFIG = OmegaConf.create(
    {
        "data": dict(DATA_CONFIG),
        "model": {
            "embedding_dim": 32,
            # "embedding_dim": 4,
            "walk_length": 50,
            # "walk_length": 10,
            "context_size": 7,
            "walks_per_node": 5,
            "num_negative_samples": 5,
            "sparse": True,
            # "metapath": [('materials', 'has', 'elements'), ('elements', 'rev_has', 'materials')]
            # "metapath": [('materials', 'has', 'crystal_systems'), ('crystal_systems', 'rev_has', 'materials')]
            "metapath": [('materials', 'has', 'space_groups'), ('space_groups', 'rev_has', 'materials')]
        },
        "training": {
            "train_dir": os.path.join("data", "training_runs", "metapath2vec"),
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "batch_size": 256,
            "num_workers": 4,
            "learning_rate": 0.01,
            "num_epochs": 5,
            "log_steps": 100,
            "eval_steps": 2000,
            "test_train_ratio": 0.8,
            "test_max_iter": 150
        }
    }
)

MLP_CONFIG = OmegaConf.create({
    "data": dict(DATA_CONFIG),
    "model": {
        "mlp_hidden_dim": 32,  # hidden dimension for the MLP baseline
    },
    "training": {
        "learning_rate": 0.001,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "epochs": 2000,
    }
})

####################################################################################################
####################################################################################################

print("-" * 100)
print(f"Torch version: {torch.__version__}")
print(f"PyTorch Geometric version: {pyg.__version__}")
print(
    f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("-" * 100)


####################################################################################################q
# Data Preprocessing
####################################################################################################
def build_heterograph():
    """Build the initial heterogeneous graph from the materials database.
    
    Returns:
        torch_geometric.data.HeteroData: The constructed heterogeneous graph
    """
    mdb = MPNearHull(DATA_CONFIG.dataset_dir)
    builder = HeteroGraphBuilder(mdb)
    
    # Define the "materials" node type (only a subset of columns is used here)
    for node_type, node_config in DATA_CONFIG.nodes.items():
        node_config = OmegaConf.to_container(node_config)
        builder.add_node_type(node_type, **node_config)
    
    for edge_type, edge_config in DATA_CONFIG.edges.items():
        edge_config = OmegaConf.to_container(edge_config)
        builder.add_edge_type(edge_type, **edge_config)

    # Add a target property for the "materials" node type.
    for target_type, target_config in DATA_CONFIG.target.items():
        target_config = OmegaConf.to_container(target_config)
        if "filters" in target_config:
            filters = target_config.pop("filters")
            filters = eval(filters)
        if "encoders" in target_config:
            encoders = target_config.pop("encoders")
            encoders = eval(encoders)
        
        builder.add_target_node_property(target_type, filters=filters, encoders=encoders, **target_config)
        
    heterodata = builder.hetero_data
    LOGGER.info(f"HeteroData: {heterodata}")
    heterodata["materials"].original_x = heterodata["materials"].x  # Save original features
    return heterodata


def heterograph_preprocessing():
    """
    Build the heterograph, apply transformations, partition the graph,
    and split the 'materials' nodes into training/validation/test subgraphs.
    
    Args:
        config (OmegaConf): A configuration object with the keys:
            - data: data-related parameters (e.g., dataset_dir, create_random_features, n_material_dim, train_ratio, val_ratio)
            - model: model-related parameters (e.g., n_partitions)
            - training: training-related parameters
            
    Returns:
        split_data (dict): A dictionary with keys "train", "train_val", "test", "test_val",
                           each containing a subgraph for the corresponding split.
    """
    # 1. Build the heterogeneous graph from the materials database
    
        
    original_heterograph = build_heterograph()
    
    
    # 2. Apply transformation: make the graph undirected.
    source_data = T.ToUndirected()(original_heterograph)
    # Free up memory.
    original_heterograph = None
    return source_data


# ####################################################################################################
# # Model
# ####################################################################################################

class MLPBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPBaseline, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(heterodata, metapath2vec_model):
    z = metapath2vec_model('materials', batch=heterodata['materials'].y_index.to(DEVICE))
    y = heterodata['materials'].y.to(DEVICE).squeeze()
    
    material_indices = torch.randperm(z.size(0))
    
    n_materials = z.size(0)
    train_ratio = MLP_CONFIG.training.train_ratio
    val_ratio = MLP_CONFIG.training.val_ratio
    test_ratio = 1 - train_ratio
    
    train_size = int(train_ratio * n_materials)
    test_size = int(test_ratio * n_materials)
    train_val_size = int(val_ratio * train_size)
    test_val_size = int(val_ratio * test_size)
    
    total_train_materials = material_indices[:train_size]
    total_test_materials = material_indices[train_size:]
        
    train_materials = total_train_materials[:train_val_size]
    test_materials = total_test_materials[:test_val_size]

    # Split train and test into their validation sets
    train_val_materials = total_train_materials[:train_val_size]
    train_materials = total_train_materials[train_val_size:]
    test_val_materials = total_test_materials[:test_val_size]
    test_materials = total_test_materials[test_val_size:]
    split_data = {
        "train": train_materials,
        "train_val": train_val_materials,
        "test": test_materials,
        "test_val": test_val_materials,
    }
    """Train an MLP baseline model using PyTorch."""
    input_dim = z.shape[1]
    hidden_dim = MLP_CONFIG.model.mlp_hidden_dim
    model = MLPBaseline(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_CONFIG.training.learning_rate)
    loss_fn = torch.nn.L1Loss()
    
    # Initialize results storage
    results = {
        "train":     {"loss": [], "mae": [], "epochs": []},
        "train_val": {"loss": [], "mae": [], "epochs": []},
        "test":      {"loss": [], "mae": [], "epochs": []},
        "test_val":  {"loss": [], "mae": [], "epochs": []},
    }

    def train_step():
        
        model.train()
        optimizer.zero_grad()  # Move this here, before the forward pass

        total_loss = 0
        y_pred = model(z[split_data["train"]]).squeeze()
        loss = loss_fn(y_pred, y[split_data["train"]])
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss += loss.item()
        
        results["train"]["loss"].append(float(total_loss))

    @torch.no_grad()
    def test_step():
        model.eval()
        
        for split_name, split_materials in split_data.items():
            y_pred = model(z[split_materials]).squeeze().cpu().numpy()
            y_real = y[split_materials].cpu().numpy()
                
            y_pred = np.array([10**value for value in y_pred])
            y_real = np.array([10**value for value in y_real])

            mae = np.mean(np.abs(y_pred - y_real))
            results[split_name]["mae"].append(float(mae))

        
    
    for epoch in range(MLP_CONFIG.training.epochs):
        train_step()
        test_step()
        results["train"]["epochs"].append(epoch)
        results["train_val"]["epochs"].append(epoch)
        results["test"]["epochs"].append(epoch)
        results["test_val"]["epochs"].append(epoch)
        
    loss_str = f"Epoch: {epoch},"
    for split_name, split_results in results.items():
        loss_str += f"{split_name}: {split_results['mae'][-1]:.4f} "
    print(loss_str)
    
    return results

def train_metapath2vec(heterodata):
    metapath=[]
    for path in METAPATH2VEC_CONFIG.model.metapath:
        metapath.append(tuple(path))
        
    num_nodes_dict = {node_type: heterodata[node_type].num_nodes for node_type in heterodata.node_types}
    model = MetaPath2Vec(heterodata.edge_index_dict, 
                         embedding_dim=METAPATH2VEC_CONFIG.model.embedding_dim,
                         metapath=metapath, 
                         walk_length=METAPATH2VEC_CONFIG.model.walk_length, 
                         context_size=METAPATH2VEC_CONFIG.model.context_size,
                         walks_per_node=METAPATH2VEC_CONFIG.model.walks_per_node, 
                         num_negative_samples=METAPATH2VEC_CONFIG.model.num_negative_samples,
                         sparse=METAPATH2VEC_CONFIG.model.sparse,
                         num_nodes_dict=num_nodes_dict).to(DEVICE)

    loader = model.loader(batch_size=METAPATH2VEC_CONFIG.training.batch_size, 
                          shuffle=True, 
                          num_workers=METAPATH2VEC_CONFIG.training.num_workers)
    print(model)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), 
                                     lr=METAPATH2VEC_CONFIG.training.learning_rate)

    results = {
        "train":     {"mae": [], "loss": [], "epochs": []},
        "train_val": {"mae": [], "loss": [], "epochs": []},
        "test":      {"mae": [], "loss": [], "epochs": []},
        "test_val":  {"mae": [], "loss": [], "epochs": []},
    }

    def train_step():
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        results["train"]["loss"].append(float(total_loss / len(loader)))

    @torch.no_grad()
    def test_step():
        model.eval()

        z = model('materials', batch=heterodata['materials'].y_index.to(DEVICE))
        y = heterodata['materials'].y

        material_indices = torch.randperm(z.size(0))
        
        n_materials = z.size(0)
        train_ratio = METAPATH2VEC_CONFIG.training.train_ratio
        val_ratio = METAPATH2VEC_CONFIG.training.val_ratio
        test_ratio = 1 - train_ratio
        
        train_size = int(train_ratio * n_materials)
        test_size = int(test_ratio * n_materials)
        train_val_size = int(val_ratio * train_size)
        test_val_size = int(val_ratio * test_size)
        
        total_train_materials = material_indices[:train_size]
        total_test_materials = material_indices[train_size:]
            
        train_materials = total_train_materials[:train_val_size]
        test_materials = total_test_materials[:test_val_size]

        # Split train and test into their validation sets
        train_val_materials = total_train_materials[:train_val_size]
        train_materials = total_train_materials[train_val_size:]
        test_val_materials = total_test_materials[:test_val_size]
        test_materials = total_test_materials[test_val_size:]
        split_data = {
            "train": train_materials,
            "train_val": train_val_materials,
            "test": test_materials,
            "test_val": test_val_materials,
        }
        
        reg = linear_model.LinearRegression()
        reg.fit(z[split_data["train"]].cpu().numpy(), y[split_data["train"]].cpu().numpy())
        
        for split_name, split_materials in split_data.items():
            y_pred = reg.predict(z[split_materials].cpu().numpy())
            y_real = y[split_materials].cpu().numpy()
            
            if split_name != "train":
                loss = np.mean(np.abs(y_pred - y_real))
                results[split_name]["loss"].append(float(loss))
                
            y_pred = np.array([10**value for value in y_pred])
            y_real = np.array([10**value for value in y_real])

            mae = np.mean(np.abs(y_pred - y_real))
            results[split_name]["mae"].append(float(mae))

    os.makedirs(METAPATH2VEC_CONFIG.training.train_dir, exist_ok=True)

    for epoch in range(METAPATH2VEC_CONFIG.training.num_epochs):
        train_step()
        test_step()
        results["train"]["epochs"].append(epoch)
        results["train_val"]["epochs"].append(epoch)
        results["test"]["epochs"].append(epoch)
        results["test_val"]["epochs"].append(epoch)
        
        loss_str = f"Epoch: {epoch},"
        for split_name, split_results in results.items():
            loss_str += f"{split_name}: {split_results['mae'][-1]:.4f} "
        print(loss_str)
        
    return model, results



def main():
    heterodata = heterograph_preprocessing()
    model, linear_results = train_metapath2vec(heterodata)

    mlp_results = train_mlp_baseline(heterodata, model)
    
    training_dir = METAPATH2VEC_CONFIG.training.train_dir
    
    runs_dir = os.path.join(training_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    n_runs = len(os.listdir(runs_dir))
    results_dir = os.path.join(runs_dir, f"run_{n_runs}")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "metapath2vec_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(METAPATH2VEC_CONFIG), f)

    with open(os.path.join(results_dir, "mlp_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(MLP_CONFIG), f)
        
    with open(os.path.join(results_dir, "linear_results.json"), "w") as f:
        json.dump(linear_results, f)
        
    with open(os.path.join(results_dir, "mlp_results.json"), "w") as f:
        json.dump(mlp_results, f)
        
    plot_learning_curves(linear_results, os.path.join(results_dir, "linear_learning_curves.png"))
    plot_learning_curves(mlp_results, os.path.join(results_dir, "mlp_learning_curves.png"))
    
    
    
    z_per_type = {
        "materials": model('materials'),
        # "elements": model('elements'),
        "space_groups": model('space_groups'),
        # "crystal_systems": model('crystal_systems'),
    }
    targets_per_type = {
        "materials": 10 ** heterodata['materials'].y.cpu().numpy(),
    }
    targets_labels_per_type = {
        "materials": heterodata['materials'].y_label_name[0],
    }
    targets_index_per_type = {
        "materials": heterodata['materials'].y_index.cpu().numpy(),
    }
    LOGGER.info(f"Targets index per type: {len(heterodata['elements'].labels)}")
    labels_per_type = {
        "elements": heterodata['elements'].labels,
        "space_groups": heterodata['space_groups'].labels,
        # "crystal_systems": heterodata['crystal_systems'].labels,
    }
    color_per_type = {
        # "elements": "black",
        "space_groups": "black",
        # "crystal_systems": "black",
    }
    
    create_umap_plot(z_per_type, 
                     targets_per_type=targets_per_type,
                     targets_index_per_type=targets_index_per_type,
                     targets_labels_per_type=targets_labels_per_type,
                     labels_per_type=labels_per_type,
                     color_per_type=color_per_type,
                     save_path=os.path.join(results_dir, "umap.png"),
                     n_neighbors=30)
    # create_umap_plot3d(z_per_type, 
    #                  targets_per_type=targets_per_type,
    #                  targets_index_per_type=targets_index_per_type,
    #                  labels_per_type=labels_per_type,
    #                  color_per_type=color_per_type,
    #                  save_path=os.path.join(results_dir, "umap_materials_elements_3d.png"),
    #                  n_neighbors=30)
    

def plot_learning_curves(results, save_path, measure='mae'):
    """
    Plots the learning curves for a specified measure from the results dictionary.

    Parameters:
        results (dict): Dictionary containing keys for each dataset split (e.g., 'train', 'train_val', etc.)
                        Each split should contain a dictionary with keys for the measure (e.g., 'loss', 'mae')
                        and 'epochs'.
        measure (str): The measure to plot (e.g., 'loss' or 'mae'). Default is 'loss'.
    """
    plt.figure(figsize=(10, 6))
    
    # Iterate over the splits in the results dictionary
    for idx, split in enumerate(results):
        split_data = results[split]
        
        # Check if the desired measure is available in this split's data
        if measure not in split_data:
            print(f"Warning: Measure '{measure}' not found for split '{split}'. Skipping.")
            continue

        # Use the provided 'epochs' list if available, otherwise create a range based on the measure length
        epochs = split_data.get("epochs", list(range(len(split_data[measure]))))
        values = split_data[measure]
        
        # Select a color for this plot
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        
        # Plot the curve for this split
        plt.plot(epochs, values, label=split, color=color, linewidth=2)

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(measure.capitalize(), fontsize=12)
    plt.title(f"Learning Curves ({measure.capitalize()})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def create_umap_plot(z_per_type, 
                    targets_per_type:dict=None,
                    targets_index_per_type:dict=None,
                    targets_labels_per_type:dict=None,
                    filter_index_per_type:dict=None,
                    labels_per_type:dict=None,
                    color_per_type:dict=None,
                    save_path=".", 
                    n_neighbors=50,
                    n_jobs=4):
    node_types = list(z_per_type.keys())
    
    if targets_per_type is None:
        targets_per_type = {}
    if targets_index_per_type is None:
        targets_index_per_type = {}
    if filter_index_per_type is None:
        filter_index_per_type = {}
    if labels_per_type is None:
        labels_per_type = {}
    if color_per_type is None:
        color_per_type = {}
    if targets_labels_per_type is None:
        targets_labels_per_type = {}
        
    z_global_idx_per_type={}
    z_local_idx_per_type={}
    local_global_idx_mapping_per_type={}
    z_node_type_mapping = {}

    
    z_all=None
    total_n_nodes=0
    for i, (node_type, z) in enumerate(z_per_type.items()):
        z=z.detach().cpu().numpy()
        n_nodes = z.shape[0]
        
        
        LOGGER.info(f"Node type: {node_type}, Number of nodes: {n_nodes}")
        
        z_node_type_mapping[node_type] = i
        z_global_idx_per_type[node_type] = np.arange(total_n_nodes, total_n_nodes + n_nodes)
        z_local_idx_per_type[node_type] = np.arange(n_nodes)
        local_global_idx_mapping_per_type[node_type] = {i:j for i,j in zip(z_local_idx_per_type[node_type], z_global_idx_per_type[node_type])}
        if z_all is None:
            z_all = z
        else:
            z_all = np.concatenate([z_all, z], axis=0)
            
        total_n_nodes+=n_nodes
        
  
    # Apply UMAP to reduce dimensions to 2.
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, n_jobs=n_jobs)
    embedding = reducer.fit_transform(z_all)

    # Create the scatter plot.
    plt.figure(figsize=(10, 8))
    
    
    handles=[]
    scatter_handles=[]
    for node_type in node_types:
        LOGGER.info(f"Plotting {node_type}")
        
        color = color_per_type.get(node_type, None)
        node_labels = labels_per_type.get(node_type, None)
        targets = targets_per_type.get(node_type, None)
        target_idx = targets_index_per_type.get(node_type, None)
        filter_idx = filter_index_per_type.get(node_type, None)
        
        node_idx = z_global_idx_per_type.get(node_type, None)
        LOGGER.info(f"Node index: {node_idx}")
        if target_idx is not None:
            LOGGER.info(f"Target index: {target_idx}")
            node_idx = [local_global_idx_mapping_per_type[node_type][i] for i in target_idx]
            if node_labels is not None:
                node_labels = node_labels[target_idx] # Needs to be local index
        if filter_idx is not None:
            node_idx = [local_global_idx_mapping_per_type[node_type][i] for i in filter_idx]
            if node_labels is not None:
                node_labels = node_labels[filter_idx] # Needs to be local index
            
        
        if targets is not None:
            c = targets
            cmap=DEFAULT_CMAP
        elif color is not None:
            c=color
            cmap=None
            handles.append(mpatches.Patch(color=color, label=node_type))
        

        x = embedding[node_idx, 0] # Needs to be global index
        y = embedding[node_idx, 1] # Needs to be global index
        scatter = plt.scatter(x, y, s=10, alpha=0.8, 
                                c=c, 
                                cmap=cmap)
        c=None
        
        if targets is not None:
            LOGGER.info(f"Plotting {node_type} targets")
            scatter_handles.append(scatter)
        
        if node_labels is not None:
            LOGGER.info(f"Plotting {node_type} labels, n_labels: {len(node_labels)}")
            for i, label in enumerate(node_labels):
                plt.annotate(label, (x[i], y[i]), fontsize=8, alpha=1)

  
    if targets_per_type:
        label=""
        for node_type in node_types:
            label+=targets_labels_per_type.get(node_type, "")
        plt.colorbar(scatter_handles[0], label=label)
    plt.legend(handles=handles)  
    plt.title("UMAP Projection of Node Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(save_path)
    plt.close()




def create_umap_plot3d(z_per_type, 
                     targets_per_type: dict = None,
                     targets_index_per_type: dict = None,
                     filter_index_per_type: dict = None,
                     labels_per_type: dict = None,
                     color_per_type: dict = None,
                     save_path="umap_3d_plot.png", 
                     n_neighbors=50,
                     n_jobs=4):
    """
    Creates a 3D UMAP scatter plot from node embeddings for multiple node types.

    Parameters:
        z_per_type (dict): Dictionary mapping node types to their embeddings (torch tensors).
        targets_per_type (dict, optional): Dictionary mapping node types to target values for coloring.
        targets_index_per_type (dict, optional): Dictionary mapping node types to indices for target selection.
        filter_index_per_type (dict, optional): Dictionary mapping node types to indices for filtering.
        labels_per_type (dict, optional): Dictionary mapping node types to labels for annotation.
        color_per_type (dict, optional): Dictionary mapping node types to a specific color.
        save_path (str): Path (including filename) to save the plot.
        n_jobs (int): Number of parallel jobs to run in UMAP.
    """
    
    node_types = list(z_per_type.keys())
    
    # Set default dictionaries if None.
    if targets_per_type is None:
        targets_per_type = {}
    if targets_index_per_type is None:
        targets_index_per_type = {}
    if filter_index_per_type is None:
        filter_index_per_type = {}
    if labels_per_type is None:
        labels_per_type = {}
    if color_per_type is None:
        color_per_type = {}
        
    z_global_idx_per_type = {}
    z_local_idx_per_type = {}
    local_global_idx_mapping_per_type = {}
    z_node_type_mapping = {}

    z_all = None
    total_n_nodes = 0
    for i, (node_type, z) in enumerate(z_per_type.items()):
        # Convert tensor to numpy array.
        z = z.detach().cpu().numpy()
        n_nodes = z.shape[0]
        LOGGER.info(f"Node type: {node_type}, Number of nodes: {n_nodes}")
        
        z_node_type_mapping[node_type] = i
        z_global_idx_per_type[node_type] = np.arange(total_n_nodes, total_n_nodes + n_nodes)
        z_local_idx_per_type[node_type] = np.arange(n_nodes)
        local_global_idx_mapping_per_type[node_type] = {
            local_idx: global_idx 
            for local_idx, global_idx in zip(z_local_idx_per_type[node_type], z_global_idx_per_type[node_type])
        }
        z_all = z if z_all is None else np.concatenate([z_all, z], axis=0)
        total_n_nodes += n_nodes

    # Apply UMAP to reduce dimensions to 3.
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=3, n_jobs=n_jobs)
    embedding = reducer.fit_transform(z_all)

    # Create the 3D scatter plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    handles = []
    scatter_handles = []
    
    for node_type in node_types:
        LOGGER.info(f"Plotting {node_type}")
        
        color = color_per_type.get(node_type, None)
        node_labels = labels_per_type.get(node_type, None)
        targets = targets_per_type.get(node_type, None)
        target_idx = targets_index_per_type.get(node_type, None)
        filter_idx = filter_index_per_type.get(node_type, None)
        
        # Start with all global indices for this node type.
        node_idx = z_global_idx_per_type.get(node_type, None)
        LOGGER.info(f"Node index: {node_idx}")
        
        # If a target index is specified, map local indices to global indices.
        if target_idx is not None:
            LOGGER.info(f"Target index: {target_idx}")
            node_idx = [local_global_idx_mapping_per_type[node_type][i] for i in target_idx]
            if node_labels is not None:
                node_labels = node_labels[target_idx]  # Select labels based on local indices.
                
        # Apply additional filtering if provided.
        if filter_idx is not None:
            node_idx = [local_global_idx_mapping_per_type[node_type][i] for i in filter_idx]
            if node_labels is not None:
                node_labels = node_labels[filter_idx]
        
        # Determine the color and colormap.
        if targets is not None:
            c = targets
            cmap = DEFAULT_CMAP
        elif color is not None:
            c = color
            cmap = None
            handles.append(mpatches.Patch(color=color, label=node_type))
        else:
            c = None
            cmap = None

        # Extract the 3 components from the embedding.
        x = embedding[node_idx, 0]
        y = embedding[node_idx, 1]
        z_coord = embedding[node_idx, 2]
        
        scatter = ax.scatter(x, y, z_coord, s=10, alpha=0.8, c=c, cmap=cmap)
        if targets is not None:
            LOGGER.info(f"Plotting {node_type} targets")
            scatter_handles.append(scatter)
        
        # Annotate points with labels if provided.
        if node_labels is not None:
            LOGGER.info(f"Plotting {node_type} labels, n_labels: {len(node_labels)}")
            for i, label in enumerate(node_labels):
                ax.text(x[i], y[i], z_coord[i], label, fontsize=8, alpha=1)

    # Add a colorbar if targets were used for coloring.
    if targets_per_type and scatter_handles:
        cbar = plt.colorbar(scatter_handles[0], ax=ax, pad=0.1)
        cbar.set_label('Label')
        
    plt.legend(handles=handles)  
    plt.title("3D UMAP Projection of Node Embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    
    plt.savefig(save_path)
    plt.close()



if __name__ == "__main__":
    
    logger = logging.getLogger("__main__")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    
    logger = logging.getLogger("matgraphdb.pyg.data.hetero_graph")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    main()












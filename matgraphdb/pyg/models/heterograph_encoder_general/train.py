import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from torch_geometric import nn as pyg_nn

from collections import defaultdict

from matgraphdb.materials.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.data import HeteroGraphBuilder
from matgraphdb.pyg.models.heterograph_encoder_general.model import MaterialEdgePredictor
from matgraphdb.utils.colors import DEFAULT_COLORS
from matgraphdb.pyg.models.heterograph_encoder_general.trainer import (
    Trainer,
    learning_curve,
    pca_plots,
    roc_curve,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn import linear_model
from matgraphdb.pyg.models.heterograph_encoder_general.metrics import (
    LearningCurve,
    ROCCurve,
    plot_pca,
)

########################################################################################################################


CONFIG = OmegaConf.create(
    {
        "data": {
            "dataset_dir": os.path.join("data", "datasets", "MPNearHull"),
            "create_random_features": False,
            "n_material_dim": 4,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "random_link_split_args": {
                "num_val": 0.0,
                "num_test": 0.0,
                "neg_sampling_ratio": 1.0,
                "is_undirected": True,
                "edge_types": [("materials", "has", "elements"), ("materials", "has", "space_groups"), ("materials", "has", "crystal_systems")],
                "rev_edge_types": [("elements", "rev_has", "materials"), ("space_groups", "rev_has", "materials"), ("crystal_systems", "rev_has", "materials")],
            },
        },
        "model": {
            "hidden_channels": 32,
            "num_decoder_layers": 2,
            "num_conv_layers": 3,
            # "num_conv_layers": 1,
            "num_ffw_layers": 3,
            "dropout_rate": 0.0,
            "use_projections": True,
            "use_embeddings": True,
            "use_shallow_embedding_for_materials": False,
        },
        "training": {
            "training_dir": os.path.join("data", "training_runs", "heterograph_encoder_general"),
            "learning_rate": 0.001,
            "num_epochs": 1001,
            "eval_interval": 100,
            "scheduler_milestones": [4000, 20000],
        }
    }
)


####################################################################################################
####################################################################################################

print("-" * 100)
print(f"Torch version: {torch.__version__}")
print(f"PyTorch Geometric version: {pyg.__version__}")
print(
    f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")
print("-" * 100)


####################################################################################################q
# Data Loading
####################################################################################################
mdb = MPNearHull(CONFIG.data.dataset_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(mdb)
builder = HeteroGraphBuilder(mdb)

builder.add_node_type(
    "materials",
    columns=[
        "core.volume",
        "core.density",
        "core.density_atomic",
        "core.nelements",
        # "core.nsites",
        # "elasticity.g_vrh"
    ],
    # embedding_vectors=True
    # drop_null=True
)


def binning(x):
    return torch.tensor(np.log10(x), dtype=torch.float32)


builder.add_node_type(
    "elements",
    columns=[
        "atomic_mass",
        "radius_covalent",
        "radius_vanderwaals",
        # "electron_affinity"
        # "heat_specific",
    ],
    drop_null=True,
    # embedding_vectors=True,
    label_column="symbol",
    encoders={"electron_affinity": binning},
)

builder.add_node_type(
    "space_groups",
    # columns=[
    #     "atomic_mass",
    #     "radius_covalent",
    #     "radius_vanderwaals",
    #     # "electron_affinity"
    #     # "heat_specific",
    # ],
    drop_null=True,
    # embedding_vectors=True,
    label_column="spg",
)

builder.add_node_type(
    "crystal_systems",
    # columns=[
    #     "atomic_mass",
    #     "radius_covalent",
    #     "radius_vanderwaals",
    #     # "electron_affinity"
    #     # "heat_specific",
    # ],
    drop_null=True,
    # embedding_vectors=True,
    label_column="crystal_system",
)

builder.add_edge_type("element_element_neighborsByGroupPeriod")
builder.add_edge_type(
    "material_element_has",
    #   columns=["weight"]
)

builder.add_edge_type("material_spg_has")
builder.add_edge_type("material_crystalSystem_has")

def to_log(x):
    return torch.tensor(np.log10(x), dtype=torch.float32)


target_is_log = True
builder.add_target_node_property(
    "materials",
    columns=["elasticity.g_vrh"],
    filters=[
        pc.field("elasticity.g_vrh") > 0,
        pc.field("elasticity.g_vrh") < 400,
    ],
    encoders={"elasticity.g_vrh": to_log},
)
data = builder.hetero_data

data["materials"].original_x = data["materials"].x
# Set random feature vector for materials
if CONFIG.data.create_random_features:
    n_materials = data["materials"].num_nodes
    
    data["materials"].x = torch.normal(
        mean=0.0, std=1.0, size=(n_materials, CONFIG.data.n_material_dim)
    )


parent_data = T.ToUndirected()(data)
# parent_data = T.AddSelfLoops()(parent_data)

print(parent_data)
data = None

# ####################################################################################################
# # Data Splitting
# ####################################################################################################

# Split materials into train/val/test sets
n_materials = parent_data["materials"].num_nodes


# material_indices = parent_data["materials"].node_ids[torch.randperm(parent_data["materials"].node_ids.shape[0])]
node_ids = parent_data["materials"].node_ids
n_materials = len(node_ids)
material_indices = torch.randperm(n_materials)

# Hyperparameters for data splitting
train_ratio = CONFIG.data.train_ratio  # 80% for training set (including validation)
test_ratio = 1 - train_ratio  # 20% for test set (including validation)
val_ratio = CONFIG.data.val_ratio  # 10% validation ratio within each set


# Calculate sizes for each split
train_size = int(train_ratio * n_materials)  # Training set size
test_size = int(test_ratio * n_materials)  # Test set size
train_val_size = int(val_ratio * train_size)  # Training validation set size
test_val_size = int(val_ratio * test_size)  # Test validation set size

total_train_materials = material_indices[:train_size]
total_test_materials = material_indices[train_size:]


train_materials = total_train_materials[train_val_size:]
train_val_materials = total_train_materials[:train_val_size]
test_materials = total_test_materials[test_val_size:]
test_val_materials = total_test_materials[:test_val_size]



n_train = len(train_materials)
# Print percentages of each split
print("\nSplit percentages:")
print(f"Total: {n_materials}")
print(f"Train: {len(train_materials)/n_materials*100:.1f}%")
print(f"Train val: {len(train_val_materials)/n_materials*100:.1f}%")
print(f"Test: {len(test_materials)/n_materials*100:.1f}%")
print(f"Test val: {len(test_val_materials)/n_materials*100:.1f}%")
print(
    f"Total: {(len(train_materials) + len(train_val_materials) + len(test_materials) + len(test_val_materials))/n_materials*100:.1f}%\n"
)



# Create subgraphs for each split
train_dict = {"materials": train_materials}
train_val_dict = {"materials": train_val_materials}
test_dict = {"materials": test_materials}
test_val_dict = {"materials": test_val_materials}

original_train_data = parent_data.subgraph(train_dict)
original_train_data["materials"].node_ids = parent_data["materials"].node_ids[train_dict["materials"]]
original_train_val_data = parent_data.subgraph(train_val_dict)
original_train_val_data["materials"].node_ids = parent_data["materials"].node_ids[train_val_dict["materials"]]
original_test_data = parent_data.subgraph(test_dict)
original_test_data["materials"].node_ids = parent_data["materials"].node_ids[test_dict["materials"]]
original_test_val_data = parent_data.subgraph(test_val_dict)
original_test_val_data["materials"].node_ids = parent_data["materials"].node_ids[test_val_dict["materials"]]

print(original_train_data["materials"].node_ids)
print(f"Train materials: {len(train_materials)}")
print(f"Train val materials: {len(train_val_materials)}")
print(f"Test materials: {len(test_materials)}")
print(f"Test val materials: {len(test_val_materials)}")

# Reduce the target values for each split. Also record the the y_node_ids and the index of the split.
y_id_map = {int(y_id): float(y) for y_id, y in zip(parent_data['materials'].y_index, parent_data['materials'].y)}
for i, data in enumerate([original_train_data, original_train_val_data, original_test_data, original_test_val_data]):        
    y_vals=[]
    ids=[]
    node_ids=[]
    for i, node_id in enumerate(data["materials"].node_ids):
        if int(node_id) in y_id_map:
            y_vals.append(y_id_map[int(node_id)])
            node_ids.append(node_id)
            ids.append(i)
    data["materials"].y = torch.tensor(y_vals)
    data["materials"].y_node_ids = torch.tensor(node_ids)
    data["materials"].y_split_index = torch.tensor(ids)
    

data = None
builder = None


# omefga config cannot handle list of tuples. Must convert back to list of tuples
# print(CONFIG.data.random_link_split_args)

random_link_split_args = OmegaConf.to_container(CONFIG.data.random_link_split_args, resolve=True)

for i,edge_type in enumerate(random_link_split_args['edge_types']):
    random_link_split_args['edge_types'][i] = tuple(edge_type)
    
for i,edge_type in enumerate(random_link_split_args['rev_edge_types']):
    random_link_split_args['rev_edge_types'][i] = tuple(edge_type)
    

# Perform a link-level split into training, validation, and test edges:
train_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_train_data)
train_val_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_train_val_data)
test_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_test_data)
test_val_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_test_val_data)

# print("Train data:")
# print(train_data)
# print("Train val data:")
# print(train_val_data)
# print("Test data:")
# print(test_data)
# print("Test val data:")
# print(test_val_data)


split_data = {
    "train": train_data,
    "train_val": train_val_data,
    "test": test_data,
    "test_val": test_val_data,
}

# Random link split does not add edge labels and index to the reverse edges. Must add them manually.
for split_label, data in split_data.items():
    data['elements', 'rev_has', 'materials'].edge_label_index = data['materials', 'has', 'elements'].edge_label_index[[1,0]]
    data['elements', 'rev_has', 'materials'].edge_label = data['materials', 'has', 'elements'].edge_label
    data['space_groups', 'rev_has', 'materials'].edge_label_index = data['materials', 'has', 'space_groups'].edge_label_index[[1,0]]
    data['space_groups', 'rev_has', 'materials'].edge_label = data['materials', 'has', 'space_groups'].edge_label
    data['crystal_systems', 'rev_has', 'materials'].edge_label_index = data['materials', 'has', 'crystal_systems'].edge_label_index[[1,0]]
    data['crystal_systems', 'rev_has', 'materials'].edge_label = data['materials', 'has', 'crystal_systems'].edge_label
train_data = train_data.to(device)
train_val_data = train_val_data.to(device)
test_data = test_data.to(device)
test_val_data = test_val_data.to(device)


for split_name, data in split_data.items():
    print(f"Split: {split_name}")
    print(data)
    print("-" * 100)

print("-" * 100)
print(f"Max memory allocated: {torch.cuda.max_memory_allocated()}")

# ####################################################################################################
# # Model
# ####################################################################################################

model = MaterialEdgePredictor(
    hidden_channels=CONFIG.model.hidden_channels,
    data=parent_data,
    decoder_kwargs={
        "num_layers": CONFIG.model.num_decoder_layers,
        "dropout": CONFIG.model.dropout_rate,
    },
    encoder_kwargs={
        "num_conv_layers": CONFIG.model.num_conv_layers,
        "num_ffw_layers": CONFIG.model.num_ffw_layers,
        "dropout": CONFIG.model.dropout_rate,
    },
    use_projections=CONFIG.model.use_projections,
    use_embeddings=CONFIG.model.use_embeddings,
    use_shallow_embedding_for_materials=CONFIG.model.use_shallow_embedding_for_materials,
).to(device)
# print(model)


# ####################################################################################################
# # Training
# ####################################################################################################


def weighted_binary_cross_entropy(pred, target, weights=None):
    if weights is None:
        weights = 1.0
        y = F.binary_cross_entropy(pred, target)
        return y
    else:
        weights = weights[target.long()]
        return F.binary_cross_entropy(pred, target, weight=weights)


optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)

scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones=CONFIG.training.scheduler_milestones, gamma=0.1, verbose=False
)
results = {
        "train": {"mae": [],"epochs": []},
        "train_val": {"mae": [],"epochs": []},
        "test": {"mae": [],"epochs": []},
        "test_val": {"mae": [],"epochs": []},
    }
results_original = {
        "train": {"mae": [],"epochs": []},
        "train_val": {"mae": [],"epochs": []},
        "test": {"mae": [],"epochs": []},
        "test_val": {"mae": [],"epochs": []},
    }

def train_step(data_batch):
    model.train()
    optimizer.zero_grad()

    loss_dict = {}
    total_loss = 0
    pred_edge_dict = model(data_batch)
    for edge_type, key in model.edge_types_to_decoder_keys.items():
        src, rel, dst = edge_type
        pred = pred_edge_dict[key]
        
        if not hasattr(data_batch[src, dst], "edge_label"):
            continue
        
        target = data_batch[src,rel,dst].edge_label
        loss = F.binary_cross_entropy(pred, target)
        total_loss += loss
        loss_dict[key] = loss
    total_loss.backward()
    optimizer.step()

    return float(total_loss.cpu()), loss_dict


@torch.no_grad()
def validation_step(data_batch):
    model.eval()
    
    loss_dict = {}
    prediction_dict = {}
    total_loss = 0

    pred_edge_dict = model(data_batch)
    for edge_type, key in model.edge_types_to_decoder_keys.items():
        src, rel, dst = edge_type
        pred = pred_edge_dict[key]
        
        if not hasattr(data_batch[src, dst], "edge_label"):
            continue
        
        target = data_batch[src,rel,dst].edge_label
        loss = F.binary_cross_entropy(pred, target)
        total_loss += loss
        loss_dict[key] = float(loss.cpu())
        if key not in prediction_dict:
            prediction_dict[key] = {}
        prediction_dict[key]['predictions'] = pred
        prediction_dict[key]['targets'] = target
    return float(total_loss.cpu()), loss_dict, prediction_dict

@torch.no_grad()
def regression_eval(data_batch_per_split):
    model.eval()
    z_material_per_split = {}
    z_original_per_split = {}
    y_per_split = {}
    tmp_str=''
    node_type='materials'
    for split_name, data_batch in data_batch_per_split.items():
        z_dict = model.encode(data_batch)
        

        y_split_index = data_batch[node_type].y_split_index
        y = data_batch[node_type].y
        z = z_dict[node_type][y_split_index]
        z_original = data_batch[node_type].original_x[y_split_index]

        z_material_per_split[split_name] = z.cpu().numpy()
        y_per_split[split_name] = y.cpu().numpy()
        z_original_per_split[split_name] = z_original.cpu().numpy()
        
        tmp_str += f'|{split_name}: {len(z)}|'
    print(tmp_str)

    
    reg = linear_model.LinearRegression()
    reg.fit(z_material_per_split['train'], y_per_split['train'])
    
    reg_original = linear_model.LinearRegression()
    reg_original.fit(z_original_per_split['train'], y_per_split['train'])
    
    test_splits = ['train', 'train_val', 'test', 'test_val']
    tmp_str = ''
    for test_split_name in test_splits:
        y_pred = reg.predict(z_material_per_split[test_split_name])
        y_real = y_per_split[test_split_name]
        
        y_pred = np.array([10**value for value in y_pred])
        y_real = np.array([10**value for value in y_real])
        
        rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
        mae = np.mean(np.abs(y_pred - y_real))
        results[test_split_name]["mae"].append(mae)
        tmp_str += f'|{test_split_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}|'
    print(tmp_str)
    
    test_splits = ['train', 'train_val', 'test', 'test_val']
    tmp_str = ''
    for test_split_name in test_splits:
        y_pred = reg_original.predict(z_original_per_split[test_split_name])
        y_real = y_per_split[test_split_name]
        
        y_pred = np.array([10**value for value in y_pred])
        y_real = np.array([10**value for value in y_real])
        
        rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
        mae = np.mean(np.abs(y_pred - y_real))
        results_original[test_split_name]["mae"].append(mae)
        
        tmp_str += f'|{test_split_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}|'
    print(tmp_str)
    
def eval_metrics(preds, targets, **kwargs):
    # Calculate metrics
    pred_binary = (preds > 0.5).float()
    accuracy = (pred_binary == targets).float().mean().cpu()

    # Calculate precision, recall, f1
    true_positives = (pred_binary * targets).sum().cpu()
    true_negatives = ((1 - pred_binary) * (1 - targets)).sum().cpu()
    false_positives = (pred_binary * (1 - targets)).sum().cpu()
    false_negatives = ((1 - pred_binary) * targets).sum().cpu()

    predicted_positives = pred_binary.sum().cpu()
    actual_positives = targets.sum().cpu()

    precision = true_positives / (predicted_positives + 1e-10)
    recall = true_positives / (actual_positives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    auc_score = roc_auc_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score),
        "true_positives": float(true_positives),
        "true_negatives": float(true_negatives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "predicted_positives": float(predicted_positives),
        "actual_positives": float(actual_positives),
    }

    return results

def calculate_embeddings(data_batch):
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(data_batch)
        
        for key, value in z_dict.items():
            z_dict[key] = value.cpu().detach().numpy()
    return z_dict


def roc_curve(metrics_per_split_dict, epoch_save_path=None, total_save_path=None):
    roc_curve_plot = ROCCurve()

    for split_label, metrics_dict in metrics_per_split_dict.items():
        pred = metrics_dict['current_predictions']
        target = metrics_dict['current_targets']

        # Add main model curve
        roc_curve_plot.add_curve(
            pred, target, split_label, split_label, is_baseline=False
        )

    roc_curve_plot.plot()
    if epoch_save_path is not None:
        roc_curve_plot.save(epoch_save_path)
        
    if total_save_path is not None:
        roc_curve_plot.save(total_save_path)
    roc_curve_plot.close()

def learning_curve(metrics_per_split, metric_name, epoch_save_path=None, total_save_path=None):
    learning_curve = LearningCurve()

    for split_label, metrics_dict in metrics_per_split.items():
        epochs = metrics_dict["epochs"]
        loss_values = metrics_dict[metric_name]

        # Add main model curve
        learning_curve.add_curve(
            epochs, loss_values, split_label, split_label, is_baseline=False
        )

    # Generate and save plots
    learning_curve.plot()
    learning_curve.set_ylabel(metric_name)
    if epoch_save_path is not None:
        learning_curve.save(epoch_save_path)
    if total_save_path is not None:
        learning_curve.save(total_save_path)
    learning_curve.close()

def pca_plots(embeddings_per_node_type, 
              n_nodes_per_node_type, 
              node_labels_per_node_type,
              pca_dir):

    os.makedirs(pca_dir, exist_ok=True)
    # 3. Combine embeddings only for the selected node types
    z_all = np.concatenate([embeddings for embeddings in embeddings_per_node_type.values()], axis=0)
    
    plot_pca(
        z_all,
        save_dir=pca_dir,
        save_name=f'embeddings_pca_grid.png',
        n_nodes_per_type=n_nodes_per_node_type,
        node_labels_per_type=node_labels_per_node_type,
        n_components=2,
        figsize=(10, 8),
        close=True
    )

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

runs_dir = os.path.join(CONFIG.training.training_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)
n_runs = len(os.listdir(runs_dir))
run_dir = os.path.join(runs_dir, f"run_{n_runs}")
os.makedirs(run_dir, exist_ok=True)

epochs_dir = os.path.join(run_dir, "epochs")
os.makedirs(epochs_dir, exist_ok=True)

run_learning_curve_dir = os.path.join(run_dir, "learning_curves")
os.makedirs(run_learning_curve_dir, exist_ok=True)

run_roc_curve_dir = os.path.join(run_dir, "roc_curves")
os.makedirs(run_roc_curve_dir, exist_ok=True)

run_pca_dir = os.path.join(run_dir, "pca")
os.makedirs(run_pca_dir, exist_ok=True)



loss, loss_dict, rel_prediction_dict = validation_step(test_val_data)
rel_names =  []
split_names = list(split_data.keys())
for rel_name, prediction_dict in rel_prediction_dict.items():
    preds = prediction_dict['predictions']
    targets = prediction_dict['targets']
    metrics = eval_metrics(preds, targets)
    metric_names=list(metrics.keys())
    rel_names.append(rel_name)
    
metric_names.append('loss')
metric_names.append('epochs')
values_to_record = copy.deepcopy(metric_names)
values_to_record.append('current_predictions')
values_to_record.append('current_targets')
values_to_record.append('current_loss')
metrics_per_rel_per_split = defaultdict(
    lambda: defaultdict(
        lambda: {value_name: [] for value_name in values_to_record}
    )
)
node_types = test_val_data.metadata()[0]
n_nodes_per_split_per_node_type = defaultdict(
    lambda: defaultdict(
        lambda: {node_type: [] for node_type in node_types}
    )
)
node_labels_per_split_per_node_type = defaultdict(
    lambda: defaultdict(
        lambda: {node_type: [] for node_type in node_types if node_type != 'materials'}
    )
)


for split_name, data_batch in split_data.items():
    for node_type in data_batch.metadata()[0]:
        n_nodes_per_split_per_node_type[split_name][node_type] = data_batch[node_type].num_nodes
        if node_type != 'materials':
            node_labels_per_split_per_node_type[split_name][node_type] = data_batch[node_type].labels



for epoch in range(CONFIG.training.num_epochs):
    # print(f"Epoch: {epoch}")
    total_loss, loss_dict = train_step(train_data)
    scheduler.step()
    if epoch % CONFIG.training.eval_interval == 0:
        current_epoch = epoch
        epoch_dir = os.path.join(epochs_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        eval_str = f"Epoch: {epoch} :"
        loss_per_rel_str = ""
        
 
        embeddings_per_split_per_node_type= {}
        for split_name, data_batch in split_data.items():
            loss, loss_dict, rel_prediction_dict = validation_step(data_batch)
            
            eval_str += f" |{split_name}_loss: {loss} "
            
            embeddings_dict = calculate_embeddings(data_batch)
            embeddings_per_split_per_node_type[split_name] = embeddings_dict
            
            
            for rel_name in rel_names:
                prediction_dict = rel_prediction_dict[rel_name]
                rel_loss=loss_dict[rel_name]
                preds = prediction_dict['predictions']
                targets = prediction_dict['targets']
                metrics = eval_metrics(preds, targets)

                metrics_per_rel_per_split[rel_name][split_name]['current_loss'] = rel_loss
                metrics_per_rel_per_split[rel_name][split_name]['current_predictions'] = preds
                metrics_per_rel_per_split[rel_name][split_name]['current_targets'] = targets
    
                metrics_per_rel_per_split[rel_name][split_name]['loss'].append(rel_loss)
                metrics_per_rel_per_split[rel_name][split_name]['epochs'].append(epoch)
                for metric_name, metric_value in metrics.items():
                    metrics_per_rel_per_split[rel_name][split_name][metric_name].append(metric_value)        
                    
                    
            results[split_name]["epochs"].append(epoch)
            results_original[split_name]["epochs"].append(epoch)
            
            
        eval_str += f"|"
        print(eval_str)
        
        for rel_name in rel_names:
            loss_list = []
            for split_name in split_names:
                loss = metrics_per_rel_per_split[rel_name][split_name]['current_loss']
                loss_list.append(str(round(loss, 4)))
            loss_per_rel_str = f"|{rel_name}: {':'.join(loss_list)}|"
            print(f"        {loss_per_rel_str}")
        
        
        regression_eval(split_data)

        # epoch_roc_curve_dir = os.path.join(epoch_dir, "roc_curves")
        # os.makedirs(epoch_roc_curve_dir, exist_ok=True)
        # for rel_name in rel_names:
        #     metrics_per_split = metrics_per_rel_per_split[rel_name]
        #     epoch_save_path=os.path.join(epoch_roc_curve_dir, f"{rel_name}_roc_curve.png")
        #     total_save_path=os.path.join(run_roc_curve_dir, f"{rel_name}_roc_curve.png")
        #     roc_curve(metrics_per_split, epoch_save_path, total_save_path)
        
        # for metric_name in metric_names:
        #     if metric_name == 'epochs':
        #         continue
        #     for rel_name in rel_names:
        #         metrics_per_split = metrics_per_rel_per_split[rel_name]
        #         rel_epoch_learning_curve_dir = os.path.join(epoch_dir, rel_name)
        #         rel_run_learning_curve_dir = os.path.join(run_learning_curve_dir, rel_name)
        #         os.makedirs(rel_run_learning_curve_dir, exist_ok=True)
        #         os.makedirs(rel_epoch_learning_curve_dir, exist_ok=True)
        #         epoch_save_path=os.path.join(rel_epoch_learning_curve_dir, f"{metric_name}_learning_curve.png")
        #         total_save_path=os.path.join(rel_run_learning_curve_dir, f"{metric_name}_learning_curve.png")
        #         learning_curve(metrics_per_split, metric_name, epoch_save_path, total_save_path)

        # for split_name, embeddings_per_node_type in embeddings_per_split_per_node_type.items():
        #     n_nodes_per_node_type = n_nodes_per_split_per_node_type[split_name]
        #     node_labels_per_type = node_labels_per_split_per_node_type[split_name]
        #     pca_plots(
        #         embeddings_per_node_type, 
        #         n_nodes_per_node_type, 
        #         node_labels_per_type,
        #         pca_dir=os.path.join(run_pca_dir, split_name))



plot_learning_curves(results, save_path=os.path.join(run_learning_curve_dir, "learning_curves.png"))
plot_learning_curves(results_original, save_path=os.path.join(run_learning_curve_dir, "learning_curves_original.png"))

epoch_roc_curve_dir = os.path.join(epoch_dir, "roc_curves")
os.makedirs(epoch_roc_curve_dir, exist_ok=True)
for rel_name in rel_names:
    metrics_per_split = metrics_per_rel_per_split[rel_name]
    # epoch_save_path=os.path.join(epoch_roc_curve_dir, f"{rel_name}_roc_curve.png")
    total_save_path=os.path.join(run_roc_curve_dir, f"{rel_name}_roc_curve.png")
    roc_curve(metrics_per_split, total_save_path=total_save_path)

for metric_name in metric_names:
    if metric_name == 'epochs':
        continue
    for rel_name in rel_names:
        metrics_per_split = metrics_per_rel_per_split[rel_name]
        rel_epoch_learning_curve_dir = os.path.join(epoch_dir, rel_name)
        rel_run_learning_curve_dir = os.path.join(run_learning_curve_dir, rel_name)
        os.makedirs(rel_run_learning_curve_dir, exist_ok=True)
        os.makedirs(rel_epoch_learning_curve_dir, exist_ok=True)
        # epoch_save_path=os.path.join(rel_epoch_learning_curve_dir, f"{metric_name}_learning_curve.png")
        total_save_path=os.path.join(rel_run_learning_curve_dir, f"{metric_name}_learning_curve.png")
        learning_curve(metrics_per_split, metric_name, total_save_path=total_save_path)

for split_name, embeddings_per_node_type in embeddings_per_split_per_node_type.items():
    n_nodes_per_node_type = n_nodes_per_split_per_node_type[split_name]
    node_labels_per_type = node_labels_per_split_per_node_type[split_name]
    pca_plots(
        embeddings_per_node_type, 
        n_nodes_per_node_type, 
        node_labels_per_type,
        pca_dir=os.path.join(run_pca_dir, split_name))
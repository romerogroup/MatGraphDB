import os
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import  SAGEConv, GCNConv, GraphConv, to_hetero, GraphSAGE,to_hetero_with_bases
import seaborn as sns
# Ensure matplotlib uses a backend that supports SVG
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx

from matgraphdb.graph_kit.pyg.metrics import ClassificationMetrics,RegressionMetrics


from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager

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
        'CRYSTAL_SYSTEM':[
        ],
        'SPACE_GROUP':[
        ],
        'ELEMENT':[
            # 'abundance_universe',
            # 'abundance_solar',
            # 'abundance_meteor',
            # 'abundance_crust',
            # 'abundance_ocean',
            # 'abundance_human',
            # 'atomic_mass',
            # 'atomic_number',
            # 'block',
            # 'boiling_point',
            # 'critical_pressure',
            # 'critical_temperature',
            # 'density_stp',
            # 'electron_affinity',
            # 'electronegativity_pauling',
            # 'extended_group',
            # 'heat_specific',
            # 'heat_vaporization',
            # 'heat_fusion',
            # 'heat_molar',
            # 'magnetic_susceptibility_mass',
            # 'magnetic_susceptibility_molar',
            # 'magnetic_susceptibility_volume',
            # 'melting_point',
            # 'molar_volume',
            # 'neutron_cross_section',
            # 'neutron_mass_absorption',
            # 'period',
            # 'radius_calculated',
            # 'radius_empirical',
            # 'radius_covalent',
            # 'radius_vanderwaals',
            # 'refractive_index',
            # 'speed_of_sound',
            # 'conductivity_electric',
            # 'electrical_resistivity',
            # 'modulus_bulk',
            # 'modulus_shear',
            # 'modulus_young',
            # 'poisson_ratio',
            # 'coefficient_of_linear_thermal_expansion',
            # 'hardness_vickers',
            # 'hardness_brinell',
            # 'hardness_mohs',
            # 'superconduction_temperature',
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

generator=DataGenerator()
relationship_properties={
    'ELEMENT-CAN_OCCUR-CHEMENV':[
        'weight',
        ],
    'ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT':[
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
    'MATERIAL-HAS-SPACE_GROUP':[
        'weight',
    ],
    'MATERIAL-HAS-CRYSTAL_SYSTEM':[
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

generator.add_node_type(node_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/nodes/SPACE_GROUP.parquet',
                        feature_columns=node_properties['SPACE_GROUP'],
                        target_columns=[],
                        filter={})
generator.add_node_type(node_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/nodes/CRYSTAL_SYSTEM.parquet',
                        feature_columns=node_properties['CRYSTAL_SYSTEM'],
                        target_columns=[],
                        filter={})
print(generator.hetero_data)



generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV.parquet',
                        feature_columns=relationship_properties['CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT.parquet',
                        feature_columns=relationship_properties['ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-CHEMENV.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-CHEMENV'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-ELEMENT.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-ELEMENT'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/ELEMENT-CAN_OCCUR-CHEMENV.parquet',
                        feature_columns=relationship_properties['ELEMENT-CAN_OCCUR-CHEMENV'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-SPACE_GROUP.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-SPACE_GROUP'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-CRYSTAL_SYSTEM.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-CRYSTAL_SYSTEM'],
                        target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)
print(generator.hetero_data)



# Assuming hetero_data is already defined
hetero_data = generator.hetero_data

# Step 1: Select 10 material nodes (randomly or based on some criteria)
# For reproducibility, we'll use a fixed seed
torch.manual_seed(42)
material_node_indices = torch.randperm(hetero_data['MATERIAL'].num_nodes)[:10]

# Step 2: Identify connected nodes
def get_connected_nodes(hetero_data, edge_type, src_indices=None, dst_indices=None):
    if edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        # If src_indices are specified, filter edges
        if src_indices is not None:
            src_mask = torch.isin(edge_index[0], src_indices)
            edge_index = edge_index[:, src_mask]
        # If dst_indices are specified, filter edges
        if dst_indices is not None:
            dst_mask = torch.isin(edge_index[1], dst_indices)
            edge_index = edge_index[:, dst_mask]
        # Get unique nodes connected to the specified indices
        src_nodes = edge_index[0].unique()
        dst_nodes = edge_index[1].unique()
        return src_nodes, dst_nodes
    else:
        print(f"Edge type {edge_type} not found.")
        return torch.tensor([]), torch.tensor([])

# Get connected nodes for different types
_, element_nodes = get_connected_nodes(
    hetero_data, ('MATERIAL', 'HAS', 'ELEMENT'), src_indices=material_node_indices)

_, chemenv_nodes = get_connected_nodes(
    hetero_data, ('MATERIAL', 'HAS', 'CHEMENV'), src_indices=material_node_indices)

_, space_group_nodes = get_connected_nodes(
    hetero_data, ('MATERIAL', 'HAS', 'SPACE_GROUP'), src_indices=material_node_indices)

_, crystal_system_nodes = get_connected_nodes(
    hetero_data, ('MATERIAL', 'HAS', 'CRYSTAL_SYSTEM'), src_indices=material_node_indices)

# Step 3: Build the NetworkX graph
G = nx.Graph()

# Function to add nodes
def add_nodes(G, node_type, node_indices):
    for idx in node_indices.tolist():
        G.add_node((node_type, idx), node_type=node_type)

# Add all relevant nodes
add_nodes(G, 'MATERIAL', material_node_indices)
add_nodes(G, 'ELEMENT', element_nodes)
add_nodes(G, 'CHEMENV', chemenv_nodes)
add_nodes(G, 'SPACE_GROUP', space_group_nodes)
add_nodes(G, 'CRYSTAL_SYSTEM', crystal_system_nodes)

# Step 4: Add edges with attributes
def add_filtered_edges(G, hetero_data, edge_type, src_indices=None, dst_indices=None, attribute_key='weight'):
    """
    Adds edges to the graph with the specified attribute.

    Parameters:
    - G: NetworkX graph
    - hetero_data: HeteroData object
    - edge_type: Tuple specifying the edge type
    - src_indices: Source node indices to filter
    - dst_indices: Destination node indices to filter
    - attribute_key: Key of the edge attribute to use for coloring
    """
    if edge_type in hetero_data.edge_types:
        edge_data = hetero_data[edge_type]
        print(dir(edge_data))
        edge_index = edge_data.edge_index

        # Filter edges based on source and destination indices
        if src_indices is not None:
            src_mask = torch.isin(edge_index[0], src_indices)
        else:
            src_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        
        if dst_indices is not None:
            dst_mask = torch.isin(edge_index[1], dst_indices)
        else:
            dst_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        
        mask = src_mask & dst_mask
        filtered_edge_index = edge_index[:, mask]
        
        # Retrieve the edge attribute
        edge_attr = edge_data.edge_attr[mask, :].cpu().numpy()[:, 0]


        # if attribute_key in edge_data:
        #     edge_attr = edge_data[attribute_key][mask].cpu().numpy()
        # else:
        #     # Assign a default value if the attribute is missing
        #     edge_attr = np.ones(filtered_edge_index.size(1))
        #     print(f"Attribute '{attribute_key}' not found for edge type {edge_type}. Using default value 1.0.")
        
        # Add edges with the attribute
        for i in range(filtered_edge_index.size(1)):
            src = (edge_type[0], filtered_edge_index[0, i].item())
            dst = (edge_type[2], filtered_edge_index[1, i].item())
            G.add_edge(src, dst, relation=edge_type[1], **{attribute_key: edge_attr[i]})
    else:
        print(f"Edge type {edge_type} not found.")

# Define the attribute key you want to use for coloring
attribute_key = 'weight'  # Replace with your actual edge attribute key

# Add edges for all relevant edge types
edge_types = [
    ('MATERIAL', 'HAS', 'ELEMENT'),
    ('MATERIAL', 'HAS', 'CHEMENV'),
    ('MATERIAL', 'HAS', 'SPACE_GROUP'),
    ('MATERIAL', 'HAS', 'CRYSTAL_SYSTEM'),
    ('ELEMENT', 'CAN_OCCUR', 'CHEMENV')  # Optional
]

for edge_type in edge_types:
    src = edge_type[0] if edge_type[0] == 'MATERIAL' else 'ELEMENT'
    dst = edge_type[2] if edge_type[2] in ['ELEMENT', 'CHEMENV', 'SPACE_GROUP', 'CRYSTAL_SYSTEM'] else 'CHEMENV'
    src_indices = material_node_indices if edge_type[0] == 'MATERIAL' else element_nodes
    dst_indices = (
        element_nodes if edge_type[2] == 'ELEMENT' else
        chemenv_nodes if edge_type[2] == 'CHEMENV' else
        space_group_nodes if edge_type[2] == 'SPACE_GROUP' else
        crystal_system_nodes
    )
    add_filtered_edges(
        G, hetero_data, edge_type,
        src_indices=src_indices, dst_indices=dst_indices,
        attribute_key=attribute_key
    )

# Step 5: Visualize the graph with edge color scale
# Define node colors based on node type
color_map = {
    'MATERIAL': '#2ca02c',       # Green
    'ELEMENT': '#1f77b4',        # Blue
    'CHEMENV': '#ff7f0e',        # Orange
    'SPACE_GROUP': '#d62728',    # Red
    'CRYSTAL_SYSTEM': '#9467bd', # Purple
}

node_colors = [color_map.get(data['node_type'], '#7f7f7f') for _, data in G.nodes(data=True)]
# Define the attribute key you want to normalize
attribute_key = 'weight'

# Collect edge attribute values
edge_attrs = [data.get(attribute_key, 1.0) for _, _, data in G.edges(data=True)]

# Create a pandas Series
edge_attrs_series = pd.Series(edge_attrs)

# Calculate median and IQR
median = edge_attrs_series.median()
q1 = edge_attrs_series.quantile(0.25)
q3 = edge_attrs_series.quantile(0.75)
iqr = q3 - q1

# Normalize
if iqr == 0:
    normalized_edge_attrs = [0] * len(edge_attrs)
else:
    normalized_edge_attrs = ((edge_attrs_series - median) / iqr).tolist()

# Percentile-based clipping
lower_percentile = edge_attrs_series.quantile(0.05)
upper_percentile = edge_attrs_series.quantile(0.95)

# Calculate corresponding normalized values
normalized_lower = (lower_percentile - median) / iqr
normalized_upper = (upper_percentile - median) / iqr

# Clip the normalized edge attributes
clipped_edge_attrs = np.clip(normalized_edge_attrs, normalized_lower, normalized_upper)

# Normalize edge attributes for colormap using clipped bounds
norm = mpl.colors.Normalize(vmin=normalized_lower, vmax=normalized_upper)
cmap = mpl.cm.viridis  # Choose appropriate colormap

# Generate edge colors based on the normalized and clipped values
edge_colors = [cmap(norm(attr)) for attr in clipped_edge_attrs]

# (Optional) Assign the clipped normalized values back to the graph edges
for (u, v, data), norm_val in zip(G.edges(data=True), clipped_edge_attrs):
    data['normalized_weight'] = norm_val

# Create labels for nodes (optional)
labels = {}
for node in G.nodes():
    node_type, idx = node
    if node_type == 'ELEMENT':
        element_symbols = hetero_data['ELEMENT'].names
        labels[node] = element_symbols[idx] if idx < len(element_symbols) else f"El {idx}"
    elif node_type == 'CHEMENV':
        chemenv_names = hetero_data['CHEMENV'].names
        labels[node] = chemenv_names[idx] if idx < len(chemenv_names) else f"CE {idx}"
    elif node_type == 'MATERIAL':
        material_names = hetero_data['MATERIAL'].names
        labels[node] = material_names[idx] if idx < len(material_names) else f"Mat {idx}"
    elif node_type == 'SPACE_GROUP':
        spg_names = hetero_data['SPACE_GROUP'].names
        labels[node] = spg_names[idx] if idx < len(spg_names) else f"SG {idx}"
    elif node_type == 'CRYSTAL_SYSTEM':
        cg_names = hetero_data['CRYSTAL_SYSTEM'].names
        labels[node] = cg_names[idx] if idx < len(cg_names) else f"CS {idx}"
    else:
        labels[node] = f"{node_type} {idx}"

# (Optional) Plot the distribution of clipped normalized edge attributes
sns.histplot(clipped_edge_attrs, bins=30, kde=True)
plt.title('Distribution of Clipped Normalized Edge Attributes')
plt.xlabel('Normalized Value')
plt.ylabel('Frequency')
plt.show()

# Create a figure with specified size
plt.figure(figsize=(20, 15))

# Generate the layout for the graph
# pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)

# Draw edges with colors
nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    alpha=0.7,
    width=2
)

# Draw labels
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# Create a scalar mappable for the colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib

# # Add colorbar to the plot
# cbar = plt.colorbar(sm, shrink=0.5, pad=0.05)
# cbar.set_label(f'Normalized {attribute_key}', rotation=270, labelpad=15)

# Remove axes for better visualization
plt.axis('off')

# Adjust layout to make space for the colorbar
plt.tight_layout()

# Save the graph as an SVG file
plt.savefig('examples/scripts/subgraph_visualization.svg', format='svg', dpi=1200)
plt.close()

print("Subgraph visualization saved as 'subgraph_visualization.svg'")
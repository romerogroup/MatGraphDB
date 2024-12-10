import os
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving the plot
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager
from matgraphdb.utils import PKG_DIR, GRAPH_DIR
from matgraphdb.graph_kit.pyg.algo import feature_propagation

if __name__ == "__main__":
    element_file = os.path.join(PKG_DIR, 'utils', 'interim_periodic_table_values.csv')
    df = pd.read_csv(element_file)

    manager = GraphManager(graph_dir=os.path.join(GRAPH_DIR, 'main'))

    manager.nodes.get_element_nodes(base_element_csv='interim_periodic_table_values.csv',
                                    from_scratch=True)

    generator = DataGenerator()

    property_names = ['abundance_universe', 'abundance_solar', 'abundance_meteor', 'abundance_crust',
                      'abundance_ocean', 'abundance_human', 'boiling_point', 'critical_pressure',
                      'critical_temperature', 'density_stp', 'conductivity_thermal', 'electron_affinity',
                      'electronegativity_pauling', 'heat_specific', 'heat_vaporization', 'heat_fusion',
                      'heat_molar', 'magnetic_susceptibility_mass', 'magnetic_susceptibility_molar',
                      'magnetic_susceptibility_volume', 'melting_point', 'molar_volume', 'neutron_cross_section',
                      'neutron_mass_absorption', 'radius_calculated', 'radius_empirical', 'radius_covalent',
                      'radius_vanderwaals', 'refractive_index', 'speed_of_sound', 'conductivity_electric',
                      'electrical_resistivity', 'modulus_bulk', 'modulus_shear', 'modulus_young',
                      'poisson_ratio', 'coefficient_of_linear_thermal_expansion', 'hardness_vickers',
                      'hardness_brinell', 'hardness_mohs', 'superconduction_temperature']

    generator.add_node_type(node_path=os.path.join(manager.node_dir, 'ELEMENT.parquet'),
                            feature_columns=property_names,
                            keep_nan=True)

    generator.add_edge_type(edge_path=os.path.join(manager.relationship_dir,
                                                   'ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT.parquet'))

    data = generator.homo_data

    properties_after = feature_propagation(data=data)
    df_final = pd.DataFrame(properties_after, columns=property_names)
    df_final.to_csv(os.path.join('examples', 'scripts', 'imputed_periodic_table_values.csv'))
    print(df_final.head())

    # Read element symbols
    element_file = os.path.join(PKG_DIR, 'utils', 'interim_periodic_table_values.csv')
    df_elements = pd.read_csv(element_file)
    element_symbols = df_elements['symbol'].tolist()

    # Convert data to NetworkX graph
    G = to_networkx(data)

    # Assign properties and symbols to nodes
    for idx, node in enumerate(G.nodes()):
        for i, prop in enumerate(property_names):
            G.nodes[node][prop] = properties_after[idx, i]
        G.nodes[node]['symbol'] = element_symbols[idx]

    # Select property to color by
    property_name = 'boiling_point'

    node_color = [G.nodes[node][property_name] for node in G.nodes()]

    # Create labels
    labels = {node: G.nodes[node]['symbol'] for node in G.nodes()}

    # Create positions
    positions = nx.spring_layout(G,k=0.5, seed=42)

    # Plot the graph
    plt.figure(figsize=(12, 12))

    # Draw nodes and capture the returned PathCollection
    nodes = nx.draw_networkx_nodes(G, pos=positions, node_color=node_color,
                                   cmap=plt.cm.viridis, node_size=300)

    # Draw edges
    nx.draw_networkx_edges(G, pos=positions)

    # Draw labels
    nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=8)

    # Create colorbar using the nodes PathCollection
    cbar = plt.colorbar(nodes, label=property_name.replace('_', ' ').title())

    # Turn off axis
    plt.axis('off')

    # Set title
    plt.title('Graph Colored by {}'.format(property_name.replace('_', ' ').title()))

    os.makedirs('examples/scripts/properties', exist_ok=True)
    plt.savefig(f'examples/scripts/properties/{property_name}.png')

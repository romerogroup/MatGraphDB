import os
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving the plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager
from matgraphdb.utils import PKG_DIR, GRAPH_DIR
from matgraphdb.graph_kit.pyg.algo import feature_propagation

if __name__ == "__main__":
    element_file = os.path.join(PKG_DIR, 'utils', 'interim_periodic_table_values.csv')
    df_elements = pd.read_csv(element_file)

    # Ensure 'group' and 'period' columns are present
    if 'group' not in df_elements.columns or 'period' not in df_elements.columns:
        raise ValueError("The element data must include 'group' and 'period' columns.")

    # Property names
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

    # Initialize GraphManager and DataGenerator
    manager = GraphManager(graph_dir=os.path.join(GRAPH_DIR, 'main'))

    manager.nodes.get_element_nodes(base_element_csv='interim_periodic_table_values.csv',
                                    from_scratch=True)

    generator = DataGenerator()

    generator.add_node_type(node_path=os.path.join(manager.node_dir, 'ELEMENT.parquet'),
                            feature_columns=property_names,
                            keep_nan=True)

    generator.add_edge_type(edge_path=os.path.join(manager.relationship_dir,
                                                   'ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT.parquet'))

    data = generator.homo_data

    # Feature propagation
    properties_after = feature_propagation(data=data)
    df_final = pd.DataFrame(properties_after, columns=property_names)
    df_final.to_csv(os.path.join('examples', 'scripts', 'imputed_periodic_table_values.csv'))
    print(df_final.head())

    # Ensure element count matches
    assert len(df_elements) == data.num_nodes

    # Element symbols and properties
    element_symbols = df_elements['symbol'].tolist()
    element_groups = df_elements['extended_group'].tolist()
    element_periods = df_elements['period'].tolist()

    # Choose property to color by
    property_name = 'speed_of_sound'
    if property_name not in property_names:
        raise ValueError(f"Property '{property_name}' not found in property_names.")

    # Get property values
    property_index = property_names.index(property_name)
    property_values = properties_after[:, property_index]

    # Map elements to positions
    element_positions = {}
    for idx in range(len(element_symbols)):
        symbol = element_symbols[idx]
        group = element_groups[idx]
        period = element_periods[idx]
        element_positions[symbol] = (group, period)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    # Max group and period for setting axes
    max_group = max(element_groups)
    max_period = max(element_periods)
    min_period = min(element_periods)

    # Mapping node indices to symbols
    node_idx_to_symbol = {idx: symbol for idx, symbol in enumerate(element_symbols)}
    symbol_to_node_idx = {symbol: idx for idx, symbol in enumerate(element_symbols)}

    # Colormap setup
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.nanmin(property_values), vmax=np.nanmax(property_values))

    # Plot elements as rectangles
    for idx in range(len(element_symbols)):
        symbol = element_symbols[idx]
        group = element_groups[idx]
        period = element_periods[idx]
        prop_value = property_values[idx]
        x = group
        y = period  # Use 'y = period'
        color = cmap(norm(prop_value)) if not np.isnan(prop_value) else 'grey'
        rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                         facecolor=color,
                         edgecolor='white')

        ax.add_patch(rect)
        # Offset element names within boxes
        # ax.text(x - 0.2, y - 0.2, symbol,
        # ha='left', va='top', fontsize=8, color='white',
        # bbox=dict(facecolor='black', alpha=1.0))
        ax.text(x, y, symbol,
        ha='center', va='center', fontsize=8, color='white',
        bbox=dict(facecolor='black', alpha=1.0, boxstyle='circle'))

    # Draw connections based on relationships
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        idx1 = edges[0, i]
        idx2 = edges[1, i]
        symbol1 = node_idx_to_symbol[idx1]
        symbol2 = node_idx_to_symbol[idx2]
        group1, period1 = element_positions[symbol1]
        group2, period2 = element_positions[symbol2]
        x1 = group1
        y1 = period1
        x2 = group2
        y2 = period2
        # Draw horizontal and vertical lines
        if x1 == x2 or y1 == y2:
            ax.plot([x1, x2], [y1, y2], color='black')

            # # Calculate midpoint
            # x_mid = (x1 + x2) / 2
            # y_mid = (y1 + y2) / 2
            
            # # Plot a dot at the midpoint
            # ax.plot(x_mid, y_mid, marker='o', color='red', markersize=5)

    # Axes settings
    ax.set_xlim(0.5, max_group + 0.5)
    ax.set_ylim(min_period - 0.5, max_period + 0.5)
    ax.set_xlabel('Group')
    ax.set_ylabel('Period')

    # Invert y-axis to have Period 1 at the top
    ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Explicitly specify the Axes when creating the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(property_name.replace('_', ' ').title())

    # Ticks and labels
    ax.set_xticks(range(1, max_group + 1))
    ax.set_xticklabels(range(1, max_group + 1))

    ax.set_yticks(range(min_period, max_period + 1))
    ax.set_yticklabels(range(min_period, max_period + 1))

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title and show plot
    plt.title(f'Periodic Table Colored by {property_name.replace("_", " ").title()}')
    plt.tight_layout()

    os.makedirs('examples/scripts/properties', exist_ok=True)
    plt.savefig(f'examples/scripts/properties/{property_name}-periodic_table-after_imputation.png')

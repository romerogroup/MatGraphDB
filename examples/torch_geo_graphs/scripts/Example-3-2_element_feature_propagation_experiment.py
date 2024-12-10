import os


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager
from matgraphdb.utils import PKG_DIR, GRAPH_DIR
from matgraphdb.graph_kit.pyg.algo import feature_propagation

import matplotlib
matplotlib.use('Agg')

def holdout_experiment(data, holdout_fraction=0.2):
    non_nan_mask = ~torch.isnan(data.x)
    holdout_mask = torch.zeros_like(non_nan_mask, dtype=torch.bool)
    
    for feature_idx in range(data.x.shape[1]):
        feature_non_nan = non_nan_mask[:, feature_idx]
        num_non_nan = feature_non_nan.sum().item()
        num_holdout = int(num_non_nan * holdout_fraction)
        
        holdout_indices = np.random.choice(torch.where(feature_non_nan)[0].numpy(), 
                                           size=num_holdout, replace=False)
        holdout_mask[holdout_indices, feature_idx] = True
    original_values = data.x.clone()
    data_holdout = data.clone()
    data_holdout.x[holdout_mask] = float('nan')

    return data_holdout, original_values, holdout_mask

def evaluate_imputation(original, imputed, mask, property_names):
    mse_dict = {}
    mae_dict = {}
    mape_dict = {}
    r2_score_dict = {}
    
    for idx, property_name in enumerate(property_names):
        original_column = original[:, idx][mask[:, idx]]
        imputed_column = imputed[:, idx][mask[:, idx]]

        # original_column = original[:, idx].numpy()
        # imputed_column = imputed[:, idx]

        # non_nan_mask = ~np.isnan(original_column)
        # original_column = original_column[non_nan_mask]
        # imputed_column = imputed_column[non_nan_mask]
        
        if len(original_column) > 0:  # Only calculate if there are values to compare
            mse_dict[property_name] = mean_squared_error(original_column, imputed_column)
            mae_dict[property_name] = mean_absolute_error(original_column, imputed_column)
            mape_dict[property_name] = mean_absolute_percentage_error(original_column, imputed_column)
            r2_score_dict[property_name] = r2_score(original_column, imputed_column)
        else:
            mse_dict[property_name] = np.nan
            mae_dict[property_name] = np.nan
            mape_dict[property_name] = np.nan
            r2_score_dict[property_name] = np.nan
    
    return mse_dict, mae_dict, mape_dict, r2_score_dict


def plot_parity(original, imputed, feature_name, save_dir,nan_mask):
    plt.figure(figsize=(6, 6))

    # Plot non-NaN values in default color
    non_nan_mask = ~nan_mask
    plt.scatter(original[non_nan_mask], imputed[non_nan_mask], alpha=0.5, color='blue', label='Non-NaN')
    
    # Plot NaN values (imputed) in red
    plt.scatter(original[nan_mask], imputed[nan_mask], alpha=0.5, color='red', label='Imputed (NaN)')
    
    
    # Plot reference line
    plt.plot([original.min(), original.max()], [original.min(), original.max()], 'k--', lw=2)
    
    # Add labels and title
    plt.xlabel('Original')
    plt.ylabel('Imputed')
    plt.title(f'Parity Plot for {feature_name}')


    # Add legend
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{feature_name}.png'))
    plt.close()


def run_experiment(data, holdout_fractions, experiment_dir, property_names):
    os.makedirs(experiment_dir, exist_ok=True)
    results_file = os.path.join(experiment_dir, 'results.csv')
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=['Holdout_Fraction', 'Metric'] + property_names)
    
    for fraction in holdout_fractions:
        print(f"Running experiment with holdout fraction: {fraction}")
        
        # Create directory for this holdout fraction
        fraction_dir = os.path.join(experiment_dir, f'holdout_{fraction:.2f}')
        os.makedirs(fraction_dir, exist_ok=True)
        
        # Create directory for feature differences
        feature_diff_dir = os.path.join(fraction_dir, 'feature_differences')
        os.makedirs(feature_diff_dir, exist_ok=True)
        
        data_holdout, original_values, holdout_mask = holdout_experiment(data, holdout_fraction=fraction)
        
        # Save held-out dataframe
        df_holdout = pd.DataFrame(data_holdout.x.numpy(), columns=property_names)
        df_holdout.to_csv(os.path.join(fraction_dir, 'holdout_data.csv'), index=False)
        
        # Perform feature propagation
        imputed_values = feature_propagation(data=data_holdout)
        
        # Save imputed dataframe
        df_imputed = pd.DataFrame(imputed_values, columns=property_names)
        df_imputed.to_csv(os.path.join(fraction_dir, 'imputed_data.csv'), index=False)
        
        # Evaluate imputation
        mse_dict, mae_dict, mape_dict, r2_score_dict = evaluate_imputation(original_values, imputed_values, holdout_mask, property_names)
        
        # Append results to the DataFrame
        mse_row = pd.DataFrame({'Holdout_Fraction': fraction, 'Metric': 'MSE', **mse_dict}, index=[0])
        mae_row = pd.DataFrame({'Holdout_Fraction': fraction, 'Metric': 'MAE', **mae_dict}, index=[0])
        mape_row = pd.DataFrame({'Holdout_Fraction': fraction, 'Metric': 'MAPE', **mape_dict}, index=[0])
        r2_row = pd.DataFrame({'Holdout_Fraction': fraction, 'Metric': 'R2', **r2_score_dict}, index=[0])
        results_df = pd.concat([results_df, r2_row], ignore_index=True)
        
        # Generate and save parity plots
        for idx, property_name in enumerate(property_names):
            # Calculate feature differences
            original_column = original_values[:, idx].numpy()
            imputed_column = imputed_values[:, idx]
            
            abs_diff_column = np.abs(original_column - imputed_column)
            
            # Create DataFrame for feature differences
            df_feature_diff = pd.DataFrame({
                'Actual': original_column,
                'Imputed': imputed_column,
                'Absolute_Difference': abs_diff_column
            })
            
            # Save the feature differences CSV
            df_feature_diff.to_csv(os.path.join(feature_diff_dir, f'{property_name}_differences.csv'), index=False)
            
            # Generate parity plot
            plot_parity(original_column, imputed_column, property_name, fraction_dir, holdout_mask[:, idx])
        
        print(f"Completed experiment for holdout fraction: {fraction}")
    
    # Save results
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    element_file = os.path.join(PKG_DIR, 'utils', 'interim_periodic_table_values.csv')
    df = pd.read_csv(element_file)

    manager = GraphManager(graph_dir=os.path.join(GRAPH_DIR, 'main'))
    manager.nodes.get_element_nodes(base_element_csv='interim_periodic_table_values.csv', 
                                    from_scratch=True)
    
    generator = DataGenerator()

    property_names = ['abundance_universe', 'abundance_solar', 'abundance_meteor', 'abundance_crust', 'abundance_ocean', 'abundance_human',
                      'boiling_point', 'critical_pressure', 'critical_temperature', 'density_stp', 'conductivity_thermal',
                      'electron_affinity', 'electronegativity_pauling',
                      'heat_specific', 'heat_vaporization', 'heat_fusion', 'heat_molar',
                      'magnetic_susceptibility_mass', 'magnetic_susceptibility_molar', 'magnetic_susceptibility_volume',
                      'melting_point', 'molar_volume', 'neutron_cross_section', 'neutron_mass_absorption',
                      'radius_calculated', 'radius_empirical', 'radius_covalent', 'radius_vanderwaals', 'refractive_index',
                      'speed_of_sound', 'conductivity_electric', 'electrical_resistivity',
                      'modulus_bulk', 'modulus_shear', 'modulus_young', 'poisson_ratio', 'coefficient_of_linear_thermal_expansion',
                      'hardness_vickers', 'hardness_brinell', 'hardness_mohs', 'superconduction_temperature']
    
    generator.add_node_type(node_path=os.path.join(manager.node_dir, 'ELEMENT.parquet'),
                            feature_columns=property_names,
                            keep_nan=True)

    generator.add_edge_type(edge_path=os.path.join(manager.relationship_dir, 'ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT.parquet'))

    data = generator.homo_data

    # Define experiment parameters
    holdout_fractions = [0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1]
    experiment_dir = os.path.join('examples', 'experiments', 'feature_propagation_holdout')

    # Run the experiment
    run_experiment(data, holdout_fractions, experiment_dir, property_names)

    print(f"Experiment results saved in {experiment_dir}")


import yaml
import os
from .dataset import PolyDataset,MPPolyDataset

def generate_datasets(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    for dataset_config in config['datasets']:
        print(f"Generating dataset {dataset_config['name']}...")
        save_root=os.path.join(dataset_config['save_root'], dataset_config['name'])
        dataset = MPPolyDataset(
            save_root=save_root,
            raw_root=dataset_config['raw_root'],
            node_features=dataset_config['node_features'],
            edge_index=dataset_config['edge_index'],
            edge_features=dataset_config['edge_features'],
            y_feature=dataset_config['y_feature'],
            n_cores=dataset_config['n_cores']
        )

        file_path = os.path.join(save_root,'dataset_config.yml')
        with open(file_path, 'w') as outfile:
            yaml.dump(dataset_config, outfile, default_flow_style=False)
        print(f"Successfully generated dataset {dataset_config['name']}.")

# Usage:

# generate_datasets('data_config.yml')
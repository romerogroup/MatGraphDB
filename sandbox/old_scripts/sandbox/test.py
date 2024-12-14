
import h5py
import json
import os
import numpy as np

import time


def load_json_files(directory):
    data = []
    for filename in os.listdir(directory[:10]):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data.append(json.load(file))
    return data

def create_hdf5(data, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf:
        # Assuming all JSON files have the same structure
        keys = data[0].keys()
        max_shape = (None,)  # Allow unlimited rows
        
        # Create datasets for each property
        datasets = {}
        for key in keys:
            # Initialize datasets
            dtype = h5py.special_dtype(vlen=str) if isinstance(data[0][key], str) else np.float32
            datasets[key] = hdf.create_dataset(key, shape=(len(data),), maxshape=max_shape, dtype=dtype)
        
        # Fill datasets with data
        for i, entry in enumerate(data):
            for key, value in entry.items():
                datasets[key][i] = value



if __name__ == '__main__':
    # Directory containing JSON files
    json_database='data/raw/materials_project_nelements_2/json_database'


    data = load_json_files(json_database)
    print(data[0].keys())
    create_hdf5(data, hdf5_path='data/raw/test.hdf5')
    print(len(data))
    print(data[0])



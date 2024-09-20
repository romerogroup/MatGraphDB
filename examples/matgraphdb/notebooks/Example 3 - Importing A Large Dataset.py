import os
import json
import multiprocessing
from functools import partial

from matgraphdb import MatGraphDB

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_structure(structure_dict):
    lattice_info = structure_dict.get('lattice', None)
    lattice = lattice_info.get('matrix', None) if lattice_info else None

    sites = structure_dict.get('sites', [])
    species = []
    coords = []

    for site in sites:
        coord = site.get('abc', None)
        if coord is not None:
            coords.append(coord)
        else:
            coords.append([0, 0, 0])  # Default or handle missing coords

        specie = site.get('species', [{}])[-1].get('element', None)
        if specie is not None:
            species.append(specie)
        else:
            species.append('X')  # Default or handle missing species

    return coords, species, lattice

def process_material_data(data):
    if data is None:
        return None

    structure_dict = data.get('structure', {})
    coords, species, lattice = get_structure(structure_dict)

    # Remove 'structure' key from data to store the rest as additional data
    data.pop('structure', None)

    material_entry = {
        'coords': coords,
        'species': species,
        'lattice': lattice,
        'coords_are_cartesian': False,  # Assuming 'abc' are fractional coordinates
        'data': data
    }

    return material_entry

def main():
    # Initialize MatGraphDB
    mgdb = MatGraphDB(main_dir=os.path.join('data', 'MatGraphDB'))

    # Directory containing the JSON files
    json_dir = '/path/to/json/files'

    # List all JSON files
    json_files = [os.path.join(json_dir, fname) for fname in os.listdir(json_dir) if fname.endswith('.json')]

    # Number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Create a pool of workers
    with multiprocessing.Pool(num_processes) as pool:
        # Read all JSON files concurrently
        json_data_list = pool.map(read_json_file, json_files)

    # Process each material data
    with multiprocessing.Pool(num_processes) as pool:
        materials_to_insert = pool.map(process_material_data, json_data_list)

    # Filter out None entries
    materials_to_insert = [mat for mat in materials_to_insert if mat is not None]

    # Batch size for insertion
    batch_size = 1000  # Adjust based on memory and performance considerations

    # Insert materials in batches
    for i in range(0, len(materials_to_insert), batch_size):
        batch = materials_to_insert[i:i+batch_size]
        mgdb.db_manager.add_many(batch)
        print(f"Inserted batch {i//batch_size + 1} containing {len(batch)} materials.")

if __name__ == '__main__':
    main()

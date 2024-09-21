import os
import logging
import json
import multiprocessing
from functools import partial

from matgraphdb import MatGraphDB
from matgraphdb.utils import timeit


# Configure logging
logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.ERROR)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

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
    
    material_entry = {
        'coords': coords,
        'species': species,
        'lattice': lattice,
        'coords_are_cartesian': False,  # Assuming 'abc' are fractional coordinates
        'data': data
    }

    return material_entry

@timeit
def multiprocess_read_json_files(json_files, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        json_data_list = pool.map(read_json_file, json_files)
    return json_data_list

@timeit
def multiprocess_process_json_files(num_processes, json_data_list):
    with multiprocessing.Pool(num_processes) as pool:
        materials_to_insert = pool.map(process_material_data, json_data_list)
    return materials_to_insert

@timeit
def write_to_database(mgdb, materials_to_insert, batch_size):
    for i in range(0, len(materials_to_insert), batch_size):
        batch = materials_to_insert[i:i+batch_size]
        mgdb.db_manager.add_many(batch)
        print(f"Inserted batch {i//batch_size + 1} containing {len(batch)} materials.")


@timeit
def process_batch(mgdb, json_files_batch, num_processes, batch_size):
    # Step 4: Read JSON files with multiprocessing for the current batch
    print(f"Reading {len(json_files_batch)} JSON files in the current batch...")
    json_data_list = multiprocess_read_json_files(json_files_batch, num_processes)
    
    # Step 5: Process material data for the current batch
    print("Processing material data for the current batch...")
    materials_to_insert = multiprocess_process_json_files(num_processes, json_data_list)
    
    # Step 6: Filter out None entries
    print("Filtering None entries...")
    materials_to_insert = [mat for mat in materials_to_insert if mat is not None]
    print(f"Filtered materials. {len(materials_to_insert)} valid materials to insert in this batch.")
    
    # Step 8: Insert materials into the database in batches
    write_to_database(mgdb, materials_to_insert, batch_size)

def main():
    # Step 1: Initialize MatGraphDB
    print("Initializing MatGraphDB...")
    mgdb = MatGraphDB(main_dir=os.path.join('data', 'MatGraphDB_Example'))

    # Step 2: Directory containing the JSON files
    print("Listing all JSON files...")
    json_dir = os.path.join('data', 'production', 'materials_project', 'json_database')
    json_files = [os.path.join(json_dir, fname) for fname in os.listdir(json_dir) if fname.endswith('.json')]

    print(f"Found {len(json_files)} JSON files.")

    # Step 3: Number of processes to use
    num_processes = 40
    print(f"Using {num_processes} processes for parallel JSON reading and processing...")

    # Step 7: Batch size for insertion
    batch_size = 10000  # Adjust based on memory and performance considerations
    print(f"Processing and inserting materials in batches of {batch_size} JSON files...")

    # Step 8: Split the JSON files into batches and process them
    for i in range(0, len(json_files), batch_size):
        json_files_batch = json_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(json_files_batch)} JSON files.")
        process_batch(mgdb, json_files_batch, num_processes, batch_size)
    
    print("All batches processed and inserted successfully.")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

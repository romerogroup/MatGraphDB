import os
import logging
import json
import multiprocessing
from functools import partial

from matgraphdb import MatGraphDB
from matgraphdb.utils import timeit
from parquetdb import ParquetDB

# Configure logging
logger = logging.getLogger('parquetdb')
logger.setLevel(logging.ERROR)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)



PARQUET_ROWS_PER_FILE=20000
JSON_BATCH_SIZE=10000
N_CORES=40
# DB_DIR='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/MaterialsDB'
DB_DIR=os.path.join('data', 'MatGraphDB_Example')


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

            # for key, value in data.items():
            #     if isinstance(value, dict):
            #         data[key] = value.get('value', None)

            structure_dict = raw_data.get('structure')

            raw_data.pop('structure')
            raw_data.pop('composition')
            raw_data.pop('composition_reduced')
            if structure_dict:
                raw_data['lattice'] = structure_dict['lattice']['matrix']
                raw_data['a'] = structure_dict['lattice']['a']
                raw_data['b'] = structure_dict['lattice']['b']
                raw_data['c'] = structure_dict['lattice']['c']
                raw_data['alpha'] = structure_dict['lattice']['alpha']
                raw_data['beta'] = structure_dict['lattice']['beta']
                raw_data['gamma'] = structure_dict['lattice']['gamma']
                species=[]
                cart_coords=[]
                frac_coords=[]

                if 'properties' in structure_dict:
                    structure_dict.pop('properties')
                for isite,site in enumerate(structure_dict['sites']):
                    species.append(site['label'])
                    frac_coords.append(site['abc'])
                    cart_coords.append(site['xyz'])

                    structure_dict['sites'][isite]['properties'].update({'dummy_variable':True})

                raw_data['species'] = species
                raw_data['frac_coords'] = frac_coords
                raw_data['cart_coords'] = cart_coords
                
                raw_data['structure'] = structure_dict

        return raw_data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


@timeit
def multiprocess_read_json_files(json_files, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        json_data_list = pool.map(read_json_file, json_files)
    return json_data_list


@timeit
def write_to_database(db, materials_to_insert, batch_size):
    
    for i in range(0, len(materials_to_insert), batch_size):
        batch = materials_to_insert[i:i+batch_size]
        db.create(batch, max_rows_per_group=PARQUET_ROWS_PER_FILE, max_rows_per_file=PARQUET_ROWS_PER_FILE)
        print(f"Inserted batch {i//batch_size + 1} containing {len(batch)} materials.")


@timeit
def process_batch(mgdb, json_files_batch, num_processes, batch_size):
    # Step 4: Read JSON files with multiprocessing for the current batch
    print(f"Reading {len(json_files_batch)} JSON files in the current batch...")
    json_data_list = multiprocess_read_json_files(json_files_batch, num_processes)

    # Step 5: Filter out None entries
    print("Filtering None entries...")
    materials_to_insert = [mat for mat in json_data_list if mat is not None]
    print(f"Filtered materials. {len(materials_to_insert)} valid materials to insert in this batch.")
    
    # Step 6: Insert materials into the database in batches
    write_to_database(mgdb, materials_to_insert, batch_size)

def main():
    # Step 1: Initialize MatGraphDB
    print("Initializing MatGraphDB...")
    mgdb = MatGraphDB(main_dir=DB_DIR)
    db = mgdb.db_manager
    try:
        db.drop_table('main')
    except Exception as e:
        print(f"Error dropping table: {e}")

    # Step 2: Directory containing the JSON files
    print("Listing all JSON files...")
    json_dir = os.path.join('data', 'production', 'materials_project', 'json_database')
    json_files = [os.path.join(json_dir, fname) for fname in os.listdir(json_dir) if fname.endswith('.json')]

    print(f"Found {len(json_files)} JSON files.")

    # Step 3: Number of processes to use
    num_processes = N_CORES
    print(f"Using {num_processes} processes for parallel JSON reading and processing...")

    # Step 7: Batch size for insertion
    print(f"Processing and inserting materials in batches of {JSON_BATCH_SIZE} JSON files...")

    # Step 8: Split the JSON files into batches and process them
    for i in range(0, len(json_files), JSON_BATCH_SIZE):
        json_files_batch = json_files[i:i+JSON_BATCH_SIZE]
        print(f"Processing batch {i//JSON_BATCH_SIZE + 1} with {len(json_files_batch)} JSON files.")
        process_batch(db, json_files_batch, num_processes, JSON_BATCH_SIZE)
    
    print("All batches processed and inserted successfully.")

if __name__ == '__main__':
    main()

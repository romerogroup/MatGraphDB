import os
import logging
import json
import multiprocessing
from functools import partial
import time

from matgraphdb import MatGraphDB
from matgraphdb.utils import timeit
from parquetdb import ParquetDB


from matgraphdb import MatGraphDB



# parameters
PARQUET_ROWS_PER_FILE=20000
JSON_BATCH_SIZE=10000
N_CORES=40
# DB_DIR='/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/MaterialsDB'
DB_DIR=os.path.join('data', 'MatGraphDB')



################################################################################################
# Script starts here
################################################################################################

# Configure logging
logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.ERROR)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


def main():

    mgdb = MatGraphDB(main_dir=DB_DIR)
    db = mgdb.db_manager

    # Measure read time
    start_time = time.perf_counter()
    table = db.read(table_name='main', columns=['id','species','chargemol_bonding_connections', 'lattice', 'frac_coords'], output_format='table')
    end_time = time.perf_counter()
    df=table.to_pandas()

    # print(df.head())
    # print(df.tail())
    print(df['species'])
    # print(df['tail'].head())
    print(df['chargemol_bonding_connections'])
    print("Min ID: ", df['id'].min())
    print("Max ID: ", df['id'].max())

    print(df[df['id'].isin([0,1,2,3,4])])
    print("Shape of table: ", table.shape)
    n_columns = table.shape[1]
    size = table.shape[0]
    
    elapsed_time = end_time - start_time
    # read_times.append(elapsed_time)

    print(f"Read time for {size} entries: {elapsed_time:.4f} seconds")
    print(f"~Number of columns: {n_columns}")

    # # Step 1: Initialize MatGraphDB
    # print("Initializing MatGraphDB...")
    # mgdb = MatGraphDB(main_dir=os.path.join('data', 'MatGraphDB_Example'))

    # # Step 2: Directory containing the JSON files
    # print("Listing all JSON files...")
    # json_dir = os.path.join('data', 'production', 'materials_project', 'json_database')
    # json_files = [os.path.join(json_dir, fname) for fname in os.listdir(json_dir) if fname.endswith('.json')]

    # print(f"Found {len(json_files)} JSON files.")

    # # Step 3: Number of processes to use
    # num_processes = 20
    # print(f"Using {num_processes} processes for parallel JSON reading and processing...")

    # # Step 7: Batch size for insertion
    # batch_size = 10000  # Adjust based on memory and performance considerations
    # print(f"Processing and inserting materials in batches of {batch_size} JSON files...")

    # # Step 8: Split the JSON files into batches and process them
    # for i in range(0, len(json_files), batch_size):
    #     json_files_batch = json_files[i:i+batch_size]
    #     print(f"Processing batch {i//batch_size + 1} with {len(json_files_batch)} JSON files.")
    #     process_batch(mgdb, json_files_batch, num_processes, batch_size)
    
    # print("All batches processed and inserted successfully.")

if __name__ == '__main__':
    main()

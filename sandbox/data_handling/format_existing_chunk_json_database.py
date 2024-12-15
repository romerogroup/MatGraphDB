import os
import json
from glob import glob
from matgraphdb import config
import time
from parquetdb import ParquetDB



def main():
    # Example usage
    output_directory = os.path.join(config.data_dir,'production','materials_project','chunked_json')
    parquetdb_dir = os.path.join(config.data_dir,'production','materials_project','ParquetDB')
    dataset_name='materials'
    
    
    chunk_files = glob(os.path.join(output_directory,'*.json'))
    parquetdb = ParquetDB(db_path=os.path.join(parquetdb_dir, dataset_name))
    parquetdb.drop_dataset()
    for chunk_file in chunk_files[:]:
        start_time = time.time()
        process_chunk(chunk_file,parquetdb)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    
    parquetdb = ParquetDB(db_path=os.path.join(parquetdb_dir, dataset_name))
    table = parquetdb.read()
    
    write_schema_summary(parquetdb_dir,endpoint='materials')
    print(table.shape)


def write_schema_summary(materials_parquetdb_dir,endpoint='chemenv'):

    db=ParquetDB(db_path=os.path.join(materials_parquetdb_dir, endpoint))
    table=db.read()
    print(table.shape)
    
    with open(os.path.join(materials_parquetdb_dir,f'{endpoint}_schema_summary.txt'), 'w') as f:
        f.write(f"Number of rows: {table.shape[0]}\n")
        f.write(f"Number of columns: {table.shape[1]}\n\n")
        f.write('-'*100+'\n\n')
        
        f.write(f"{'Field Name':<50} | {'Field Type'}\n")
        f.write('-'*50+'\n')
        for field in table.schema:
            f.write(f"{field.name:<50} | {field.type}\n")


def process_chunk(chunk_file, parquetdb):
    start_time = time.time()
    
    
    with open(chunk_file, 'r') as f:
        data = json.load(f)
        
        n_records = len(data['entries'])
        base_name = os.path.basename(chunk_file)
        file_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)  # Convert bytes to MB
        
        print(f"Number of records in {base_name}: {n_records}")
        print(f"File size: {file_size_mb:.2f} MB")
        # print(data['entries'][0].keys())
        
        # print(data['entries'][0]['material_id'])
        
        if base_name == 'matgraphdb_chunk_8.json':
            for entry in data['entries']:
                if 'structure' in entry:
                    for site in entry['structure']['sites']:
                        if 'selective_dynamics' in site['properties']:
                            site['properties'].pop('selective_dynamics')

        parquetdb.create(data['entries'], 
                        treat_fields_as_ragged=['chargemol_cubed_moments',
                                                'chargemol_fourth_moments',
                                                'chargemol_squared_moments',
                                                'chargemol_bonding_connections',
                                                'chargemol_bonding_orders',
                                                'bond_cutoff_connections',
                                                'bonding_cutoff_connections',
                                                'oxidation_states.possible_valences',
                                                'coordination_multi_numbers',
                                                'coordination_multi_connections'])

            
        
        
        # for entry in data['entries']:
        #     if 'material_id' not in entry:
        #         print(entry.keys())
        
        print('-'*100)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
        




if __name__ == "__main__":
    
    main()
    
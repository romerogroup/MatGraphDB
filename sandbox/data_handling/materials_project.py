from glob import glob
import os
import shutil
import json
from dotenv import load_dotenv
import time
from mp_api.client import MPRester
from parquetdb import ParquetDB, config
import pyarrow as pa
from pyarrow import compute as pc

config.logging_config.loggers.parquetdb.level='DEBUG'
config.apply()



def main():
    external_dir=os.path.join('data','external')
    materials_dir = os.path.join(external_dir,'materials_project','materials')
    materials_parquetdb_dir=os.path.join(external_dir,'materials_project', 'materials_ParquetDB')

    
    list_of_endpoints=['materials_summary','chemenv','absorption','eos','grain_boundaries','provenance', 'elasticity',
                       'electronic_structure','phonon','piezoelectric','thermo','dielectric',
                       'magnetism','similarity','synthesis','surface_properties']

    # for endpoint in list_of_endpoints:
    #     try:
    #         write_schema_summary(materials_parquetdb_dir,endpoint)
    #     except Exception as e:
    #         print(f"Error writing schema summary for {endpoint}: {e}")
            
    db=ParquetDB('elasticity', dir=materials_parquetdb_dir)
    table=db.read(filters=[pc.field('material_id')=='mp-28692'])
    
    print(table)
    print(table.shape)
            
    


def write_schema_summary(materials_parquetdb_dir,endpoint='chemenv'):

    db=ParquetDB(endpoint, dir=materials_parquetdb_dir)
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
            



# db=ParquetDB('elasticity', dir=materials_parquetdb_dir)
# db=ParquetDB('bonds', dir=materials_parquetdb_dir)
# db=ParquetDB('piezoelectric', dir=materials_parquetdb_dir)
# db=ParquetDB('thermo', dir=materials_parquetdb_dir)
# db=ParquetDB('dielectric', dir=materials_parquetdb_dir)
# db=ParquetDB('oxidation_states', dir=materials_parquetdb_dir)
# db=ParquetDB('electronic_structure', dir=materials_parquetdb_dir)
# db=ParquetDB('phonon', dir=materials_parquetdb_dir)


# db=ParquetDB('absorption', dir=materials_parquetdb_dir)
# db=ParquetDB('eos', dir=materials_parquetdb_dir)
# db=ParquetDB('grain_boundaries', dir=materials_parquetdb_dir)
# db=ParquetDB('provenance', dir=materials_parquetdb_dir)
# db=ParquetDB('magnetism', dir=materials_parquetdb_dir)
# db=ParquetDB('similarity', dir=materials_parquetdb_dir)
# db=ParquetDB('synthesis', dir=materials_parquetdb_dir)
# db=ParquetDB('surface_properties', dir=materials_parquetdb_dir)






if __name__=='__main__':
    main()
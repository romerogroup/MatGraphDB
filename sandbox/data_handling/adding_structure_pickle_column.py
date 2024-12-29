import os
import inspect
import pandas as pd
import shutil
import re
import logging
import yaml
import logging.config
import pickle

import pyarrow as pa
from pymatgen.core import Structure
from parquetdb import ParquetDB
from pyarrow import compute as pc

from matgraphdb import MatGraphDB, config
from matgraphdb.utils.parquet_tools import write_schema_summary

import time

# class MyStructure(Structure):
def _serialize_MyStructure(self, value):
    return value.as_dict()

def _deserialize_MyStructure(self, value):
    return Structure.from_dict(value)


root_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(config.data_dir, 'materials_schema'), exist_ok=True)

materials_db = ParquetDB(db_path=os.path.join(config.data_dir, 'materials'))

# materials_db.delete(columns=['structre_pickle'])

schema = materials_db.get_schema()
for field in schema:
    print(field.name, field.type)
# df=table.combine_chunks().to_pandas()
# df['structure']=df['structre_pickle'].map(pickle.loads)
# print(df.head())


# print(type(df.iloc[0]['structure']))
# # df=table.combine_chunks().to_pandas()



# # write_schema_summary(os.path.join(config.data_dir, 'materials'),
# #                      os.path.join(config.data_dir, 'materials_schema','materials_schema.txt'))


table = materials_db.read(columns=['id','structure','core.material_id'],rebuild_nested_struct=True, rebuild_nested_from_scratch=False)


start_time = time.time()
df = table.combine_chunks().to_pandas()
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")



# start_time = time.time()
# df = table.combine_chunks().to_pandas()
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")

# start_time = time.time()
# df['structre_py']=df['structure'].map(Structure.from_dict)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")

# start_time = time.time()
# df['structure_pickle']=df['structure'].map(pickle.dumps)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")


# print(df.head())
# start_time = time.time()
# df['structre_pickle_load']=df['structure_pickle'].map(pickle.loads)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")


# print(df.head())
# materials_db.update(df[['id','structure_pickle']])



import os
import json
from glob import glob
from matgraphdb import config
import time
from parquetdb import ParquetDB
import pyarrow as pa
from pyarrow import compute as pc

from matgraphdb.utils.parquet_tools import write_schema_summary
from matgraphdb import config, MatGraphDB
from matgraphdb.core.graph_store import GraphStore


config.logging_config.loggers.matgraphdb.level = 'DEBUG'
config.apply()

def main():
    # materials_db = ParquetDB(db_path=os.path.join(config.data_dir,'materials'))
    # table=materials_db.read()
    # print(table)
    
    # matgraphdb=MatGraphDB(db_path=os.path.join(config.data_dir,'MatGraphDB'))
    
    # table=matgraphdb.matdb.read()
    
    
    
    # results = pc.index_in(table['id'], pa.array([1,2,3]))
    # null_values=table['id'].filter(pc.is_null(results))
    # # null_indices = pc.index(pc.is_null(results))
    # print("Indices of null values:", null_values)
    
    
    
    graph = GraphStore(os.path.join(config.data_dir,'GraphStore'))
    
    
    # print(table)
    # print(table.shape)
    
    
    
    
    

if __name__ == "__main__":
    main()
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


# config.logging_config.loggers.matgraphdb.level = 'DEBUG'
# config.logging_config.loggers.parquetdb.level = 'DEBUG'
# config.apply()


root_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(config.data_dir, 'materials_schema'), exist_ok=True)

materials_db = ParquetDB(db_path=os.path.join(config.data_dir, 'materials'))
table = materials_db.read(columns=['core.atomic_numbers'])
df=table.to_pandas()
# print(df.head())

# start_time = time.time()
# struct_df=df['structure'].apply(Structure.from_dict)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")


# matgraphdb = MatGraphDB(main_dir=os.path.join(config.data_dir, 'MatGraphDB'))

# table = matgraphdb.matdb.read(columns=['core.frac_coords'])
# print(table['core.frac_coords'].type)

# mgo_structure = Structure(
#     lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
#     species=["Mg", "O"],
#     coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
# )

# mgo_structure.add_site_property('magmom', [2.0, 2.0])

# properties=dict(
#     electronic_structure=dict(
#         band_gap=1.0
#     )
# )
# matgraphdb.matdb.add(structure=mgo_structure, properties=properties, verbose=4)





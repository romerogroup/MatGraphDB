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


root_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(config.data_dir, 'materials_schema'), exist_ok=True)

materials_db = ParquetDB(db_path=os.path.join(config.data_dir, 'materials'))



table=materials_db.read(columns=['id', 'structure'],rebuild_nested_struct=True)


df = table.to_pandas()

df['structure']=df['structure'].map(Structure.from_dict)

df['core.frac_coords']=df['structure'].map(lambda x: x.frac_coords.tolist())
df['core.cartesian_coords']=df['structure'].map(lambda x: x.cart_coords.tolist())
df['core.lattice']=df['structure'].map(lambda x: x.lattice.matrix.tolist())
df['core.species']=df['structure'].map(lambda x: [specie.symbol for specie in x.species])
df['core.formula']=df['structure'].map(lambda x: x.composition.formula)
df['core.atomic_numbers']=df['structure'].map(lambda x: x.atomic_numbers)

df= df.drop(columns=['structure'])    


materials_db.update(df[['id', 'core.frac_coords', 'core.cartesian_coords', 
                        'core.lattice', 'core.species', 'core.formula', 'core.atomic_numbers']])

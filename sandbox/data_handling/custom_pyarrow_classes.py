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

# df=table.combine_chunks().to_pandas()



# write_schema_summary(os.path.join(config.data_dir, 'materials'),
#                      os.path.join(config.data_dir, 'materials_schema','materials_schema.txt'))


# table = materials_db.read(columns=['id','structure','core.material_id'],rebuild_nested_struct=True)





class StructureType(pa.ExtensionType):
    def __init__(self, data_type: pa.DataType):
        if not pa.types.is_struct(data_type):
            raise TypeError(f"data_type must be an integer type not {data_type}")

        super().__init__(
            data_type,
            "matgraphdb.structure",
        )
    def __arrow_ext_serialize__(self) -> bytes:
        # No parameters are necessary
        return b""
    
    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # Sanity checks, not required but illustrate the method signature.
        assert pa.types.is_struct(storage_type)
        assert serialized == b""

        # return an instance of this subclass
        return StructureType(storage_type)
    
    def __arrow_ext_class__(self):
        return StructureArray



class StructureArray(pa.ExtensionArray):
    def to_structure(self):
        return self.storage.to_pandas().map(Structure.from_dict)


table = materials_db.read(columns=['structure'], rebuild_nested_struct=True)
print(type(table['structure'].type))
structure_type = StructureType(table['structure'].type)


storage_array = pa.array(
    table['structure'].combine_chunks(),
    type=structure_type.storage_type,
)


arr = pa.ExtensionArray.from_storage(structure_type, storage_array)
print(arr)

print(dir(arr))

start_time = time.time()
print(arr.to_structure())
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")



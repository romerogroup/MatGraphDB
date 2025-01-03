
import os
import json
import shutil
from glob import glob
from parquetdb import ParquetDB

if os.path.exists('data/external/chemenv'):
    shutil.rmtree('data/external/chemenv')

db = ParquetDB('data/external/chemenv')



json_dir='data/external/coordination_geometries'

files=glob(os.path.join(json_dir, '*.json'))

records=[]
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        
        # print(data.keys())
        # print(data['_algorithms'])
        data.pop('_algorithms')
        records.append(data)
        
        
db.create(records)



table=db.read()
print(table.shape)



import os

import numpy as np

import pyarrow.compute as pc
from matgraphdb import MatGraphDB, config
from matgraphdb.utils import timeit


def main():
    mgdb = MatGraphDB(main_dir=os.path.join('data','MatGraphDB_dev'))

    # json_database_path = os.path.join(config.data_dir,'production','materials_project','json_database_with_bonddoc')
    json_database_path = os.path.join(config.data_dir,'production','materials_project','json_database')
    
    
    # # Create a list of 5 identical materials
    # materials = [material for _ in range(5)]
    
    # # Add multiple materials at once using add_many
    # mgdb.matdb.add_many(materials)
    
    # # Verify the materials were added
    # results = mgdb.matdb.read()
    # df = results.to_pandas()
    # print(f"Number of materials after add_many: {df.shape[0]}")

    # mgdb.matdb.delete(ids=[0,1])
    # results=mgdb.matdb.read(columns=['id'])
    # df=results.to_pandas()
    
    # print(df)
    # print(df.shape)
    
    # mgdb.matdb.update([{'id':1, 'density':100000.0}, {'id':2, 'density':100000.0}])
    # results=mgdb.matdb.read(columns=['id','density'], filters=[pc.field('density') > 2.0])
    
    # df=results.to_pandas()
    # print(df)
    # print(df.shape)



if __name__ == "__main__":
    main()
import os

import numpy as np

import pyarrow.compute as pc
from matgraphdb import MatGraphDB, config
from matgraphdb.utils import timeit


@timeit
def init_matgraphdb():
    mgdb = MatGraphDB(main_dir=os.path.join('data','MatGraphDB'))
    return mgdb

def main():
    mgdb = init_matgraphdb()



    mgdb.db_manager.add(coords=np.array([[0,0,0]]), 
                        species=['Fe'], 
                        lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                        properties={'density':1.0})
    
    mgdb.db_manager.add(coords=np.array([[0,0,0]]), 
                        species=['Fe'], 
                        lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                        properties={'density':1.0})
    

    mgdb.db_manager.add(coords=np.array([[0,0,0]]), 
                        species=['Fe'], 
                        lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                        properties={'density':1.0})
    # mgdb.db_manager.add(composition='TiO2')

    # mgdb.db_manager.delete(ids=[0])
    results=mgdb.db_manager.read(columns=['id'])
    df=results.to_pandas()
    print(df)
    print(df.shape)
    
    mgdb.db_manager.update([{'id':1, 'density':100000.0}, {'id':2, 'density':100000.0}])
    results=mgdb.db_manager.read(columns=['id','density'], filters=[pc.field('density') > 2.0])
    
    df=results.to_pandas()
    print(df)
    print(df.shape)


    # results=mgdb.db_manager.read(selection='has_structure=False')

    # from ase import Atoms


    # h = Atoms('H')







if __name__ == "__main__":
    main()
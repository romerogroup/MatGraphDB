import logging
import os

from httpx import delete
import numpy as np
from zmq import has

from matgraphdb import MatGraphDB
from matgraphdb.utils import timeit

logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

@timeit
def init_matgraphdb():
    mgdb = MatGraphDB(main_dir=os.path.join('data','MatGraphDB'))
    return mgdb

def main():
    mgdb = init_matgraphdb()



    mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
                                 species=['Fe'], 
                                 lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                                 data={'density':1.0})
    
    mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
                                 species=['Fe'], 
                                 lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                                 data={'density':1.0})
    

    mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
                                 species=['Fe'], 
                                 lattice=[[1,0,0],[0,1,0],[0,0,1]], 
                                 data={'density':1.0})
    mgdb.db_manager.add_material(composition='TiO2')

    # mgdb.db_manager.delete_material(material_ids=[8])
    results=mgdb.db_manager.read()

    for result in results:
        print(result)
        print(result.id)
        print(result.data)
        print('-'*100)


    results=mgdb.db_manager.read(selection='has_structure=False')

    for result in results:
        print(result)
        print(result.id)
        print(result.data)
        print('-'*100)
    # from ase import Atoms


    # h = Atoms('H')







if __name__ == "__main__":
    main()
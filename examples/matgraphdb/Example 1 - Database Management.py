import logging
import os

import numpy as np

from matgraphdb import MatGraphDB


logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)




def main():
    mgdb = MatGraphDB(main_dir=os.path.join('data','MatGraphDB'))



    # mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
    #                              species=['Fe'], 
    #                              lattice=[[1,0,0],[0,1,0],[0,0,1]], 
    #                              data={'density':1.0})
    
    # mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
    #                              species=['Fe'], 
    #                              lattice=[[1,0,0],[0,1,0],[0,0,1]], 
    #                              data={'density':1.0})
    

    # mgdb.db_manager.add_material(coords=np.array([[0,0,0]]), 
    #                              species=['Fe'], 
    #                              lattice=[[1,0,0],[0,1,0],[0,0,1]], 
    #                              data={'density':1.0})
    # mgdb.db_manager.add_material(composition='TiO2')

    # Update the density of material with ID 1
    mgdb.db_manager.update_material(
        material_id=1,
        data={'density': 8.00}
    )

    # Read the updated material
    results = mgdb.db_manager.read(selection=1)
    for result in results:
        print(f"ID: {result.id}")
        print(f"Data: {result.data}")
    # mgdb.db_manager.delete_material(material_ids=[8])
    # results=mgdb.db_manager.read()

    # for result in results:
    #     print(result)
    #     print(result.id)
    #     print(result.data)
    #     print('-'*100)


    # results=mgdb.db_manager.read(selection='has_structure=False')

    # for result in results:
    #     print(result)
    #     print(result.id)
    #     print(result.data)
    #     print('-'*100)
    # from ase import Atoms


    # h = Atoms('H')







if __name__ == "__main__":
    main()
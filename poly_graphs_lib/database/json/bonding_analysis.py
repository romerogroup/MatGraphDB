import os
import json
from glob import glob 

from multiprocessing import Pool
import pymatgen.core as pmat
from poly_graphs_lib.utils import PROJECT_DIR
from pymatgen.analysis.local_env import CutOffDictNN
from poly_graphs_lib.utils.periodic_table import covalent_cutoff_map

CUTOFF_DICT=covalent_cutoff_map(tol=0.1)

def bonding_analysis(file):

    try:
        print(file)
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])

        cutoff_nn=CutOffDictNN(cut_off_dict=CUTOFF_DICT)
        all_nn=cutoff_nn.get_all_nn_info(structure=struct)
        nearest_neighbors=[]
        for site_nn in all_nn:
            neighbor_index=[]
            for nn in site_nn:

                index=nn['site_index']
                neighbor_index.append(index)
            nearest_neighbors.append(neighbor_index)

      
        db['bonding_cutoff_connections']=nearest_neighbors


    except Exception as e:
        print(e)
        print(file)
        db['bonding_cutoff_connections']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def process_database(n_cores=1):
    
    database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database')

    database_files=glob(database_dir + '\*.json')

    if n_cores==1:
        for i,file in enumerate(database_files[:50]):
            if i%100==0:
                print(i)
            print(file)
            bonding_analysis(file)
    else:
        with Pool(n_cores) as p:
            p.map(bonding_analysis, database_files)


if __name__=='__main__':
    process_database(n_cores=6)

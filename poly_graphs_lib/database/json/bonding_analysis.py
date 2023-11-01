import json

import pymatgen.core as pmat
from pymatgen.analysis.local_env import CutOffDictNN

from poly_graphs_lib.utils.periodic_table import covalent_cutoff_map
from poly_graphs_lib.database.json import process_database
from poly_graphs_lib.database import DB_DIR

CUTOFF_DICT=covalent_cutoff_map(tol=0.1)

def bonding_analysis(file, from_scratch=False):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        if 'bonding_cutoff_connections' not in db or from_scratch:
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



if __name__=='__main__':
    print('Running Bonding Cutoff analysis')
    print('Database Dir : ', DB_DIR)
    process_database(bonding_analysis)

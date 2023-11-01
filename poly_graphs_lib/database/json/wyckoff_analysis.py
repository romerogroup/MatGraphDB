import json

import pymatgen.core as pmat
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from poly_graphs_lib.database.json import process_database
from poly_graphs_lib.database import DB_DIR

def wyckoff_analysis(file, from_scratch=False):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        if 'wyckoffs' not in db or from_scratch:
            spg_a = SpacegroupAnalyzer(struct)
            sym_dataset=spg_a.get_symmetry_dataset()

            db['wyckoffs']=sym_dataset['wyckoffs']


    except Exception as e:
        print(e)
        print(file)
        db['wyckoffs']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)



if __name__=='__main__':
    print('Running wyckoffs position analysis')
    print('Database Dir : ', DB_DIR)
    process_database(wyckoff_analysis)

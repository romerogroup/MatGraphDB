import os
import json

import pymatgen.core as pmat
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.data.utils import process_database
from matgraphdb.utils import DB_DIR, LOGGER

def wyckoff_calc_task(file, from_scratch=False):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        mpid=file.split(os.sep)[-1].split('.')[0]
        if 'wyckoffs' not in db or from_scratch:
            spg_a = SpacegroupAnalyzer(struct)
            sym_dataset=spg_a.get_symmetry_dataset()

            db['wyckoffs']=sym_dataset['wyckoffs']


    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")
        db['wyckoffs']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)


def wyckoff_calc():
    LOGGER.info('#' * 100)
    LOGGER.info('Running wyckoffs position analysis')
    LOGGER.info('#' * 100)

    process_database(wyckoff_calc_task)

if __name__=='__main__':
    wyckoff_calc()

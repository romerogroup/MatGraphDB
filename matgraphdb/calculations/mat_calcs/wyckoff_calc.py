import os
import json

import pymatgen.core as pmat
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.data.utils import process_database
from matgraphdb.utils import DB_DIR, LOGGER


def wyckoff_calc_task(file, from_scratch=False):
    """
    Perform Wyckoff calculations on a given file.

    Args:
        file (str): The path to the file to be processed.
        from_scratch (bool, optional): If True, perform calculations from scratch. 
            If False, use existing data if available. Default is False.

    Returns:
        None

    Raises:
        Exception: If there is an error processing the file.

    """
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
    """
    Performs wyckoffs position analysis.

    This function runs the wyckoffs position analysis by logging information and
    calling the `process_database` function with the `wyckoff_calc_task` parameter.

    Parameters:
        None

    Returns:
        None
    """
    LOGGER.info('#' * 100)
    LOGGER.info('Running wyckoffs position analysis')
    LOGGER.info('#' * 100)

    process_database(wyckoff_calc_task)

if __name__=='__main__':
    wyckoff_calc()

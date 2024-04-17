import os
from glob import glob
from multiprocessing import Pool

from matgraphdb.utils import DB_DIR, N_CORES

def process_database(func, n_cores=N_CORES):
    """
    Process the database files using the provided function.

    Args:
        func (function): The function to be applied to each database file.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is N_CORES.

    Returns:
        list: A list of results from applying the function to each database file.
    """

    database_files = glob(DB_DIR + os.sep + '*.json')

    if n_cores == 1:
        for i, file in enumerate(database_files[:]):
            if i % 100 == 0:
                print(i)
            print(file)
            func(file)
    else:
        with Pool(n_cores) as p:
            results = p.map(func, database_files)
    return results


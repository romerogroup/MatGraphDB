import os

from poly_graphs_lib.utils import LOGGER

N_CORES=20

# N_CORES=os.cpu_count()
LOGGER.info(f'NCORES : {N_CORES}')
print('NCORES : ', N_CORES)
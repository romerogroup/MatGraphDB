# Import necessary libraries
import logging
import os
import numpy as np
from matgraphdb import MatGraphDB

# Configure logging
logger = logging.getLogger('matgraphdb')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)



mgdb = MatGraphDB(main_dir=os.path.join('data', 'MatGraphDB'))

results = mgdb.db_manager.read(columns=['id'])

# Display the results
count = 0
for result in results:
    if count % 100 == 0:
        print(count)
    count += 1

print(f"Found {count} materials.")
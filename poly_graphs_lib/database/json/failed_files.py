import os
import json
from glob import glob 

from poly_graphs_lib.utils import PROJECT_DIR

def process_database(n_cores=1):
    
    database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database')

    database_files=glob(database_dir + '\*.json')

    count=0
    for i,file in enumerate(database_files):
        if i%100==0:
            print(i)
    
        with open(file) as f:   
            db = json.load(f)

        # db['coordination_environments_multi_weight']=coordination_environments
        # db['coordination_multi_connections']=nearest_neighbors
        # db['coordination_multi_numbers']=coordination_numbers

        if db['coordination_multi_connections'] == None:
            count+=1
            print(file)

    print(count)


if __name__=='__main__':
    process_database(n_cores=6)

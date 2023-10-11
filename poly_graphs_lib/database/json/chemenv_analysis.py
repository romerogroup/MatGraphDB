import os
import json
from glob import glob 

from multiprocessing import Pool
import numpy as np
import pymatgen.core as pmat
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.local_env import NearNeighbors,VoronoiNN
from poly_graphs_lib.utils import PROJECT_DIR


def chemenv_analysis(file):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])

        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=struct)

        se = lgf.compute_structure_environments(maximum_distance_factor=1.41,only_cations=False)
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
        lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        # list of possible coordination environements per site
        coordination_environments = lse.coordination_environments
        coordination_numbers=[]
        for env in coordination_environments:
            if env is None:
                coordination_numbers.append('NaN')
            else:
                coordination_numbers.append(int(env[0]['ce_symbol'].split(':')[-1]))
        
        nearest_neighbors=[]
        for i_site, neighbors in enumerate(lse.neighbors_sets):
            neighbors=neighbors[0]
            neighbor_index=[]
            for neighbor_site in neighbors.neighb_sites_and_indices:
                index=neighbor_site['index']
                neighbor_index.append(index)
            nearest_neighbors.append(neighbor_index)

        db['coordination_environments']=coordination_environments
        db['coordination_connections']=nearest_neighbors
        db['coordination_numbers']=coordination_numbers

    except Exception as e:
        print(e)
        print(file)
        db['coordination_environments']=None
        db['coordination_connections']=None
        db['coordination_numbers']=None

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
            chemenv_analysis(file)
    else:
        with Pool(n_cores) as p:
            p.map(chemenv_analysis, database_files)


if __name__=='__main__':
    process_database(n_cores=6)

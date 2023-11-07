import json

import pymatgen.core as pmat
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from poly_graphs_lib.database.json import DB_DIR
from poly_graphs_lib.database.json.process_database import process_database

def chemenv_analysis(file, from_scratch=False):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        if 'coordination_environments_multi_weight' not in db or from_scratch:
            lgf = LocalGeometryFinder()
            lgf.setup_structure(structure=struct)

            se = lgf.compute_structure_environments(maximum_distance_factor=1.41,only_cations=False)
            # strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)

            strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
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

            db['coordination_environments_multi_weight']=coordination_environments
            db['coordination_multi_connections']=nearest_neighbors
            db['coordination_multi_numbers']=coordination_numbers
        
    except Exception as e:
        print(e)
        print(file)
        db['coordination_environments_multi_weight']=None
        db['coordination_multi_connections']=None
        db['coordination_multi_numbers']=None


    with open(file,'w') as f:
        json.dump(db, f, indent=4)



if __name__=='__main__':
    print('Running Chemenv analysis using Multi Weight Strategy')
    print('Database Dir : ', DB_DIR)
    process_database(chemenv_analysis)

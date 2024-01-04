import os
import json
import numpy as np
import pymatgen.core as pmat
from pymatgen.analysis.local_env import CutOffDictNN
from pymatgen.core.periodic_table import Element

from poly_graphs_lib.utils import LOGGER
from poly_graphs_lib.utils.periodic_table import covalent_cutoff_map
from poly_graphs_lib.database.json.utils import process_database
from poly_graphs_lib.database.json import DB_DIR,GLOBAL_PROP_FILE

# List of element names from pymatgen's Element
ELEMENTS = dir(Element)[:-4]

def bond_stats_calc(files, from_scratch=False, save=True):



    # Initialize arrays for bond order calculations
    n_elements = len(ELEMENTS)
    n_bond_orders = np.zeros(shape=(n_elements, n_elements))
    bond_orders_avg = np.zeros(shape=(n_elements, n_elements))
    bond_orders_std = np.zeros(shape=(n_elements, n_elements))

    # First iteration: calculate sum and count of bond orders
    for file in files:
        # Load database from JSON file
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        
        # Extract material project ID from file name
        mpid = file.split(os.sep)[-1].split('.')[0]
        
        try:
            
            bond_orders = db["chargemol_bonding_orders"]
            bond_connections = db["chargemol_bonding_connections"]
            site_element_names = [x['label'] for x in db['structure']['sites']]

            
            for isite, site in enumerate(bond_orders):
                site_element = site_element_names[isite]
                neighbors = bond_connections[isite]

                for jsite in neighbors:
                    neighbor_site_element = site_element_names[jsite]
                    bond_order = bond_orders[isite][jsite]

                    i_element = ELEMENTS.index(site_element)
                    j_element = ELEMENTS.index(neighbor_site_element)

                    bond_orders_avg[i_element, j_element] += bond_order
                    n_bond_orders[i_element, j_element] += 1
                
        except Exception as e:
            LOGGER.error(f"Error processing file {mpid}: {e}")

    bond_orders_avg /= n_bond_orders

    # Second iteration to get the standard deviation
    for file in files:
        # Load database from JSON file
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        
        # Extract material project ID from file name
        mpid = file.split(os.sep)[-1].split('.')[0]
        
        try:
            
            bond_orders = db["chargemol_bonding_orders"]
            bond_connections = db["chargemol_bonding_connections"]
            site_element_names = [x['label'] for x in db['structure']['sites']]

            # First iteration: calculate sum and count of bond orders
            for isite, site in enumerate(bond_orders):
                site_element = site_element_names[isite]
                neighbors = bond_connections[isite]

                for jsite in neighbors:
                    neighbor_site_element = site_element_names[jsite]
                    bond_order = bond_orders[isite][jsite]

                    i_element = ELEMENTS.index(site_element)
                    j_element = ELEMENTS.index(neighbor_site_element)

                    bond_order_avg=bond_orders_avg[i_element, j_element]
                    bond_orders_std[i_element, j_element] += (bond_order- bond_order_avg)**2
         
                
        except Exception as e:
            LOGGER.error(f"Error processing file {mpid}: {e}")


    bond_orders_std /= n_bond_orders
    bond_orders_std = bond_orders_std**0.5

    # Save changes to the file if needed
    if save:
        with open(GLOBAL_PROP_FILE, 'w') as f:
            data = json.load(f)

        data['bond_orders_avg']=bond_orders_avg.tolist()
        data['bond_orders_std']=bond_orders_std.tolist()
        data['n_bond_orders']=n_bond_orders.tolist()

        with open(GLOBAL_PROP_FILE, 'w') as f:
            json.dump(data, f, indent=4)

# Main execution
if __name__ == '__main__':
    print('#' * 100)
    print('Running Bonding Cutoff analysis')
    print('#' * 100)
    process_database(bond_stats_calc)

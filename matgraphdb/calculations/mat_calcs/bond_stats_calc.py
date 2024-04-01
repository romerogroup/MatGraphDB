import os
import json
import numpy as np
import pymatgen.core as pmat

from matgraphdb.data.utils import process_database
from matgraphdb.utils import LOGGER, DB_DIR, GLOBAL_PROP_FILE
from matgraphdb.utils.periodic_table import atomic_symbols


# List of element names from pymatgen's Element
ELEMENTS = atomic_symbols[1:]

def bond_orders_sum_calc(file):
    # Load database from JSON file
    with open(file) as f:
        db = json.load(f)
        struct = pmat.Structure.from_dict(db['structure'])
    
    # Extract material project ID from file name
    mpid = file.split(os.sep)[-1].split('.')[0]

     # Initialize arrays for bond order calculations
    n_elements = len(ELEMENTS)
    n_bond_orders = np.zeros(shape=(n_elements, n_elements))
    bond_orders_sum = np.zeros(shape=(n_elements, n_elements))

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

                bond_orders_sum[i_element, j_element] += bond_order
                n_bond_orders[i_element, j_element] += 1

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return bond_orders_sum, n_bond_orders

def bond_orders_std_calc(file):
    # Load database from JSON file
    with open(file) as f:
        db = json.load(f)
        struct = pmat.Structure.from_dict(db['structure'])
    
    with open(GLOBAL_PROP_FILE) as f:
        data = json.load(f)
        bond_orders_avg=np.array(data['bond_orders_avg'])
        n_bond_orders=np.array(data['n_bond_orders'])

    # Extract material project ID from file name
    mpid = file.split(os.sep)[-1].split('.')[0]

    # Initialize arrays for bond order calculations
    n_elements = len(n_bond_orders)
    bond_orders_std = np.zeros(shape=(n_elements, n_elements))
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

                bond_order_avg = bond_orders_avg[i_element, j_element]
                bond_orders_std[i_element, j_element] += (bond_order - bond_order_avg) ** 2

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

    return bond_orders_std, 1


def bond_stats_calc():

    LOGGER.info('#' * 100)
    LOGGER.info('Running Bonding Stats Calculation')
    LOGGER.info('#' * 100)

    # Initialize arrays for bond order calculations
    n_elements = len(ELEMENTS)
    n_bond_orders = np.zeros(shape=(n_elements, n_elements))
    bond_orders_avg = np.zeros(shape=(n_elements, n_elements))
    bond_orders_std = np.zeros(shape=(n_elements, n_elements))

    LOGGER.info('#' * 100)
    LOGGER.info('Getting bond order avg')
    LOGGER.info('#' * 100)
    results = process_database(bond_orders_sum_calc)

    for result in results:
        bond_orders_avg+=result[0] 
        n_bond_orders+=result[1]


    bond_orders_avg = np.divide(bond_orders_avg, n_bond_orders, out=np.zeros_like(bond_orders_avg), where=n_bond_orders!=0)


    with open(GLOBAL_PROP_FILE) as f:
        data = json.load(f)
        data['bond_orders_avg']=bond_orders_avg.tolist()
        data['n_bond_orders']=n_bond_orders.tolist()


    with open(GLOBAL_PROP_FILE, 'w') as f:
        json.dump(data, f, indent=4)


    LOGGER.info('#' * 100)
    LOGGER.info('Getting bond order std')
    LOGGER.info('#' * 100)
    results = process_database(bond_orders_std_calc)
    for result in results:
        bond_orders_std+=result[0] 

    bond_orders_std = np.divide(bond_orders_std, n_bond_orders, out=np.zeros_like(bond_orders_std), where=n_bond_orders!=0)
    bond_orders_std = bond_orders_std ** 0.5


    with open(GLOBAL_PROP_FILE) as f:
        data = json.load(f)
        data['bond_orders_std']=bond_orders_std.tolist()

    with open(GLOBAL_PROP_FILE, 'w') as f:
        json.dump(data, f, indent=4)



# Main execution
if __name__ == '__main__':
    bond_stats_calc()






# import os
# import json
# import numpy as np
# import pymatgen.core as pmat
# from pymatgen.analysis.local_env import CutOffDictNN
# from pymatgen.core.periodic_table import Element

# from poly_graphs_lib.utils import LOGGER
# from poly_graphs_lib.utils.periodic_table import atomic_symbols
# from poly_graphs_lib.database.utils.process_database import process_database
# from poly_graphs_lib.database.utils import DB_DIR,GLOBAL_PROP_FILE

# # List of element names from pymatgen's Element
# ELEMENTS = atomic_symbols[1:]

# def bond_orders_sum_calc(file):
#     # Load database from JSON file
#     with open(file) as f:
#         db = json.load(f)
#         struct = pmat.Structure.from_dict(db['structure'])
    
#     # Extract material project ID from file name
#     mpid = file.split(os.sep)[-1].split('.')[0]

#      # Initialize arrays for bond order calculations
#     n_elements = len(ELEMENTS)
#     n_bond_orders = np.zeros(shape=(n_elements, n_elements))
#     bond_orders_sum = np.zeros(shape=(n_elements, n_elements))

#     try:
#         bond_orders = db["chargemol_bonding_orders"]
#         bond_connections = db["chargemol_bonding_connections"]
#         site_element_names = [x['label'] for x in db['structure']['sites']]

#         # First iteration: calculate sum and count of bond orders
#         for isite, site in enumerate(bond_orders):
#             site_element = site_element_names[isite]
#             neighbors = bond_connections[isite]

#             for jsite in neighbors:
#                 neighbor_site_element = site_element_names[jsite]
#                 bond_order = bond_orders[isite][jsite]

#                 i_element = ELEMENTS.index(site_element)
#                 j_element = ELEMENTS.index(neighbor_site_element)

#                 # Avoid double counting diagonal
#                 if i_element != j_element:
#                     bond_orders_sum[i_element, j_element] += bond_order
#                     bond_orders_sum[j_element, i_element] += bond_order
#                     n_bond_orders[i_element, j_element] += 1
#                     n_bond_orders[j_element, i_element] += 1
#                 else:
#                     bond_orders_sum[i_element, j_element] += bond_order
#                     n_bond_orders[j_element, i_element] += 1
#     except Exception as e:
#         LOGGER.error(f"Error processing file {mpid}: {e}")

#     return bond_orders_sum, n_bond_orders

# def bond_orders_std_calc(file):
#     # Load database from JSON file
#     with open(file) as f:
#         db = json.load(f)
#         struct = pmat.Structure.from_dict(db['structure'])
    
#     with open(GLOBAL_PROP_FILE) as f:
#         data = json.load(f)
#         bond_orders_avg=np.array(data['bond_orders_avg'])
#         n_bond_orders=np.array(data['n_bond_orders'])

#     # Extract material project ID from file name
#     mpid = file.split(os.sep)[-1].split('.')[0]

#     # Initialize arrays for bond order calculations
#     n_elements = len(n_bond_orders)
#     bond_orders_std = np.zeros(shape=(n_elements, n_elements))
#     try:
#         bond_orders = db["chargemol_bonding_orders"]
#         bond_connections = db["chargemol_bonding_connections"]
#         site_element_names = [x['label'] for x in db['structure']['sites']]

#         # First iteration: calculate sum and count of bond orders
#         for isite, site in enumerate(bond_orders):
#             site_element = site_element_names[isite]
#             neighbors = bond_connections[isite]

#             for jsite in neighbors:
#                 neighbor_site_element = site_element_names[jsite]
#                 bond_order = bond_orders[isite][jsite]

#                 i_element = ELEMENTS.index(site_element)
#                 j_element = ELEMENTS.index(neighbor_site_element)

#                 bond_order_avg = bond_orders_avg[i_element, j_element]

#                 # Avoid double counting diagonal
#                 if i_element != j_element:
#                     bond_orders_std[i_element, j_element] += (bond_order - bond_order_avg) ** 2
#                     bond_orders_std[j_element, i_element] += (bond_order - bond_order_avg) ** 2
#                 else:
#                     bond_orders_std[i_element, j_element] += (bond_order - bond_order_avg) ** 2
#     except Exception as e:
#         LOGGER.error(f"Error processing file {mpid}: {e}")

#     return bond_orders_std, 1


# def bond_stats_calc():

#     LOGGER.info('#' * 100)
#     LOGGER.info('Running Bonding Stats Calculation')
#     LOGGER.info('#' * 100)

#     # Initialize arrays for bond order calculations
#     n_elements = len(ELEMENTS)
#     n_bond_orders = np.zeros(shape=(n_elements, n_elements))
#     bond_orders_avg = np.zeros(shape=(n_elements, n_elements))
#     bond_orders_std = np.zeros(shape=(n_elements, n_elements))

#     LOGGER.info('#' * 100)
#     LOGGER.info('Getting bond order avg')
#     LOGGER.info('#' * 100)
#     results = process_database(bond_orders_sum_calc)

#     for result in results:
#         bond_orders_avg+=result[0] 
#         n_bond_orders+=result[1]


#     bond_orders_avg = np.divide(bond_orders_avg, n_bond_orders, out=np.zeros_like(bond_orders_avg), where=n_bond_orders!=0)


#     with open(GLOBAL_PROP_FILE) as f:
#         data = json.load(f)
#         data['bond_orders_avg']=bond_orders_avg.tolist()
#         data['n_bond_orders']=n_bond_orders.tolist()


#     with open(GLOBAL_PROP_FILE, 'w') as f:
#         json.dump(data, f, indent=4)


#     LOGGER.info('#' * 100)
#     LOGGER.info('Getting bond order std')
#     LOGGER.info('#' * 100)
#     results = process_database(bond_orders_std_calc)
#     for result in results:
#         bond_orders_std+=result[0] 

#     bond_orders_std = np.divide(bond_orders_std, n_bond_orders, out=np.zeros_like(bond_orders_std), where=n_bond_orders!=0)
#     bond_orders_std = bond_orders_std ** 0.5


#     with open(GLOBAL_PROP_FILE) as f:
#         data = json.load(f)
#         data['bond_orders_std']=bond_orders_std.tolist()

#     with open(GLOBAL_PROP_FILE, 'w') as f:
#         json.dump(data, f, indent=4)



# # Main execution
# if __name__ == '__main__':
#     bond_stats_calc()


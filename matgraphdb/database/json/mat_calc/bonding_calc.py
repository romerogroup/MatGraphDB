import json

import pymatgen.core as pmat
from pymatgen.analysis.local_env import CutOffDictNN

from matgraphdb.utils.periodic_table import covalent_cutoff_map
from matgraphdb.database.utils import process_database
from matgraphdb.utils import DB_DIR, LOGGER

def calculate_geometric_electric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders):
    """
    Adjusts the electric bond orders and connections to be consistent with the geometric bond connections.

    Args:
        geo_coord_connections (list): List of geometric bond connections.
        elec_coord_connections (list): List of electric bond connections.
        bond_orders (list): List of bond orders.

    Returns:
        tuple: A tuple containing the adjusted electric bond connections and bond orders.

    """
    final_connections=[]
    final_bond_orders=[]

    for elec_site_connections,geo_site_connections, site_bond_orders in zip(elec_coord_connections,geo_coord_connections,bond_orders):

        # Determine most likely electric bonds
        elec_reduced_bond_indices = [i for i,order in enumerate(site_bond_orders) if order > 0.1]
        n_elec_bonds=len(elec_reduced_bond_indices)
        n_geo_bonds=len(geo_site_connections)

        # If there is only one geometric bond and one or less electric bonds, then we can use the electric bond orders and connections as is
        if n_geo_bonds == 1 and n_elec_bonds <= 1:
            reduced_bond_orders=site_bond_orders
            reduced_elec_site_connections=elec_site_connections

        # Else if there is only one geometric bond and more than 1 electric bonds, then we can use the electric reduced bond orders and connections as is
        elif n_geo_bonds == 1 and n_elec_bonds > 1:
            reduced_elec_site_connections = [elec_site_connections[i] for i in elec_reduced_bond_indices]
            reduced_bond_orders = [site_bond_orders[i] for i in elec_reduced_bond_indices]

        # If there are more than one geometric bonds, then we need to sort the bond orders and elec connections by the total number of geometric connections
        # Geometric bonds and electric bonds should have a correspondence with each other
        else:
            geo_reduced_bond_order_indices = sorted(range(len(site_bond_orders)), key=lambda i: site_bond_orders[i], reverse=True)[:n_geo_bonds]

            geo_reduced_bond_orders = [site_bond_orders[i] for i in geo_reduced_bond_order_indices]
            geo_reduced_elec_site_connections = [elec_site_connections[i] for i in geo_reduced_bond_order_indices]

            # I take only bond orders greater than 0.1 because geometric connection alone can be wrong sometimes. For example in the case of oxygen.
            geo_elec_reduced_bond_indices = [i for i,order in enumerate(geo_reduced_bond_orders) if order > 0.1]

            reduced_elec_site_connections = [geo_reduced_elec_site_connections[i] for i in geo_elec_reduced_bond_indices]
            reduced_bond_orders = [geo_reduced_bond_orders[i] for i in geo_elec_reduced_bond_indices]

        final_site_connections=reduced_elec_site_connections
        final_site_bond_orders=reduced_bond_orders
            
        final_connections.append(final_site_connections)
        final_bond_orders.append(final_site_bond_orders)
    return final_connections, final_bond_orders

def calculate_electric_consistent_bonds(elec_coord_connections, bond_orders):
    """
    Adjusts the electric bond orders and connections to be consistent with the geometric bond connections.

    Args:
        geo_coord_connections (list): List of geometric bond connections.
        elec_coord_connections (list): List of electric bond connections.
        bond_orders (list): List of bond orders.

    Returns:
        tuple: A tuple containing the adjusted electric bond connections and bond orders.

    """
    final_connections=[]
    final_bond_orders=[]

    for elec_site_connections, site_bond_orders in zip(elec_coord_connections,bond_orders):

        # Determine most likely electric bonds
        elec_reduced_bond_indices = [i for i,order in enumerate(site_bond_orders) if order > 0.1]
        n_elec_bonds=len(elec_reduced_bond_indices)

        reduced_elec_site_connections = [elec_site_connections[i] for i in elec_reduced_bond_indices]
        reduced_bond_orders = [site_bond_orders[i] for i in elec_reduced_bond_indices]

        final_site_connections=reduced_elec_site_connections
        final_site_bond_orders=reduced_bond_orders
            
        final_connections.append(final_site_connections)
        final_bond_orders.append(final_site_bond_orders)
    return final_connections, final_bond_orders

def calculate_geometric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders):
    """
    Adjusts the electric bond orders and connections to be consistent with the geometric bond connections.

    Args:
        geo_coord_connections (list): List of geometric bond connections.
        elec_coord_connections (list): List of electric bond connections.
        bond_orders (list): List of bond orders.

    Returns:
        tuple: A tuple containing the adjusted electric bond connections and bond orders.

    """
    final_connections=[]
    final_bond_orders=[]

    for geo_site_connections,elec_site_connections, site_bond_orders in zip(geo_coord_connections,elec_coord_connections,bond_orders):

        n_geo_bonds=len(geo_site_connections)
        geo_reduced_bond_order_indices = sorted(range(len(site_bond_orders)), key=lambda i: site_bond_orders[i], reverse=True)[:n_geo_bonds]
        geo_reduced_bond_orders = [site_bond_orders[i] for i in geo_reduced_bond_order_indices]
  

        reduced_elec_site_connections = [elec_site_connections[i] for i in geo_reduced_bond_order_indices]


        final_site_connections=reduced_elec_site_connections
        final_site_bond_orders=geo_reduced_bond_orders
            
        final_connections.append(final_site_connections)
        final_bond_orders.append(final_site_bond_orders)
    return geo_coord_connections, final_bond_orders



def bonding_calc_task(file, from_scratch=False):
    CUTOFF_DICT=covalent_cutoff_map(tol=0.1)
    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        mpid=file.split('/')[-1].split('.')[0]
        if 'bonding_cutoff_connections' not in db or from_scratch:
            cutoff_nn=CutOffDictNN(cut_off_dict=CUTOFF_DICT)
            all_nn=cutoff_nn.get_all_nn_info(structure=struct)
            nearest_neighbors=[]
            for site_nn in all_nn:
                neighbor_index=[]
                for nn in site_nn:

                    index=nn['site_index']
                    neighbor_index.append(index)
                nearest_neighbors.append(neighbor_index)

            db['bonding_cutoff_connections']=nearest_neighbors


    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")


        db['bonding_cutoff_connections']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def bonding_calc():
    LOGGER.info('#' * 100)
    LOGGER.info('Running Bonding Cutoff Calculation')
    LOGGER.info('#' * 100)
    process_database(bonding_calc_task)


def geometric_consistent_bonding_task(file, from_scratch=True):

    try:
        with open(file) as f:
            db = json.load(f)

        mpid=file.split('/')[-1].split('.')[0]

        if 'geo_consistent_bond_connections' not in db or from_scratch:

            geo_coord_connections = db['coordination_multi_connections']
            elec_coord_connections = db['chargemol_bonding_connections']
            chargemol_bond_orders=db['chargemol_bonding_orders']
            final_geo_connections, final_bond_orders = calculate_geometric_consistent_bonds(geo_coord_connections, elec_coord_connections, chargemol_bond_orders)

            db['geometric_consistent_bond_connections']=final_geo_connections
            db['geometric_consistent_bond_orders']=final_bond_orders

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

        db['geometric_consistent_bond_connections']=None
        db['geometric_consistent_bond_orders']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def geometric_consistent_bonding():
    LOGGER.info('#' * 100)
    LOGGER.info('Running Geometric Consistent Bonding Calculation')
    LOGGER.info('#' * 100)
    process_database(geometric_consistent_bonding_task)



def geometric_electric_consistent_bonding_task(file, from_scratch=True):

    try:
        with open(file) as f:
            db = json.load(f)

        mpid=file.split('/')[-1].split('.')[0]

        if 'geometric_electric_consistent_bond_connections' not in db or from_scratch:

            geo_coord_connections = db['coordination_multi_connections']
            elec_coord_connections = db['chargemol_bonding_connections']
            chargemol_bond_orders=db['chargemol_bonding_orders']
            final_geo_connections, final_bond_orders = calculate_geometric_electric_consistent_bonds(geo_coord_connections, elec_coord_connections, chargemol_bond_orders)

            db['geometric_electric_consistent_bond_connections']=final_geo_connections
            db['geometric_electric_consistent_bond_orders']=final_bond_orders

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

        db['geometric_electric_electric_consistent_bond_connections']=None
        db['geometric_electric_consistent_bond_orders']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def geometric_electric_consistent_bonding():
    LOGGER.info('#' * 100)
    LOGGER.info('Running Geometric Electric Consistent Bonding Calculation')
    LOGGER.info('#' * 100)
    process_database(geometric_electric_consistent_bonding_task)



def electric_consistent_bonding_task(file, from_scratch=True):

    try:
        with open(file) as f:
            db = json.load(f)

        mpid=file.split('/')[-1].split('.')[0]

        if 'electric_consistent_bond_connections' not in db or from_scratch:

            elec_coord_connections = db['chargemol_bonding_connections']
            chargemol_bond_orders=db['chargemol_bonding_orders']
            final_geo_connections, final_bond_orders = calculate_electric_consistent_bonds( elec_coord_connections, chargemol_bond_orders)

            db['electric_consistent_bond_connections']=final_geo_connections
            db['electric_consistent_bond_orders']=final_bond_orders

    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

        db['electric_consistent_bond_connections']=None
        db['electric_consistent_bond_orders']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def electric_consistent_bonding():
    LOGGER.info('#' * 100)
    LOGGER.info('Running Electric Consistent Bonding Calculation')
    LOGGER.info('#' * 100)
    process_database(electric_consistent_bonding_task)



if __name__=='__main__':
    # bonding_calc()

    # geometric_consistent_bonding()
    geometric_electric_consistent_bonding()
    electric_consistent_bonding()

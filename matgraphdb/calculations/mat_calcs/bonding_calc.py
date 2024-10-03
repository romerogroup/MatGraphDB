
from pymatgen.analysis.local_env import CutOffDictNN

from matgraphdb.utils.chem_utils.periodic import covalent_cutoff_map
from matgraphdb.utils import  LOGGER

def calculate_geometric_electric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders, threshold=0.1):
    """
    Adjusts the electric bond orders and connections to be consistent with the geometric bond connections above a given threshold.

    Args:
        geo_coord_connections (list): List of geometric bond connections.
        elec_coord_connections (list): List of electric bond connections.
        bond_orders (list): List of bond orders.
        threshold (float, optional): Threshold for bond orders. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the adjusted electric bond connections and bond orders.

    """
    try:
        final_connections=[]
        final_bond_orders=[]

        for elec_site_connections,geo_site_connections, site_bond_orders in zip(elec_coord_connections,geo_coord_connections,bond_orders):

            # Determine most likely electric bonds
            elec_reduced_bond_indices = [i for i,order in enumerate(site_bond_orders) if order > threshold]
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
    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        final_connections=None
        final_bond_orders=None

    return final_connections, final_bond_orders

def calculate_electric_consistent_bonds(elec_coord_connections, bond_orders, threshold=0.1):
    """
    Calculates the electric consistent bonds for a given set of electric bond connections and bond orders above a given threshold.

    Args:
        elec_coord_connections (list): List of electric bond connections.
        bond_orders (list): List of bond orders.
        threshold (float, optional): Threshold for bond orders. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the adjusted electric bond connections and bond orders.

    """
    try:
        final_connections=[]
        final_bond_orders=[]

        for elec_site_connections, site_bond_orders in zip(elec_coord_connections,bond_orders):

            # Determine most likely electric bonds
            elec_reduced_bond_indices = [i for i,order in enumerate(site_bond_orders) if order > threshold]
            n_elec_bonds=len(elec_reduced_bond_indices)

            reduced_elec_site_connections = [elec_site_connections[i] for i in elec_reduced_bond_indices]
            reduced_bond_orders = [site_bond_orders[i] for i in elec_reduced_bond_indices]

            final_site_connections=reduced_elec_site_connections
            final_site_bond_orders=reduced_bond_orders
                
            final_connections.append(final_site_connections)
            final_bond_orders.append(final_site_bond_orders)

    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        final_connections=None
        final_bond_orders=None

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
    try:
        final_connections=[]
        final_bond_orders=[]

        for geo_site_connections,elec_site_connections, site_bond_orders in zip(geo_coord_connections,elec_coord_connections,bond_orders):
            
            # Orders the electric bond orders by magnitudes up to the total amount of geometric bonds
            n_geo_bonds=len(geo_site_connections)
            geo_reduced_bond_order_indices = sorted(range(len(site_bond_orders)), key=lambda i: site_bond_orders[i], reverse=True)[:n_geo_bonds]
            
            # Reduces the electric bond orders and reduces the number of electric bond connections to the number of geometric bonds
            geo_reduced_bond_orders = [site_bond_orders[i] for i in geo_reduced_bond_order_indices]
            reduced_elec_site_connections = [elec_site_connections[i] for i in geo_reduced_bond_order_indices]

            final_site_connections=reduced_elec_site_connections
            final_site_bond_orders=geo_reduced_bond_orders
                
            final_connections.append(final_site_connections)
            final_bond_orders.append(final_site_bond_orders)
    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        final_connections=None
        final_bond_orders=None
    return final_connections, final_bond_orders

def calculate_cutoff_bonds(structure):
    """
    Calculates the cutoff bonds for a given crystal structure.

    Args:
        structure (Structure): The crystal structure for which to calculate the cutoff bonds.

    Returns:
        list: A list of lists, where each inner list contains the indices of the nearest neighbors for each site in the structure.
    """
    try:
        CUTOFF_DICT = covalent_cutoff_map(tol=0.1)
        cutoff_nn = CutOffDictNN(cut_off_dict=CUTOFF_DICT)
        all_nn = cutoff_nn.get_all_nn_info(structure=structure)
        nearest_neighbors = []
        for site_nn in all_nn:
            neighbor_index = []
            for nn in site_nn:
                index = nn['site_index']
                neighbor_index.append(index)
            nearest_neighbors.append(neighbor_index)
    except Exception as e:
        LOGGER.error(f"Error calculating cutoff bonds: {e}")
        nearest_neighbors = None
    return nearest_neighbors

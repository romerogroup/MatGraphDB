import numpy as np

from matgraphdb.utils import LOGGER
from matgraphdb.utils.chem_utils.periodic import atomic_symbols


def calculate_bond_orders_sum(bond_orders,bond_connections, site_element_names):
    """
    Calculates the sum and count of bond orders for a given file.

    Args:
        file (str): The path to the JSON file containing the database.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array represents the sum of bond orders
               between different elements, and the second array represents the count of bond orders.

    Raises:
        Exception: If there is an error processing the file.
    """
    # List of element names from pymatgen's Element
    ELEMENTS = atomic_symbols[1:]
    # Initialize arrays for bond order calculations
    n_elements = len(ELEMENTS)
    n_bond_orders = np.zeros(shape=(n_elements, n_elements))
    bond_orders_sum = np.zeros(shape=(n_elements, n_elements))

    try:
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
        LOGGER.error(f"Error processing file {e}")
 
    return bond_orders_sum, n_bond_orders

def calculate_bond_orders_sum_squared_differences(bond_orders,bond_connections, site_element_names, bond_orders_avg, n_bond_orders):
    """
    Calculate the standard deviation of bond orders for a given material.

    Parameters:
    bond_orders (numpy.ndarray): The bond orders between different elements for a material.
    bond_connections (numpy.ndarray): The bond connections between different elements for a material.
    site_element_names (list): The names of the elements in the structure for a material.
    bond_orders_avg (numpy.ndarray): The average bond orders between different elements in the material database
    n_bond_orders (numpy.ndarray): The count of bond orders between different elements in the material database.

    Returns:
    bond_orders_std (numpy.ndarray): The standard deviation of bond orders between different elements.
    """
    # List of element names from pymatgen's Element
    ELEMENTS = atomic_symbols[1:]
    # Initialize arrays for bond order calculations
    n_elements = len(n_bond_orders)
    bond_orders_std = np.zeros(shape=(n_elements, n_elements))
    try:
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
        LOGGER.error(f"Error processing file: {e}")

    return bond_orders_std


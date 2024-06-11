import os
import json
import copy
from glob import glob

import pymatgen.core as pmat
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from matgraphdb.utils import LOGGER,DB_DIR
from matgraphdb.data.utils import process_database

def calculate_chemenv_connections(structure):
    """
    Calculate the coordination environments, nearest neighbors, and coordination numbers for a given structure.

    Parameters:
        structure (Structure): The input structure for which the chemenv calculations are performed.

    Returns:
        tuple: A tuple containing the coordination environments, nearest neighbors, and coordination numbers.
            - coordination_environments (list): A list of dictionaries representing the coordination environments for each site.
            - nearest_neighbors (list): A list of lists containing the indices of the nearest neighbors for each site.
            - coordination_numbers (list): A list of coordination numbers for each site.
    """

    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure=structure)

    # Compute the structure environments
    se = lgf.compute_structure_environments(maximum_distance_factor=1.41, only_cations=False)

    # Define the strategy for environment calculation
    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
    lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)

    # Get a list of possible coordination environments per site
    coordination_environments = copy.copy(lse.coordination_environments)

    # Replace empty environments with default value
    for i, env in enumerate(lse.coordination_environments):
        if env is None or env==[]:
            coordination_environments[i] = [{'ce_symbol': 'S:1', 'ce_fraction': 1.0, 'csm': 0.0, 'permutation': [0]}]

    # Calculate coordination numbers
    coordination_numbers = []
    for env in coordination_environments:
        if env is None:
            coordination_numbers.append('NaN')
        else:
            coordination_numbers.append(int(env[0]['ce_symbol'].split(':')[-1]))
    
    # Determine nearest neighbors
    nearest_neighbors = []
    for i_site, neighbors in enumerate(lse.neighbors_sets):

        neighbor_index = []
        if neighbors!=[]:
            neighbors = neighbors[0]
            for neighbor_site in neighbors.neighb_sites_and_indices:
                index = neighbor_site['index']
                neighbor_index.append(index)
            nearest_neighbors.append(neighbor_index)
        else:
            pass

    return coordination_environments, nearest_neighbors, coordination_numbers


def chemenv_calc_task(file, from_scratch=True):
    """
    Perform chemical environment calculation for a given material.

    Args:
        file (str): The path to the JSON file containing the material data.
        from_scratch (bool, optional): Flag indicating whether to perform the calculation from scratch or use existing data. 
            Defaults to True.

    Returns:
        None

    Raises:
        Exception: If an error occurs during the calculation.

    """
    # Load data from JSON file
    with open(file) as f:
        db = json.load(f)
        struct = pmat.Structure.from_dict(db['structure'])

    # Extract material project ID from file name
    mpid = file.split(os.sep)[-1].split('.')[0]

    try:
        # Check if calculation is needed
        if 'coordination_environments_multi_weight' not in db or from_scratch:
            coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(struct)
            # Update the database with computed values
            db['coordination_environments_multi_weight'] = coordination_environments
            db['coordination_multi_connections'] = nearest_neighbors
            db['coordination_multi_numbers'] = coordination_numbers
        
    except Exception as e:
        # Log any errors encountered during processing
        LOGGER.error(f"Error processing file {mpid}: {e}")

        # Set fields to None in case of error
        db['coordination_environments_multi_weight'] = None
        db['coordination_multi_connections'] = None
        db['coordination_multi_numbers'] = None

    # Write the updated data back to the JSON file
    with open(file, 'w') as f:
        json.dump(db, f, indent=4)


def chemenv_calc():
    """
    Perform Chemenv Calculation using Multi Weight Strategy.

    This function runs the Chemenv Calculation using the Multi Weight Strategy.
    It prints a header for the process and then processes the database with the defined function.

    Parameters:
    None

    Returns:
    None
    """

    # Print header for the process
    LOGGER.info('#' * 100)
    LOGGER.info('Running Chemenv Calculation using Multi Weight Strategy')
    LOGGER.info('#' * 100)

    # Process the database with the defined function
    process_database(chemenv_calc_task)

# Main execution block
if __name__ == '__main__':
    chemenv_calc()
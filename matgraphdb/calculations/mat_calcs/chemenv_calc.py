import copy
import logging

from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

logger = logging.getLogger(__name__)


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

    try:
        error=None
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

    except Exception as error:
        logger.error(f"Error processing file: {error}")
        coordination_environments = None
        nearest_neighbors = None
        coordination_numbers = None


    return coordination_environments, nearest_neighbors, coordination_numbers
import os
import json
from glob import glob

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.graph.node_types import (ELEMENTS,ELEMENT_PROPERTIES, MAGNETIC_STATES, CRYSTAL_SYSTEMS, CHEMENV_NAMES,
                                                CHEMENV_ELEMENT_NAMES, SPG_NAMES,OXIDATION_STATES,OXIDATION_STATES_NAMES,
                                                MATERIAL_FILES,MATERIAL_IDS, MATERIAL_PROPERTIES, SPG_WYCKOFFS,
                                                LATTICE_IDS, LATTICE_PROPERTIES,SITE_IDS, SITE_PROPERTIES, SITES_IDS, SITES_PROPERTIES)
from matgraphdb.utils import  GLOBAL_PROP_FILE, NODE_DIR, LOGGER, ENCODING_DIR

def create_nodes(node_names, node_type, node_prefix, node_properties=None, filepath=None):
    """
    Create nodes for a graph database.

    Args:
        node_names (list): List of node names.
        node_type (str): Type of nodes.
        node_prefix (str): Prefix for node IDs.
        node_properties (list, optional): List of dictionaries containing additional properties for each node. Defaults to None.
        filepath (str, optional): Filepath to save the node data as a CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the node data.

    """
    
    node_dict = {
        f'{node_prefix}Id:ID({node_prefix}-ID)': [],
        'type:LABEL': [],
        'name:string': []
    }


    # Initialize the node_dict with the properties from node_properties
    if node_properties:
        for property_name in node_properties[0].keys():
            node_dict[property_name] = []


    for i, node_name in enumerate(node_names):
        node_dict[f'{node_prefix}Id:ID({node_prefix}-ID)'].append(i)
        node_dict['type:LABEL'].append(node_type)
        node_dict['name:string'].append(node_name.replace(':', '_'))

        # Add the extra attributes to the node_dict
        if node_properties:
            for property_name, property_value in node_properties[i].items():
                node_dict[property_name].append(property_value)

    df = pd.DataFrame(node_dict)

    if filepath:
        df.to_csv(filepath, index=False)

    return df


def main():
    save_path = os.path.join(NODE_DIR,'new')

    print('Save_path:', save_path)
    os.makedirs(save_path, exist_ok=True)
    
    print('Creating nodes...')
    
    # # Elements
    create_nodes(node_names=ELEMENTS, 
                node_type='Element', 
                node_prefix='element', 
                node_properties=ELEMENT_PROPERTIES,
                filepath=os.path.join(save_path, 'elements.csv'))
    
    # # Crystal Systems
    # create_nodes(node_names=CRYSTAL_SYSTEMS, 
    #             node_type='CrystalSystem', 
    #             node_prefix='crystalSystem', 
    #             filepath=os.path.join(save_path, 'crystal_systems.csv'))
    
    # # Chemenv
    # create_nodes(node_names=CHEMENV_NAMES, 
    #             node_type='Chemenv', 
    #             node_prefix='chemenv', 
    #             filepath=os.path.join(save_path, 'chemenv_names.csv'))
    
    # # Chemenv Element
    # create_nodes(node_names=CHEMENV_ELEMENT_NAMES, 
    #             node_type='ChemenvElement', 
    #             node_prefix='chemenvElement', 
    #             filepath=os.path.join(save_path, 'chemenv_element_names.csv'))
    
    # # Magnetic States
    # create_nodes(node_names=MAGNETIC_STATES, 
    #             node_type='MagneticState', 
    #             node_prefix='magState', 
    #             filepath=os.path.join(save_path, 'magnetic_states.csv'))
    
    # # Space Groups
    # create_nodes(node_names=SPG_NAMES, 
    #             node_type='SpaceGroup', 
    #             node_prefix='spg', 
    #             filepath=os.path.join(save_path, 'spg.csv'))
    
    # # Oxidation States
    # create_nodes(node_names=OXIDATION_STATES_NAMES, 
    #             node_type='OxidationState', 
    #             node_prefix='oxiState', 
    #             filepath=os.path.join(save_path, 'oxidation_states.csv'))
    
    # Materials
    create_nodes(node_names=MATERIAL_IDS,
                node_type='Material',
                node_prefix='materials',
                node_properties=MATERIAL_PROPERTIES,
                filepath=os.path.join(save_path, 'materials.csv'))
    
    # SPG_WYCKOFFS
    # create_nodes(node_names=SPG_WYCKOFFS,
    #             node_type='SPGWyckoff',
    #             node_prefix='spgWyckoff',
    #             filepath=os.path.join(save_path, 'spg_wyckoff.csv'))

    ##################################################################################################
    # # Lattice
    # create_nodes(node_names=LATTICE_IDS, 
    #             node_type='Lattice', 
    #             node_prefix='lattice', 
    #             node_properties=LATTICE_PROPERTIES,
    #             filepath=os.path.join(save_path, 'lattice.csv'))
    
    # # Sites
    # create_nodes(node_names=SITES_IDS, 
    #             node_type='Site', 
    #             node_prefix='site', 
    #             filepath=os.path.join(save_path, 'sites.csv'))
    
    # # Site
    # create_nodes(node_names=SITE_IDS, 
    #             node_type='Site', 
    #             node_prefix='site', 
    #             node_properties=SITE_PROPERTIES,
    #             filepath=os.path.join(save_path, 'site.csv'))

    print('Finished creating nodes')

if __name__ == '__main__':
    main()
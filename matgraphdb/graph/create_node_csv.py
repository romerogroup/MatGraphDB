import os
import json
from typing import List,Dict
from glob import glob

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.graph.material_graph import NodeTypes
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


    for i, node_name in enumerate(node_names[:]):
        node_dict[f'{node_prefix}Id:ID({node_prefix}-ID)'].append(i)
        node_dict['type:LABEL'].append(node_type)
        node_dict['name:string'].append(node_name.replace(':', '_'))

        # Add the extra attributes to the node_dict
        if node_properties:
            for property_name, property_value in node_properties[i].items():
                if isinstance(property_value, List ):
                    property_value = ";".join([str(item) for item in property_value])

                node_dict[property_name].append(property_value)

    df = pd.DataFrame(node_dict)

    if filepath:
        df.to_csv(filepath, index=False)

    return df


def main():
    save_path = os.path.join(NODE_DIR)

    print('Save_path:', save_path)
    os.makedirs(save_path, exist_ok=True)
    
    print('Creating nodes...')
    node_types=NodeTypes()

    # # Elements
    # elements,elements_properties,element_id_map=node_types.get_element_nodes()
    # create_nodes(node_names=elements, 
    #             node_type='Element', 
    #             node_prefix='element', 
    #             node_properties=elements_properties,
    #             filepath=os.path.join(save_path, 'elements.csv'))
    
    # # # Crystal Systems
    # crystal_systems,crystal_systems_properties,crystal_system_id_map=node_types.get_crystal_system_nodes()
    # create_nodes(node_names=crystal_systems, 
    #             node_type='CrystalSystem', 
    #             node_prefix='crystalSystem', 
    #             node_properties=crystal_systems_properties,
    #             filepath=os.path.join(save_path, 'crystal_systems.csv'))

    # # Chemenv
    # chemenv_names,chemenv_names_properties,chemenv_name_id_map=node_types.get_chemenv_nodes()
    # create_nodes(node_names=chemenv_names, 
    #             node_type='Chemenv', 
    #             node_prefix='chemenv', 
    #             node_properties=chemenv_names_properties,
    #             filepath=os.path.join(save_path, 'chemenv_names.csv'))
    
    # # Chemenv Element
    # chemenv_element_names,chemenv_element_names_properties,chemenv_element_name_id_map=node_types.get_chemenv_element_nodes()
    # create_nodes(node_names=chemenv_element_names, 
    #             node_type='ChemenvElement', 
    #             node_prefix='chemenvElement', 
    #             filepath=os.path.join(save_path, 'chemenv_element_names.csv'))
    
    # # Magnetic States
    # magnetic_states,magnetic_states_properties,magnetic_state_id_map=node_types.get_magnetic_states_nodes()
    # create_nodes(node_names=magnetic_states, 
    #             node_type='MagneticState', 
    #             node_prefix='magState', 
    #             filepath=os.path.join(save_path, 'magnetic_states.csv'))
    
    # # Space Groups
    # space_groups,space_groups_properties,space_groups_id_map=node_types.get_space_group_nodes()
    # create_nodes(node_names=space_groups, 
    #             node_type='SpaceGroup', 
    #             node_prefix='spg', 
    #             filepath=os.path.join(save_path, 'spg.csv'))
    
    # # Oxidation States
    # oxidation_states,oxidation_states_names,oxidation_state_id_map=node_types.get_oxidation_states_nodes()
    # create_nodes(node_names=oxidation_states, 
    #             node_type='OxidationState', 
    #             node_prefix='oxiState', 
    #             filepath=os.path.join(save_path, 'oxidation_states.csv'))
    
    # # Materials
    # materials,materials_properties,material_id_map=node_types.get_material_nodes()
    # create_nodes(node_names=materials,
    #             node_type='Material',
    #             node_prefix='material',
    #             node_properties=materials_properties,
    #             filepath=os.path.join(save_path, 'materials.csv'))
    
    # # SPG_WYCKOFFS
    # spg_wyckoffs,spg_wyckoff_properties,spg_wyckoff_id_map=node_types.get_wyckoff_positions_nodes()
    # create_nodes(node_names=spg_wyckoffs,
    #             node_type='SPGWyckoff',
    #             node_prefix='spgWyckoff',
    #             node_properties=spg_wyckoff_properties,
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
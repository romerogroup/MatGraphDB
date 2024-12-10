
import os

import pandas as pd
import torch

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from matgraphdb.graph_kit.data import DataGenerator
from sandbox.matgraphdb.graph_kit.graphs import GraphManager
from matgraphdb.utils import PKG_DIR, GRAPH_DIR
from matgraphdb.graph_kit.pyg.algo import feature_propagation


if __name__ == "__main__":
    element_file=os.path.join(PKG_DIR,'utils','interim_periodic_table_values.csv')
    df = pd.read_csv(element_file)

    # We initialize the GraphManager. This will initialize all the base nodes and relationships 
    # if they are not in the graph directory
    manager=GraphManager(graph_dir=os.path.join(GRAPH_DIR,'main'))

    # For our example we want to use periodic table values before the imputation. 
    # So we can overwrite the default element nodes file by the folloiwng
    manager.nodes.get_element_nodes(base_element_csv='interim_periodic_table_values.csv', 
                                           from_scratch=True)
    
    
    # For feature propagation we need a homogenous pytorch geometric Data object.
    # We can easily create this object with the DataGenerator class
    generator=DataGenerator()

    # First we add the element nodes to the graph. Here, we need to specify the feature columns to be used.
    # I select all the features that are float values for this.
    property_names=['abundance_universe','abundance_solar','abundance_meteor','abundance_crust','abundance_ocean','abundance_human',
                    'boiling_point','critical_pressure','critical_temperature','density_stp','conductivity_thermal',
                    'electron_affinity','electronegativity_pauling',
                    'heat_specific','heat_vaporization','heat_fusion','heat_molar',
                    'magnetic_susceptibility_mass','magnetic_susceptibility_molar','magnetic_susceptibility_volume',
                    'melting_point','molar_volume','neutron_cross_section','neutron_mass_absorption',
                    'radius_calculated','radius_empirical','radius_covalent','radius_vanderwaals','refractive_index',
                    'speed_of_sound','conductivity_electric','electrical_resistivity',
                    'modulus_bulk','modulus_shear','modulus_young','poisson_ratio','coefficient_of_linear_thermal_expansion',
                    'hardness_vickers','hardness_brinell','hardness_mohs','superconduction_temperature']
    
    # By default we drop all NaN values. If you want to keep them, you can set keep_nan=True, 
    # which is what we want in feature propatation
    generator.add_node_type(node_path=os.path.join(manager.node_dir,'ELEMENT.parquet'),
                            feature_columns=property_names,
                            keep_nan=True)

    # Next we add the relationships we want to use for feature propagation. 
    # In this case we are using Group-Period relationships, 
    # this is choosen becuase feature propagation demands that the relationships are homophilic. 
    # Meaning the relationships should connect similar elements, which is exactly what Group-Period relationships do
    generator.add_edge_type(edge_path=os.path.join(manager.relationship_dir,'ELEMENT-GROUP_PERIOD_CONNECTS-ELEMENT.parquet'))

    # Now with the nodes and relationships added we can create the homogenous pytorch geometric Data object
    data=generator.homo_data

    print(data.x.shape)

    print(data.edge_index.shape)

    properties_after=feature_propagation(data=data)
    
    df_final = pd.DataFrame(properties_after,columns=property_names)

    df_final.to_csv(os.path.join('examples', 'scripts','imputed_periodic_table_values.csv'))
    # print(len(df_final),len(df))
    print(df_final.head())
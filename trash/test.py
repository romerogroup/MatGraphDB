# import numpy as np

# index_to_replace = 8
# mean, std = 0, 1  # Standard normal distribution parameters
# num_samples = 10
# samples = np.linspace(mean - 3*std, mean + 3*std, num_samples)

# for i,sample in enumerate(samples):
#     print(f"Sample {i+1}: {sample}")
##########################################################################################
# from matgraphdb.utils import OPENAI_API_KEY
# from openai import OpenAI
# client = OpenAI(api_key=OPENAI_API_KEY)

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo-0125",
#   response_format={ "type": "json_object" },
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#     {"role": "user", "content": "Who won the world series in 2020?"}
#   ]
# )
# print(response.choices[0].message.content)
##########################################################################################


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# import os
# import json
# from matgraphdb.utils import DB_DIR,DATA_DIR
# from pymatgen.core import Structure
# from pymatgen.core.periodic_table import Element
# from glob import glob

# files=glob(DB_DIR + '/*.json')

# material_file=files[1]
# print(material_file)

# # index=4 has only 1 geometric bond


# # # Load material data from file
# with open(material_file) as f:
#     db = json.load(f)
#     structure=Structure.from_dict(db['structure'])
#     elec_coord_connections = db['chargemol_bonding_connections']
#     geo_coord_connections = db['coordination_multi_connections']

#     chem_environments=db['coordination_environments_multi_weight']

#     bond_orders=db['chargemol_bonding_orders']
# print(elec_coord_connections)
# print(bond_orders)
# print(geo_coord_connections)
# print('_'*200)


def geometric_electric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders):
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
        print(n_elec_bonds)
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

def electric_consistent_bonds(elec_coord_connections, bond_orders):
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
        print(n_elec_bonds)

        reduced_elec_site_connections = [elec_site_connections[i] for i in elec_reduced_bond_indices]
        reduced_bond_orders = [site_bond_orders[i] for i in elec_reduced_bond_indices]

        final_site_connections=reduced_elec_site_connections
        final_site_bond_orders=reduced_bond_orders
            
        final_connections.append(final_site_connections)
        final_bond_orders.append(final_site_bond_orders)
    return final_connections, final_bond_orders

def geomtric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders):
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

# results= geometric_electric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders)

# print('_'*200)
# print(results)
# bond_orders=results[0]
# for x in bond_orders:
#     print(len(x))

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# import os
# import json
# from matgraphdb.utils import DB_DIR,DATA_DIR
# from pymatgen.core import Structure
# from pymatgen.core.periodic_table import Element
# from pymatgen.io.vasp import Poscar
# from glob import glob

# # Define directories and files
# bonding_benchmark_dir=os.path.join(DATA_DIR, 'bonding')
# poscar_dir=os.path.join(bonding_benchmark_dir, 'POSCARS')


# files=glob(DB_DIR + '/*.json')
# geo_predicted_bonds = {}
# geo_elec_predicted_bonds = {}
# elec_predicted_bonds = {}
# for material_file in files[:20]:
#     print(material_file)
#     # # Load material data from file
#     with open(material_file) as f:
#         bonds_unit_cell = []

#         db = json.load(f)

#         mpid=db['material_id']
#         structure=Structure.from_dict(db['structure'])
        
#         geo_coord_connections = db['coordination_multi_connections']

#         elec_coord_connections = db['chargemol_bonding_connections']
#         chargemol_bond_orders=db['chargemol_bonding_orders']

#         print(elec_coord_connections,chargemol_bond_orders)
#         ##########################################################################################################
#         results = geometric_electric_consistent_bonds(geo_coord_connections,elec_coord_connections, chargemol_bond_orders)
#         bond_orders=results[0]
#         bonds_unit_cell = []
#         for site_bond_orders in bond_orders:
#             n_bonds=len(site_bond_orders)
#             bonds_unit_cell.append(n_bonds)

#         geo_elec_predicted_bonds.update({mpid: bonds_unit_cell})
#         ##########################################################################################################
#         results = electric_consistent_bonds(elec_coord_connections, chargemol_bond_orders)
#         bond_orders=results[0]
#         bonds_unit_cell = []
#         for site_bond_orders in bond_orders:
#             n_bonds=len(site_bond_orders)
#             bonds_unit_cell.append(n_bonds)

#         elec_predicted_bonds.update({mpid: bonds_unit_cell})
#         ###########################################################################################################
#         results = geomtric_consistent_bonds(geo_coord_connections,elec_coord_connections, bond_orders)
#         bond_orders=results[0]
#         bonds_unit_cell = []
#         for site_bond_orders in bond_orders:
#             n_bonds=len(site_bond_orders)
#             bonds_unit_cell.append(n_bonds)

#         geo_predicted_bonds.update({mpid: bonds_unit_cell})
#         ###########################################################################################################
        
#         # save the structure as poscar file
#         poscar=Poscar(structure)
#         poscar.write_file(os.path.join(poscar_dir, f'{mpid}.vasp'))


        

# with open(os.path.join(bonding_benchmark_dir, 'geo_elec_predicted_bonds.json'), 'w') as f:
#     json.dump(geo_elec_predicted_bonds, f)

# with open(os.path.join(bonding_benchmark_dir, 'elec_predicted_bonds.json'), 'w') as f:
#     json.dump(elec_predicted_bonds, f)

# with open(os.path.join(bonding_benchmark_dir, 'geo_predicted_bonds.json'), 'w') as f:
#     json.dump(geo_predicted_bonds, f)

# print(elec_coord_connections)
# print(bond_orders)
# print(geo_coord_connections)
# print('_'*200)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# This section computes the error in the predicted bonds based on a benchmark dataset


import os
import json
from matgraphdb.utils import DB_DIR,DATA_DIR
from pymatgen.core import Structure
from glob import glob


bonding_benchmark_dir=os.path.join(DATA_DIR, 'bonding')
poscar_dir=os.path.join(bonding_benchmark_dir, 'POSCARS')



# Compute error in predicted bonds per site per material
def compute_dataset_error(predicted_bonds, benchmark_bonds):
    """
    Compute the error in the predicted bonds based on a benchmark dataset.

    Args:
        predicted_bonds (dict): Dictionary containing the predicted bonds.
        benchmark_bonds (dict): Dictionary containing the benchmark bonds.

    Returns:
        dict: A dictionary containing the error in the predicted bonds.

    """
    error = {}
    site_count=0
    total_error=0
    for mpid, mat_bonds in predicted_bonds.items():

        mat_benchmark_bonds = benchmark_bonds[mpid]


        error_mpid=[]
        for n_predict_bonds_on_site, n_bonds_on_site in zip(mat_bonds, mat_benchmark_bonds):

            # Handles cases where there are no bonds on the site
            if n_bonds_on_site==0 and n_predict_bonds_on_site==0:
                error_mpid.append(0.0)
            elif n_bonds_on_site==0 and n_predict_bonds_on_site>0:
                error_mpid.append(1.0)
            else:
                error_mpid.append(abs(n_predict_bonds_on_site - n_bonds_on_site) / n_bonds_on_site)

        error.update({mpid: error_mpid})

        total_error+=sum(error_mpid)
        site_count+=len(error_mpid)

    avg_error_per_site=total_error/site_count
    return error,avg_error_per_site


# Opening the benchmark and predicted bonds
with open(os.path.join(bonding_benchmark_dir, 'bonding_benchmark.json')) as f:
    benchmark_bonds = json.load(f)

with open(os.path.join(bonding_benchmark_dir, 'geo_elec_predicted_bonds.json')) as f:
    geo_elec_predicted_bonds = json.load(f)

with open(os.path.join(bonding_benchmark_dir, 'elec_predicted_bonds.json')) as f:
    elec_predicted_bonds = json.load(f)

with open(os.path.join(bonding_benchmark_dir, 'geo_predicted_bonds.json')) as f:
    geo_predicted_bonds = json.load(f)

# Compute the error in the predicted bonds
geo_elec_error,error_per_site = compute_dataset_error(geo_elec_predicted_bonds, benchmark_bonds)
print("Average error per site: ",error_per_site)
print(geo_elec_error)
print('_'*200)

elec_error,error_per_site = compute_dataset_error(elec_predicted_bonds, benchmark_bonds)
print("Average error per site: ",error_per_site)
print(elec_error)
print('_'*200)

geo_error,error_per_site = compute_dataset_error(geo_predicted_bonds, benchmark_bonds)
print("Average error per site: ",error_per_site)
print(geo_error)
print('_'*200)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
# from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
# from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments


# lgf = LocalGeometryFinder()
# lgf.setup_structure(structure=struct)

# # Compute the structure environments
# se = lgf.compute_structure_environments(maximum_distance_factor=1.41, only_cations=False)

# # Define the strategy for environment calculation
# strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
# lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)

# # Get a list of possible coordination environments per site
# coordination_environments = copy.copy(lse.coordination_environments)

# # Replace empty environments with default value
# for i, env in enumerate(lse.coordination_environments):
#     if env is None or env==[]:
#         coordination_environments[i] = [{'ce_symbol': 'S:1', 'ce_fraction': 1.0, 'csm': 0.0, 'permutation': [0]}]

# # Calculate coordination numbers
# coordination_numbers = []
# for env in coordination_environments:
#     if env is None:
#         coordination_numbers.append('NaN')
#     else:
#         coordination_numbers.append(int(env[0]['ce_symbol'].split(':')[-1]))

# # Determine nearest neighbors
# nearest_neighbors = []
# for i_site, neighbors in enumerate(lse.neighbors_sets):

#     neighbor_index = []
#     if neighbors!=[]:
#         neighbors = neighbors[0]
#         for neighbor_site in neighbors.neighb_sites_and_indices:
#             index = neighbor_site['index']
#             neighbor_index.append(index)
#         nearest_neighbors.append(neighbor_index)
#     else:
#         pass












# print(Element)
# element_symbols = [e.symbol for e in Element]
# print(element_symbols)
# oxi_states=[]
# for element in element_symbols:
#     print(element,Element(element).oxidation_states)
#     # oxi_states.append((element,Element(element)._data['Oxidation states']))

# print(oxi_states)


# print(dir(structure))
# print('_'*200)
# print(dir(structure.composition))
# print('_'*200)
# print("Total electrons: ", structure.composition.total_electrons)
# print("Oxidataion state: ", structure.composition.oxi_state_guesses())
# print("Average electronegativity: ", structure.composition.average_electroneg)
# print(Element('Fe')._data)

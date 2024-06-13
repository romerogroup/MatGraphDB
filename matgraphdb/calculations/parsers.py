
import re
import os

from matgraphdb.utils import LOGGER

def parse_chargemol_bond_orders(file, bond_order_cutoff=0.0):
    """
    Parses the Chargemol bond order file and extracts the bonding orders and connections.

    Args:
        file (str): The path to the Chargemol bond order file.
        bond_order_cutoff (float, optional): The minimum bond order cutoff. Bonds with order below this value will be ignored. Defaults to 0.0.

    Returns:
        tuple: A tuple containing two lists. The first list contains the bonding orders for each atom, and the second list contains the atom indices of the connected atoms.
    """
    try:
        with open(file,'r') as f:
            text=f.read()

        bond_blocks=re.findall('(?<=Printing BOs for ATOM).*\n([\s\S]*?)(?=The sum of bond orders for this atom is SBO)',text)

        bonding_connections=[]
        bonding_orders=[]

        for bond_block in bond_blocks:

            bonds=bond_block.strip().split('\n')

            bond_orders=[]
            atom_indices=[]
            # Catches cases where there are no bonds
            if bonds[0]!='':
                for bond in bonds:

                    bond_order=float(re.findall('bond order\s=\s*([.0-9-]*)\s*',bond)[0])

                    # shift so index starts at 0
                    atom_index=int(re.findall('translated image of atom number\s*([0-9]*)\s*',bond)[0]) -1

                    if bond_order >= bond_order_cutoff:
                        bond_orders.append(bond_order)
                        atom_indices.append(atom_index)
            else:
                pass

            bonding_connections.append(atom_indices)
            bonding_orders.append(bond_orders)
    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        bonding_orders=None
        bonding_connections=None

    return bonding_connections,bonding_orders

def parse_chargemol_atomic_moments(file):
    """
    Parses the Chargemol atomic moments file and extracts the atomic moments.

    Args:
        file (str): The path to the Chargemol atomic moments file.

    Returns:
        list: A list containing the atomic moments.
    """
    try:
        with open(file,'r') as f:
            text=f.read()


        raw_atomic_moments_info=re.findall('Same information as above printed with atom number.*\n([\s\S]*)',text)[0].strip().split('\n')
        atomic_moments=[]
        for moment_info_line in raw_atomic_moments_info:
            moment_info=moment_info_line.split()
            moment= float(moment_info[5])
            atomic_moments.append(moment)

    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        atomic_moments=None

    return atomic_moments


def parse_chargemol_net_atomic_charges(file):
    """
    Parses the Chargemol net atomic charges file and extracts the net atomic charges.

    Args:
        file (str): The path to the Chargemol net atomic charges file.

    Returns:
        list: A list containing the net atomic charges.
    """
    try:
        with open(file,'r') as f:
            text=f.read()

        num_atoms=int(text.split('\n')[0])

        parameter_names= ('Number of radial integration shells',
                        'Cutoff radius \(pm\)',
                        'Error in the integrated total number of electrons before renormalization \(e\)',
                        'Charge convergence tolerance',
                        'Minimum radius for electron cloud penetration fitting \(pm\)',
                        'Minimum decay exponent for electron density of buried atom tails',
                        'Maximum decay exponent for electron density of buried atom tails',
                        'Number of iterations to convergence')
        parameters=[]
        for parameter_name in parameter_names:
            reg_expression=parameter_name+'\s*=\s*([-E\d.]+)'
            raw_parameter_name=parameter_name.replace('\\','')
            parameter_value=float(re.findall(reg_expression,text)[0])
            parameters.append((raw_parameter_name,parameter_value))


        raw_info=re.findall('The following XYZ coordinates are in angstroms. The atomic dipoles and quadrupoles are in atomic units.*\n([\s\S]*)',text)[0].split('\n \n')
        
        moment_info=raw_info[0].split('\n')
        moment_info_description='The following XYZ coordinates are in angstroms. The atomic dipoles and quadrupoles are in atomic units'
        moment_info_names=[name.strip() for name in moment_info[0].split(',')]
        moment_info_values=[]
        for raw_values in moment_info[1:]:
            values = [value if i == 1 else float(value) for i, value in enumerate(raw_values.split())]
            moment_info_values.append(values)

        electron_density_fit_description='The sperically averaged electron density of each atom fit to a function of the form exp(a - br) for r >=rmin_cloud_penetration.'
        electron_density_fit_info=re.findall('The sperically averaged electron density of each atom fit to a function of the form exp\(a \- br\) for r \>\=rmin_cloud_penetration.*\n([\s\S]*)',raw_info[1])[0].split('\n')
        electron_density_fit_names=[name.strip() for name in electron_density_fit_info[0].split(',')]
        electron_density_fit_values=[]
        for raw_values in electron_density_fit_info[1:]:
            values = [value if i == 1 else float(value) for i, value in enumerate(raw_values.split())]
            electron_density_fit_values.append(values)

        
        net_atomic_charges_info={
            'moment_info_description':moment_info_description,
            'moment_info_names':moment_info_names,
            'moment_info_values':moment_info_values,
            'electron_density_fit_description':electron_density_fit_description,
            'electron_density_fit_names':electron_density_fit_names,
            'electron_density_fit_values':electron_density_fit_values,
            'computational_parameters':parameters
        }
    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        net_atomic_charges_info=None

    return net_atomic_charges_info

def parse_chargemol_overlap_populations(file):
    """
    Parses the Chargemol overlap populations file and extracts the overlap populations.

    Args:
        file (str): The path to the Chargemol overlap populations file.

    Returns:
        list: A list containing the overlap populations.
    """

    try:
        with open(file,'r') as f:
            text=f.read()

        num_atoms=int(text.split('\n')[0])
        overlap_populations_names='atom1, atom2, translation A, translation B, translation C, overlap population'.split(',')
        raw_overlap_populations_values=re.findall('atom1, atom2, translation A, translation B, translation C, overlap population.*\n([\s\S]*)',text)[0].strip().split('\n')

        overlap_populations_values=[[ float(value) for value in overlap_population_line.split()] for overlap_population_line in raw_overlap_populations_values]

        overlap_populations_info={
            'overlap_populations_names':overlap_populations_names,
            'overlap_populations_values':overlap_populations_values
        }
    except Exception as e:
        LOGGER.error(f"Error processing file: {e}")
        overlap_populations_info=None
    return overlap_populations_info


if __name__=='__main__':


    squared_moments_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/DDEC_atomic_Rsquared_moments.xyz'
    cubed_moments_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/DDEC_atomic_Rcubed_moments.xyz'
    fourth_moments_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/DDEC_atomic_Rfourth_moments.xyz'
    bond_orders_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/DDEC6_even_tempered_bond_orders.xyz'
    atomic_charges_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/DDEC6_even_tempered_net_atomic_charges.xyz'
    overlap_population_file='/users/lllang/SCRATCH/projects/MatGraphDB/data/raw/materials_project_nelements_2/calculations/MaterialsData/mp-170/chargemol/overlap_populations.xyz'
    
    
    # bond_order_info= parse_chargemol_bond_orders(file=bond_orders_file)

    # for bond_orders, neihbors in zip(*bond_order_info):
    #     print(bond_orders)
    #     print(neihbors)
    #     print('_'*200)   
    squared_moments_file='/gpfs20/scratch/lllang/projects/MatGraphDB/data/production/materials_project/calculations/MaterialsData/mp-1228566/chargemol/DDEC_atomic_Rsquared_moments.xyz'
    print(squared_moments_file)
    squared_moments_info=parse_chargemol_atomic_moments(file=squared_moments_file)
    print(squared_moments_info)
    # cubed_moments_info=parse_chargemol_atomic_moments(file=cubed_moments_file)
    # fourth_moments_info=parse_chargemol_atomic_moments(file=fourth_moments_file)

    # print(squared_moments_info)
    # print(cubed_moments_info)
    # print(fourth_moments_info)

    # net_atomic_charges_info=parse_chargemol_net_atomic_charges(file=atomic_charges_file)
    # print(net_atomic_charges_info)

    # overlap_population_info=parse_chargemol_overlap_populations(file=overlap_population_file)
    # print(overlap_population_info)
import os
from glob import glob
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from pymatgen.core.periodic_table import Element

from pymatgen.core import Structure
import pandas as pd

from matgraphdb.utils.periodic_table import atomic_symbols
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import DB_DIR
from matgraphdb.utils import LOGGER, ENCODING_DIR
from matgraphdb.data.manager import DBManager

# TODO: Need a better way to define node types and their properties.
# TODO: Use built-in periodic_table module for element properties

class NodeTypes:
    def __init__(self):
        self.db_manager=DBManager()

    def get_element_nodes(self):
        elements = atomic_symbols[1:]
        element_id_map = {element: i for i, element in enumerate(elements)}
        elements_properties = []
        for i, element in enumerate(elements[:]):
            # pymatgen object. Given element string, will have useful properties
            pmat_element = Element(element)

            # Handling None and nan value cases
            if str(pmat_element.Z) != 'nan':
                atomic_number = pmat_element.Z
            else:
                atomic_number = None
            if str(pmat_element.X) != 'nan':
                x = pmat_element.X
            else:
                x = None
            if str(pmat_element.atomic_radius) != 'nan' and str(pmat_element.atomic_radius) != 'None':
                atomic_radius = float(pmat_element.atomic_radius)
            else:
                atomic_radius = None
            if str(pmat_element.group) != 'nan':
                group = pmat_element.group
            else:
                group = None
            if str(pmat_element.row) != 'nan':
                row = pmat_element.row
            else:
                row = None
            if str(pmat_element.atomic_mass) != 'nan':
                atomic_mass = float(pmat_element.atomic_mass)
            else:
                atomic_mass = None

            elements_properties.append({"element_name:string": element,
                                    "atomic_number:float": atomic_number,
                                    "X:float": x,
                                    "atomic_radius:float": atomic_radius,
                                    "group:int": group,
                                    "row:int": row,
                                    "atomic_mass:float": atomic_mass})
        return elements, elements_properties, element_id_map
    
    def get_crystal_system_nodes(self):
        crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic',
                           'tetragonal', 'trigonal', 'hexagonal', 'cubic']
        crystal_system_id_map = {name: i for i, name in enumerate(crystal_systems)}
        crystal_systems_properties = []
        for i, crystal_system in enumerate(crystal_systems[:]):
            crystal_systems_properties.append({"crystal_system:string": crystal_system})
        return crystal_systems, crystal_systems_properties, crystal_system_id_map

    def get_magnetic_states_nodes(self):
        magnetic_states = ['NM', 'FM', 'FiM', 'AFM', 'Unknown']
        magnetic_state_id_map = {name: i for i, name in enumerate(magnetic_states)}
        magnetic_states_properties = []
        for i, magnetic_state in enumerate(magnetic_states[:]):
            magnetic_states_properties.append({"magnetic_state:string": magnetic_state})
        return magnetic_states, magnetic_states_properties, magnetic_state_id_map
    
    def get_oxidation_states_nodes(self):
        oxidation_states = np.arange(-9, 10)
        oxidation_states_names = [f'ox_{i}' for i in oxidation_states]
        oxidation_state_id_map = {name: i for i, name in enumerate(oxidation_states_names)}
        oxidation_states_properties = []
        for i, oxidation_state in enumerate(oxidation_states_names):
            oxidation_number = oxidation_state.split('_')[1]
            oxidation_states_properties.append({"oxidation_state:float": oxidation_state})
        return oxidation_states_names, oxidation_states_properties, oxidation_state_id_map
    
    def get_space_group_nodes(self):
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        space_groups_id_map = {name: i for i, name in enumerate(space_groups)}
        space_groups_properties = []
        for i, space_group in enumerate(space_groups[:]):
            spg_num=space_group.split('_')[1]
            space_groups_properties.append({"space_group:int": spg_num})
        return space_groups, space_groups_properties, space_groups_id_map
    
    def get_chemenv_nodes(self):
        chemenv_names = list(mp_coord_encoding.keys())
        chemenv_name_id_map = {name: i for i, name in enumerate(chemenv_names)}
        chemenv_names_properties = []
        for i, chemenv_name in enumerate(chemenv_names):
            coordination = int(chemenv_name.split(':')[1])
            chemenv_names_properties.append({"chemenv_name:string": chemenv_name, "coordination:int": coordination})
        return chemenv_names, chemenv_names_properties, chemenv_name_id_map
    
    def get_chemenv_element_nodes(self):
        chemenv_names = list(mp_coord_encoding.keys())
        elements = atomic_symbols[1:]
        chemenv_element_names = []
        for element_name in elements:
            for chemenv_name in chemenv_names:
                
                class_name = element_name + '_' + chemenv_name
                chemenv_element_names.append(class_name)

        chemenv_element_name_id_map = {name: i for i, name in enumerate(chemenv_element_names)}

        chemenv_element_names_properties = []
        for i, chemenv_element_name in enumerate(chemenv_element_names):
            element_name=chemenv_element_name.split('_')[0]
            pmat_element = Element(element_name)

            # Handling None and nan value cases
            if str(pmat_element.Z) != 'nan':
                atomic_number = pmat_element.Z
            else:
                atomic_number = None
            if str(pmat_element.X) != 'nan':
                x = pmat_element.X
            else:
                x = None
            if str(pmat_element.atomic_radius) != 'nan' and str(pmat_element.atomic_radius) != 'None':
                atomic_radius = float(pmat_element.atomic_radius)
            else:
                atomic_radius = None
            if str(pmat_element.group) != 'nan':
                group = pmat_element.group
            else:
                group = None
            if str(pmat_element.row) != 'nan':
                row = pmat_element.row
            else:
                row = None
            if str(pmat_element.atomic_mass) != 'nan':
                atomic_mass = float(pmat_element.atomic_mass)
            else:
                atomic_mass = None
            coordination = int(chemenv_element_name.split(':')[1])
            chemenv_element_names_properties.append({
                                    "chemenv_element_name:string": chemenv_element_name,
                                    "atomic_number:float": atomic_number,
                                    "X:float": x,
                                    "atomic_radius:float": atomic_radius,
                                    "group:int": group,
                                    "row:int": row,
                                    "atomic_mass:float": atomic_mass,
                                    "coordination:int": coordination
                                    })
        return chemenv_element_names, chemenv_element_names_properties, chemenv_element_name_id_map
    
    def get_wyckoff_positions_nodes(self):
        space_groups = [f'spg_{i}' for i in np.arange(1, 231)]
        wyckoff_letters = ['a', 'b', 'c', 'd', 'e', 'f']
        spg_wyckoffs = []
        for wyckoff_letter in wyckoff_letters:
            for spg_name in space_groups:
                spg_wyckoffs.append(spg_name + '_' + wyckoff_letter)
        spg_wyckoff_id_map = {name: i for i, name in enumerate(spg_wyckoffs)}
        spg_wyckoff_properties = []
        for i, spg_wyckoff in enumerate(spg_wyckoffs):
            spg_wyckoff_properties.append({"spg_wyckoff:string": spg_wyckoff})
        return spg_wyckoffs, spg_wyckoff_properties, spg_wyckoff_id_map
    
    def screen_materials(
                        material_dict,
                        material_ids:List[str]=None, 
                        elements:List[str]=None,
                        nelements:Tuple[int,int]=None,
                        crystal_systems:List[str]=None):
        """
        Retrieves materials from the database based on specified criteria.

        Args:
            material_dict (dict): A dictionary containing the material data.
            material_ids (List[str], optional): List of material IDs to filter the results. Defaults to None.
            elements (List[str], optional): List of elements to filter the results. Defaults to None.
            crystal_systems (List[str], optional): List of crystal systems to filter the results. Defaults to None.
            
        Returns:
            results: If material meets the conditions, return True. Otherwise, return False.
        """

        if elements:
            for i,element in enumerate(material_dict['elements']):
                if element in elements:
                    return True
        
        if nelements:
            min_nelements=nelements[0]
            max_nelements=nelements[1]
            if min_nelements <= material_dict['nelements'] <= max_nelements:
                return True

        if material_ids:
            for i,material_id in enumerate(material_ids):
                if material_id in material_dict['material_id']:
                    return True

        if crystal_systems:
            for crystal_system in crystal_systems:
                if crystal_system in [material_dict['symmetry']['crystal_system']]:
                    return True

        return False

    def get_material_nodes_task(self, json_file,**kwargs):
                        
        try:
            with open(json_file) as f:
                data = json.load(f)
                structure = Structure.from_dict(data['structure'])
        except Exception as e:
            LOGGER.error(f"Error processing file {json_file}: {e}")
            return None, None, None, None

        mpid_name = json_file.split(os.sep)[-1].split('.')[0]
        mpid_name = mpid_name.replace('-', '_')
        lattice_name='lattice_'+mpid_name
        lattice_properties_dict = {'a:float': structure.lattice.a,
                               'b:float': structure.lattice.b,
                               'c:float': structure.lattice.c,
                               'alpha:float': structure.lattice.alpha,
                               'beta:float': structure.lattice.beta,
                               'gamma:float': structure.lattice.gamma}
        

        material_property_dict = {}

        material_property_dict['material_id:string']=data.get('material_id')
        material_property_dict['nsites:int']=data.get('nsites')
        material_property_dict['nelements:int']=data.get('nelements')
        material_property_dict['elements:string[]']=data.get('elements')
        material_property_dict['composition:string']=data.get('composition')
        material_property_dict['composition_reduced:string']=data.get('composition_reduced')
        material_property_dict['formula_pretty:string']=data.get('formula_pretty')
        material_property_dict['volume:float']=data.get('volume')
        material_property_dict['density:float']=data.get('density')
        material_property_dict['density_atomic:float']=data.get('density_atomic')
        
        material_property_dict['crystal_system:string']=data.get('symmetry').get('crystal_system')
        material_property_dict['space_group:int']=data.get('symmetry').get('number')
        material_property_dict['point_group:string']=data.get('symmetry').get('point_group')
        material_property_dict['hall_symbol:string']=data.get('symmetry').get('symbol')

        # material_property_dict['structure:string']=data.get('structure')

        material_property_dict['energy_per_atom:float']=data.get('energy_per_atom')
        material_property_dict['formation_energy_per_atom:float']=data.get('formation_energy_per_atom')
        material_property_dict['energy_above_hull:float']=data.get('energy_above_hull')
        material_property_dict['is_stable:boolean']=data.get('is_stable')
        material_property_dict['band_gap:float']=data.get('band_gap')
        material_property_dict['cbm:float']=data.get('cbm')
        material_property_dict['vbm:float']=data.get('vbm')
        material_property_dict['efermi:float']=data.get('efermi')
        material_property_dict['is_gap_direct:boolean']=data.get('is_gap_direct')
        material_property_dict['is_metal:boolean']=data.get('is_metal')
        material_property_dict['is_magnetic:boolean']=data.get('is_magnetic')
        material_property_dict['ordering:string']=data.get('ordering')
        material_property_dict['total_magnetization:float']=data.get('total_magnetization')
        material_property_dict['total_magnetization_normalized_vol:float']=data.get('total_magnetization_normalized_vol')
        material_property_dict['num_magnetic_sites:int']=data.get('num_magnetic_sites')

        material_property_dict['e_total:float']=data.get('e_total')
        material_property_dict['e_ionic:float']=data.get('e_ionic')
        material_property_dict['e_electronic:float']=data.get('e_electronic')

        elasticity_dict = data.get('elasticity')
        elastic_properties=[
            'k_voigt:float',
            'k_reuss:float',
            'k_vrh:float',
            'g_voigt:float',
            'g_reuss:float',
            'g_vrh:float',
            'sound_velocity_transverse:float',
            'sound_velocity_longitudinal:float',
            'sound_velocity_total:float',
            'sound_velocity_acoustic:float',
            'sound_velocity_optical:float',
            'thermal_conductivity_clarke:float',
            'thermal_conductivity_cahill:float',
            'youngs_modulus:float',
            'universal_anisotropy:float',
            'homogeneous_poisson:float',
            'debye_temperature:float',
            'state:string']
        for elastic_property in elastic_properties:
            elastic_property_name=elastic_property.split(':')[0]
            if elasticity_dict is not None:
                material_property_dict[elastic_property]=elasticity_dict.get(elastic_property_name)
            else:
                material_property_dict[elastic_property]=None

        oxidation_states_dict = data.get('oxidation_states')
        oxidation_stats_properties=[
            'possible_species:string[]',
            'possible_valences:int[]',
            'average_oxidation_state:string']
        for oxidation_stat_property in oxidation_stats_properties:
            oxidation_stat_property_name=oxidation_stat_property.split(':')[0]
            if oxidation_states_dict is not None:
                material_property_dict[oxidation_stat_property]=oxidation_states_dict.get(oxidation_stat_property_name)
            else:
                material_property_dict[oxidation_stat_property]=None


        material_property_dict['uncorrected_energy_per_atom:float']=data.get('uncorrected_energy_per_atom')
        material_property_dict['equilibrium_reaction_energy_per_atom:float']=data.get('equilibrium_reaction_energy_per_atom')
        material_property_dict['n:float']=data.get('n')
        material_property_dict['e_ij_max:float']=data.get('e_ij_max')
        material_property_dict['weighted_surface_energy_EV_PER_ANG2:float']=data.get('weighted_surface_energy_EV_PER_ANG2')
        material_property_dict['weighted_surface_energy:float']=data.get('weighted_surface_energy')
        material_property_dict['weighted_work_function:float']=data.get('weighted_work_function')
        material_property_dict['surface_anisotropy:float']=data.get('surface_anisotropy')
        material_property_dict['shape_factor:float']=data.get('shape_factor')

        feature_vectors_dict = data.get('feature_vectors')
        feature_vectors=[
            'sine_coulomb_matrix:float[]',
            'element_fraction:float[]',
            'element_property:float[]',
            'xrd_pattern:float[]',
        ]
        
        for feature_vector in feature_vectors:
            feature_vector_property_name=feature_vector.split(':')[0]
            if feature_vectors_dict is not None:
                tmp_dict=feature_vectors_dict.get(feature_vector_property_name)
                if tmp_dict is not None:
                    material_property_dict[feature_vector]=tmp_dict.get('values')
            else:
                material_property_dict[feature_vector]=None

        # if os.path.exists(ENCODING_DIR):
        #     encoding_file=os.path.join(ENCODING_DIR,mpid_name.replace('_', '-')+'.json')
        #     with open(encoding_file) as f:
        #         encoding_dict = json.load(f)

        #     for encoding_name,encoding in encoding_dict.items():
        #         tmp_encoding_list=[str(i) for i in encoding]
        #         encoding=';'.join(tmp_encoding_list)

        #         material_property_dict.update({f'{encoding_name}:float[]':encoding})

        #     mpid=encoding_file.split(os.sep)[-1].split('.')[0]

        return mpid_name, material_property_dict, lattice_name, lattice_properties_dict
    
    def get_material_nodes(self):
        files=self.db_manager.database_files()
        results=self.db_manager.process_task(self.get_material_nodes_task, files)
        material_ids=[]
        material_properties=[]
        lattice_ids=[]
        lattice_properties=[]
        for result in results:
            if result[0] is None:
                continue
            
            material_ids.append(result[0])
            material_properties.append(result[1])
            lattice_ids.append(result[2])
            lattice_properties.append(result[3])
            
        material_id_map = {name: i for i, name in enumerate(material_ids)}
        return material_ids, material_properties, material_id_map




if __name__=='__main__':
    nodes=NodeTypes()
    # names,properties,id_map=nodes.get_chemenv_element_nodes()
    # print(properties[:10])


    # names,properties,id_map=nodes.get_chemenv_nodes()
    # print(properties[:10])

    names,properties,id_map=nodes.get_material_nodes()
    # print(properties[:10])
    # print(names[:10])

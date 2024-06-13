import os
import shutil
from glob import glob
from typing import List, Tuple, Union

import pandas as pd

from matgraphdb import DBManager
from matgraphdb.utils import DB_DIR, DB_CALC_DIR, N_CORES, GLOBAL_PROP_FILE, ENCODING_DIR, MAIN_GRAPH_DIR, GRAPH_DIR,LOGGER
from matgraphdb.graph.create_node_csv import create_nodes
from matgraphdb.graph.create_relationship_csv import (create_relationships,create_bonding_task, create_chemenv_element_task, create_material_element_task, 
                                                      create_material_chemenv_task, create_material_chemenvElement_task, create_material_spg_task,
                                                      create_material_crystal_system_task)
from matgraphdb.graph.node_types import NodeTypes

def is_in_range(val:Union[float, int],min_val:Union[float, int],max_val:Union[float, int], negation:bool=True):
    """
    Screens a list of floats to keep only those that are within a given range.

    Args:
        floats (Union[float, int]): A list of floats to be screened.
        min_val (float): The minimum value to keep.
        max_val (float): The maximum value to keep.
        negation (bool, optional): Whether to use the negation condition. Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is within the given range.
    """
    keep=False
    if min_val <= val <= max_val and negation:
        keep=True
    elif min_val >= val or val >= max_val and not negation:
        keep=True
    return keep

def is_in_list(val,string_list:List, negation:bool=True):
    """
    Screens a value to keep only those that are in a given list.

    Args:
        val : The value to be screened.
        string_list (List): The list to check against.
        negation (bool, optional): Whether to use the negation of the list. Defaults to True.

    Returns:
        bool: A boolean indicating whether value is in the list.
    """
    keep=False
    if val in string_list and negation:
        keep=True
    elif val not in string_list and not negation:
        keep=True
    return keep

class GraphGenerator:
    def __init__(self, directory_path=DB_DIR, calc_path=DB_CALC_DIR, from_scratch=False, n_cores=N_CORES):
        """
        Initializes the GraphGenerator object.

        Args:
            directory_path (str): The path to the directory where the database is stored.
            calc_path (str): The path to the directory where calculations are stored.
            n_cores (int): The number of CPU cores to be used for parallel processing.

        """
        self.directory_path = directory_path
        self.calculation_path = calc_path
        self.from_scratch = from_scratch
        self.n_cores = N_CORES
        self.node_types=NodeTypes()
        self.db_manager = DBManager()

        self.main_node_dir=os.path.join(MAIN_GRAPH_DIR,'nodes')
        self.main_relationship_dir=os.path.join(MAIN_GRAPH_DIR,'relationships')
        
        os.makedirs(self.main_node_dir,exist_ok=True)
        os.makedirs(self.main_relationship_dir,exist_ok=True)
        
        # Initialize the main nodes
        self.initialize_nodes(node_dir=self.main_node_dir,from_scratch=from_scratch)
        self.initialize_relationships(node_dir=self.main_node_dir,relationship_dir=self.main_relationship_dir,from_scratch=from_scratch)

    def get_node_id_maps(self,node_dir=None,graph_dir=None):
        if node_dir is None and graph_dir is None:
            raise Exception("Either node_dir or graph_dir must be provided")
        if node_dir:
            files=glob(os.path.join(node_dir,'*.csv'))
        if graph_dir:
            files=glob(os.path.join(graph_dir,'nodes', '*.csv'))

        all_maps = {}
        for file in files:
            filename=file.split(os.sep)[-1].split(',')[0]
            df=pd.read_csv(file,index_col=0)
            # Create the node name id map
            node_name_id_map = df['name:string'].to_dict()
            all_maps[filename] = node_name_id_map
        return all_maps

    def initialize_nodes(self,node_dir, from_scratch=False):
  
        # Materials
        if not os.path.exists(os.path.join(node_dir,'materials.csv')) or from_scratch:
            LOGGER.info("Creating material nodes")
            materials,materials_properties,material_id_map=self.node_types.get_material_nodes()
            create_nodes(node_names=materials,
                        node_type='Material',
                        node_prefix='material',
                        node_properties=materials_properties,
                        filepath=os.path.join(node_dir, 'materials.csv'))
        # Elements
        if not os.path.exists(os.path.join(node_dir,'elements.csv')) or from_scratch:
            LOGGER.info("Creating element nodes")
            elements,elements_properties,element_id_map=self.node_types.get_element_nodes()
            create_nodes(node_names=elements, 
                        node_type='Element', 
                        node_prefix='element', 
                        node_properties=elements_properties,
                        filepath=os.path.join(node_dir, 'elements.csv'))
        # Crystal Systems
        if not os.path.exists(os.path.join(node_dir,'crystal_systems.csv')) or from_scratch:
            LOGGER.info("Creating crystal system nodes")
            crystal_systems,crystal_systems_properties,crystal_system_id_map=self.node_types.get_crystal_system_nodes()
            create_nodes(node_names=crystal_systems, 
                        node_type='CrystalSystem', 
                        node_prefix='crystalSystem', 
                        # node_properties=crystal_systems_properties,
                        filepath=os.path.join(node_dir, 'crystal_systems.csv'))
        # Chemenv
        if not os.path.exists(os.path.join(node_dir,'chemenv.csv')) or from_scratch:
            LOGGER.info("Creating chemenv nodes")
            chemenv_names,chemenv_names_properties,chemenv_name_id_map=self.node_types.get_chemenv_nodes()
            create_nodes(node_names=chemenv_names, 
                        node_type='Chemenv', 
                        node_prefix='chemenv', 
                        # node_properties=chemenv_names_properties,
                        filepath=os.path.join(node_dir, 'chemenv.csv'))
        # Chemenv Element
        if not os.path.exists(os.path.join(node_dir,'chemenv_element.csv')) or from_scratch:
            LOGGER.info("Creating chemenv element nodes")
            chemenv_element_names,chemenv_element_names_properties,chemenv_element_name_id_map=self.node_types.get_chemenv_element_nodes()
            create_nodes(node_names=chemenv_element_names, 
                        node_type='ChemenvElement', 
                        node_prefix='chemenvElement', 
                        filepath=os.path.join(node_dir, 'chemenv_element.csv'))
        # Magnetic States
        if not os.path.exists(os.path.join(node_dir,'magnetic_states.csv')) or from_scratch:
            LOGGER.info("Creating magnetic state nodes")
            magnetic_states,magnetic_states_properties,magnetic_state_id_map=self.node_types.get_magnetic_states_nodes()
            create_nodes(node_names=magnetic_states, 
                        node_type='MagneticState', 
                        node_prefix='magState', 
                        filepath=os.path.join(node_dir, 'magnetic_states.csv'))
        # Space Groups
        if not os.path.exists(os.path.join(node_dir,'spg.csv')) or from_scratch:
            LOGGER.info("Creating space group nodes")
            space_groups,space_groups_properties,space_groups_id_map=self.node_types.get_space_group_nodes()
            create_nodes(node_names=space_groups, 
                        node_type='SpaceGroup', 
                        node_prefix='spg', 
                        filepath=os.path.join(node_dir, 'spg.csv'))
        # Oxidation States
        if not os.path.exists(os.path.join(node_dir,'oxidation_states.csv')) or from_scratch:
            LOGGER.info("Creating oxidation state nodes")
            oxidation_states,oxidation_states_names,oxidation_state_id_map=self.node_types.get_oxidation_states_nodes()
            create_nodes(node_names=oxidation_states, 
                        node_type='OxidationState', 
                        node_prefix='oxiState', 
                        filepath=os.path.join(node_dir, 'oxidation_states.csv'))
        
        # SPG_WYCKOFFS
        if not os.path.exists(os.path.join(node_dir,'spg_wyckoff.csv')) or from_scratch:
            LOGGER.info("Creating space group wyckoff nodes")
            spg_wyckoffs,spg_wyckoff_properties,spg_wyckoff_id_map=self.node_types.get_wyckoff_positions_nodes()
            create_nodes(node_names=spg_wyckoffs,
                        node_type='SPGWyckoff',
                        node_prefix='spgWyckoff',
                        # node_properties=spg_wyckoff_properties,
                        filepath=os.path.join(node_dir, 'spg_wyckoff.csv'))
        return None

    def initialize_relationships(self,node_dir,relationship_dir,from_scratch=False):

        # Element - Element Connections
        if not os.path.exists(os.path.join(relationship_dir,f'element_element_geometric-electric.csv')) or from_scratch:
            LOGGER.info("Creating element-element geometric-electric relationship")
            create_relationships(node_a_csv=os.path.join(node_dir,'elements.csv'),
                                node_b_csv=os.path.join(node_dir,'elements.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric_electric'},
                                connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'element_element_geometric-electric.csv'),
                                )
        if not os.path.exists(os.path.join(relationship_dir,f'element_element_geometric.csv')) or from_scratch:
            LOGGER.info("Creating element-element geometric relationship")
            create_relationships(node_a_csv=os.path.join(node_dir,'elements.csv'),
                                node_b_csv=os.path.join(node_dir,'elements.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric'},
                                connection_name='GEOMETRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'element_element_geometric.csv'))
        if not os.path.exists(os.path.join(relationship_dir,f'element_element_electric.csv')) or from_scratch:
            LOGGER.info("Creating element-element electric relationship")
            create_relationships(node_a_csv=os.path.join(node_dir,'elements.csv'),
                                node_b_csv=os.path.join(node_dir,'elements.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'), 
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'electric'}, 
                                connection_name='ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'element_element_electric.csv'))
            
        # Chemenv - Chemenv Connections
        if not os.path.exists(os.path.join(relationship_dir,f'chemenv_chemenv_geometric-electric.csv')) or from_scratch:
            LOGGER.info("Creating Chemenv - Chemenv Geometric-Electric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv.csv'), 
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric_electric'},
                                connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'chemenv_chemenv_geometric-electric.csv'))
        if not os.path.exists(os.path.join(relationship_dir,f'chemenv_chemenv_geometric.csv')) or from_scratch:
            LOGGER.info("Creating Chemenv - Chemenv Geometric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric'}, 
                                connection_name='GEOMETRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'chemenv_chemenv_geometric.csv'))
        if not os.path.exists(os.path.join(relationship_dir,f'chemenv_chemenv_electric.csv')) or from_scratch:
            LOGGER.info("Creating Chemenv - Chemenv Electric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'electric'},
                                connection_name='ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,f'chemenv_chemenv_electric.csv'))
            
        # ChemenvElement - ChemenvElement Connections
        if not os.path.exists(os.path.join(relationship_dir,f'chemenvElement_chemenvElement_geometric-electric.csv')) or from_scratch:
            LOGGER.info("Creating ChemenvElement - ChemenvElement Geometric-Electric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric_electric'},
                                connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,'chemenvElement_chemenvElement_geometric-electric.csv'))
        
        if not os.path.exists(os.path.join(relationship_dir,f'chemenvElement_chemenvElement_geometric.csv')) or from_scratch:
            LOGGER.info("Creating ChemenvElement - ChemenvElement Geometric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'geometric'},
                                connection_name='GEOMETRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,'chemenvElement_chemenvElement_geometric.csv'))
        
        if not os.path.exists(os.path.join(relationship_dir,f'chemenvElement_chemenvElement_electric.csv')) or from_scratch:
            LOGGER.info("Creating ChemenvElement - ChemenvElement Electric Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_bonding_task,
                                mp_task_params={'bonding_method':'electric'},
                                connection_name='ELECTRIC_CONNECTS',
                                filepath=os.path.join(relationship_dir,'chemenvElement_chemenvElement_electric.csv'))

        # Chemenv - Element Connections
        if not os.path.exists(os.path.join(relationship_dir,f'chemenv_elements.csv')) or from_scratch:
            LOGGER.info("Creating Chemenv - Element Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                                node_b_csv=os.path.join(node_dir,'elements.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'), 
                                mp_task=create_chemenv_element_task,
                                connection_name='CAN_OCCUR',
                                filepath=os.path.join(relationship_dir,f'chemenv_elements.csv'))
        
        # Material - Element Connections
        if not os.path.exists(os.path.join(relationship_dir,f'materials_elements.csv')) or from_scratch:
            LOGGER.info("Creating Material - Element Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'materials.csv'),
                                node_b_csv=os.path.join(node_dir,'elements.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_material_element_task,
                                connection_name='COMPOSED_OF',
                                filepath=os.path.join(relationship_dir,f'materials_elements.csv'))
            
        # Material - Chemenv Connections
        if not os.path.exists(os.path.join(relationship_dir,f'materials_chemenv.csv')) or from_scratch:
            LOGGER.info("Creating Material - Chemenv Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'materials.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_material_chemenv_task,
                                connection_name='COMPOSED_OF',
                                filepath=os.path.join(relationship_dir,f'materials_chemenv.csv'))
        
        # Material - ChemenvElement Connections
        if not os.path.exists(os.path.join(relationship_dir,f'materials_chemenvElement.csv')) or from_scratch:
            LOGGER.info("Creating Material - ChemenvElement Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'materials.csv'),
                                node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_material_chemenvElement_task,
                                connection_name='COMPOSED_OF',
                                filepath=os.path.join(relationship_dir,f'materials_chemenvElement.csv'))
            
        # Material - spg Connections
        if not os.path.exists(os.path.join(relationship_dir,f'materials_spg.csv')) or from_scratch:
            LOGGER.info("Creating Material - spg Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'materials.csv'),
                                node_b_csv=os.path.join(node_dir,'spg.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_material_spg_task,
                                connection_name='HAS_SPACE_GROUP_SYMMETRY',
                                filepath=os.path.join(relationship_dir,f'materials_spg.csv'))

        # Material - crystal_system Connections
        if not os.path.exists(os.path.join(relationship_dir,f'materials_crystal_system.csv')) or from_scratch:
            LOGGER.info("Creating Material - crystal_system Connections")
            create_relationships(node_a_csv=os.path.join(node_dir,'materials.csv'),
                                node_b_csv=os.path.join(node_dir,'crystal_systems.csv'),
                                material_csv=os.path.join(node_dir,'materials.csv'),
                                mp_task=create_material_crystal_system_task,
                                connection_name='HAS_CRYSTAL_SYSTEM',
                                filepath=os.path.join(relationship_dir,f'materials_crystal_system.csv'))

    def screen_material_nodes(self,
                        material_csv:str,
                        include:bool=True,
                        material_ids:List[str]=None, 
                        elements:List[str]=None,
                        compositions:List[str]=None,
                        space_groups:List[int]=None,
                        point_groups:List[str]=None,
                        magnetic_states:List[str]=None,
                        crystal_systems:List[str]=None,
                        nsites:Tuple[int,int]=None,
                        nelements:Tuple[int,int]=None,
                        energy_per_atom:Tuple[float,float]=None,
                        formation_energy_per_atom:Tuple[float,float]=None,
                        energy_above_hull:Tuple[float,float]=None,
                        band_gap:Tuple[float,float]=None,
                        cbm:Tuple[float,float]=None,
                        vbm:Tuple[float,float]=None,
                        efermi:Tuple[float,float]=None,
                        k_voigt:Tuple[float,float]=None,
                        k_reuss:Tuple[float,float]=None,
                        k_vrh:Tuple[float,float]=None,
                        g_voigt:Tuple[float,float]=None,
                        g_reuss:Tuple[float,float]=None,
                        g_vrh:Tuple[float,float]=None,
                        universal_anisotropy:Tuple[float,float]=None,
                        homogeneous_poisson:Tuple[float,float]=None,
                        is_stable:bool=None,
                        is_gap_direct:bool=None,
                        is_metal:bool=None,
                        is_magnetic:bool=None,
                        ):
        
        df=pd.read_csv(material_csv,index_col=0)
        rows_to_keep=[]
        # Iterate through the rows of the dataframe
        for irow, row in df.iterrows():
            keep_material=False
            if material_ids:
                material_id=row['name:string']
                keep_material=is_in_list(material_id,material_ids)
            if elements:
                material_elements=row['elements:string[]'].split(';')
                for material_element in material_elements:
                    keep_material=is_in_list(val=material_element,string_list=elements, negation=include)
            if magnetic_states:
                material_magnetic_state=row['magnetic_states:string']
                keep_material=is_in_list(val=material_magnetic_state,string_list=magnetic_states, negation=include)
            if crystal_systems:
                material_crystal_system=row['crystal_system:string']
                keep_material=is_in_list(val=material_crystal_system,string_list=crystal_systems, negation=include)
            if compositions:
                material_composition=row['composition:string']
                keep_material=is_in_list(val=material_composition,string_list=compositions, negation=include)
            if space_groups:
                material_space_group=row['space_group:int']
                keep_material=is_in_list(val=material_space_group,string_list=space_groups, negation=include)
            if point_groups:
                material_point_group=row['point_group:string']
                keep_material=is_in_list(val=material_point_group,string_list=point_groups, negation=include)
            if nelements:
                min_nelements=nelements[0]
                max_nelements=nelements[1]
                material_nelements=row['nelements:int']
                keep_material=is_in_range(val=material_nelements,min_val=min_nelements,max_val=max_nelements, negation=include)
            if nsites:
                min_nsites=nsites[0]
                max_nsites=nsites[1]
                material_nsites=row['nsites:int']
                keep_material=is_in_range(val=material_nsites,min_val=min_nsites,max_val=max_nsites, negation=include)
            if energy_per_atom:
                min_energy_per_atom=energy_per_atom[0]
                max_energy_per_atom=energy_per_atom[1]
                material_energy_per_atom=row['energy_per_atom:float']
                keep_material=is_in_range(val=material_energy_per_atom,min_val=min_energy_per_atom,max_val=max_energy_per_atom, negation=include)
            if formation_energy_per_atom:
                min_formation_energy_per_atom=formation_energy_per_atom[0]
                max_formation_energy_per_atom=formation_energy_per_atom[1]
                material_formation_energy_per_atom=row['formation_energy_per_atom:float']
                keep_material=is_in_range(val=material_formation_energy_per_atom,min_val=min_formation_energy_per_atom,max_val=max_formation_energy_per_atom, negation=include)
            if energy_above_hull:
                min_energy_above_hull=energy_above_hull[0]
                max_energy_above_hull=energy_above_hull[1]
                material_energy_above_hull=row['energy_above_hull:float']
                keep_material=is_in_range(val=material_energy_above_hull,min_val=min_energy_above_hull,max_val=max_energy_above_hull, negation=include)
            if band_gap:
                min_band_gap=band_gap[0]
                max_band_gap=band_gap[1]
                material_band_gap=row['band_gap:float']
                keep_material=is_in_range(val=material_band_gap,min_val=min_band_gap,max_val=max_band_gap, negation=include)
            if cbm:
                min_cbm=cbm[0]
                max_cbm=cbm[1]
                material_cbm=row['cbm:float']
                keep_material=is_in_range(val=material_cbm,min_val=min_cbm,max_val=max_cbm, negation=include)
            if vbm:
                min_vbm=vbm[0]
                max_vbm=vbm[1]
                material_vbm=row['vbm:float']
                keep_material=is_in_range(val=material_vbm,min_val=min_vbm,max_val=max_vbm, negation=include)
            if efermi:
                min_efermi=efermi[0]
                max_efermi=efermi[1]
                material_efermi=row['efermi:float']
                keep_material=is_in_range(val=material_efermi,min_val=min_efermi,max_val=max_efermi, negation=include)
            if k_voigt:
                min_k_voigt=k_voigt[0]
                max_k_voigt=k_voigt[1]
                material_k_voigt=row['k_voigt:float']
                keep_material=is_in_range(val=material_k_voigt,min_val=min_k_voigt,max_val=max_k_voigt, negation=include)
            if k_reuss:
                min_k_reuss=k_reuss[0]
                max_k_reuss=k_reuss[1]
                material_k_reuss=row['k_reuss:float']
                keep_material=is_in_range(val=material_k_reuss,min_val=min_k_reuss,max_val=max_k_reuss, negation=include)
            if k_vrh:
                min_k_vrh=k_vrh[0]
                max_k_vrh=k_vrh[1]   
                material_k_vrh=row['k_vrh:float']
                keep_material=is_in_range(val=material_k_vrh,min_val=min_k_vrh,max_val=max_k_vrh, negation=include)
            if g_voigt:
                min_g_voigt=g_voigt[0]
                max_g_voigt=g_voigt[1]
                material_g_voigt=row['g_voigt:float']
                keep_material=is_in_range(val=material_g_voigt,min_val=min_g_voigt,max_val=max_g_voigt, negation=include)
            if g_reuss:
                min_g_reuss=g_reuss[0]
                max_g_reuss=g_reuss[1]
                material_g_reuss=row['g_reuss:float']
                keep_material=is_in_range(val=material_g_reuss,min_val=min_g_reuss,max_val=max_g_reuss, negation=include)
            if g_vrh:
                min_g_vrh=g_vrh[0]
                max_g_vrh=g_vrh[1]
                material_g_vrh=row['g_vrh:float']
                keep_material=is_in_range(val=material_g_vrh,min_val=min_g_vrh,max_val=max_g_vrh, negation=include)
            if universal_anisotropy:
                min_universal_anisotropy=universal_anisotropy[0]
                max_universal_anisotropy=universal_anisotropy[1]
                material_universal_anisotropy=row['universal_anisotropy:float']
                keep_material=is_in_range(val=material_universal_anisotropy,min_val=min_universal_anisotropy,max_val=max_universal_anisotropy, negation=include)
            if homogeneous_poisson:
                min_homogeneous_poisson=homogeneous_poisson[0]
                max_homogeneous_poisson=homogeneous_poisson[1]
                material_homogeneous_poisson=row['homogeneous_poisson:float']
                keep_material=is_in_range(val=material_homogeneous_poisson,min_val=min_homogeneous_poisson,max_val=max_homogeneous_poisson, negation=include)
            if is_stable:
                if not (is_stable ^ include):
                    keep_material=True
            if is_gap_direct:
                if not (is_gap_direct ^ include):
                    keep_material=True
            if is_metal:
                if not (is_metal ^ include):
                    keep_material=True
            if is_magnetic:
                if not (is_magnetic ^ include):
                    keep_material=True

            if keep_material:
                rows_to_keep.append(irow)
        
        filtered_df=df.iloc[rows_to_keep]
        return filtered_df

    def screen_graph_database(self,graph_dirname,from_scratch=False,**kwargs):
        
        
        graph_dir=os.path.join(GRAPH_DIR,graph_dirname)
        if from_scratch:
            LOGGER.info('Starting from scratch')
            shutil.rmtree(graph_dir)

        LOGGER.info('Screening the graph database')
        node_dir=os.path.join(graph_dir,'nodes')
        relationship_dir=os.path.join(graph_dir,'relationships')

        os.makedirs(node_dir,exist_ok=True)
        os.makedirs(relationship_dir,exist_ok=True)

        # Copy all the nodes besides the material node to the graph directory
        node_files=glob(self.main_node_dir+ os.sep +'*.csv')
        for file_paths in node_files:
            filename=os.path.basename(file_paths)
            if filename == "materials.csv":
                continue
            LOGGER.info(f"Copying {filename} to {node_dir}")
            shutil.copy(os.path.join(self.main_node_dir,filename),os.path.join(node_dir,filename))

        LOGGER.info(f"Creating new materials.csv")
        original_materials_file=os.path.join(self.main_node_dir,'materials.csv')
        new_materials_file=os.path.join(node_dir,'materials.csv')
        materials_df=self.screen_material_nodes(material_csv=original_materials_file,**kwargs)
        materials_df.to_csv(new_materials_file,index=True)

        self.initialize_relationships(node_dir=node_dir,relationship_dir=relationship_dir,from_scratch=from_scratch)


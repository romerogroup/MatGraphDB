import os
import shutil
from glob import glob
from typing import List, Tuple, Union

import pandas as pd
import networkx as nx

from matgraphdb import DBManager
from matgraphdb.utils import  GRAPH_DIR, LOGGER
from matgraphdb.graph.create_node_csv import create_nodes
from matgraphdb.graph.create_relationship_csv import (create_relationships,create_bonding_task, create_chemenv_element_task, create_material_element_task, 
                                                      create_material_chemenv_task, create_material_chemenvElement_task, create_material_spg_task,
                                                      create_material_crystal_system_task)
from matgraphdb.graph.node_types import NodeTypes

# TODO Move is_in_range and is_in_list to utils
# TODO Create docstrings for screen materials
# Give unique connection names to the relationships

def is_in_range(val:Union[float, int],min_val:Union[float, int],max_val:Union[float, int], negation:bool=True):
    """
    Screens a list of floats to keep only those that are within a given range.

    Args:
        floats (Union[float, int]): A list of floats to be screened.
        min_val (float): The minimum value to keep.
        max_val (float): The maximum value to keep.
        negation (bool, optional): If True, returns True if the value is within the range. 
                                   If False, returns True if the value is outside the range.
                                   Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is within the given range.
    """
    if negation:
        return min_val <= val <= max_val
    else:
        return not (min_val <= val <= max_val)

def is_in_list(val, string_list: List, negation: bool = True) -> bool:
    """
    Checks if a value is (or is not, based on the inverse_check flag) in a given list.

    Args:
        val: The value to be checked.
        string_list (List): The list to check against.
        negation (bool, optional): If True, returns True if the value is in the list.
                                        If False, returns True if the value is not in the list.
                                        Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is (or is not) in the list based on 'inverse_check'.
    """
    return (val in string_list) if negation else (val not in string_list)

class GraphGenerator:

    def __init__(self,db_manager=DBManager(), node_types=NodeTypes(), main_graph_dir=GRAPH_DIR, from_scratch=False, skip_main_init=True):
        """
        Initializes the GraphGenerator object.

        Args:
            db_manager (DBManager,optional): The database manager object. Defaults to DBManager().
            node_types (NodeTypes,optional): The node types object. Defaults to NodeTypes().
            main_graph_dir (str,optional): The directory where the main graph is stored. Defaults to MAIN_GRAPH_DIR.
            from_scratch (bool,optional): If True, deletes the graph database and recreates it from scratch.
            skip_main_init (bool,optional): If True, skips the initialization of the main nodes and relationships.

        """

        self.from_scratch = from_scratch
        self.node_types=node_types
        self.db_manager = db_manager

        self.main_graph_dir=os.path.join(main_graph_dir,'main')
        self.main_node_dir=os.path.join(self.main_graph_dir,'nodes')
        self.main_relationship_dir=os.path.join(self.main_graph_dir,'relationships')
        
        if from_scratch and os.path.exists(self.main_node_dir):
            LOGGER.info('Starting from scratch. Deleting main graph directory')
            shutil.rmtree(self.main_node_dir)
        os.makedirs(self.main_node_dir,exist_ok=True)
        os.makedirs(self.main_relationship_dir,exist_ok=True)

        # Initialize the main nodes
        if not skip_main_init:
            self.initialize_nodes(node_dir=self.main_node_dir)
            self.initialize_relationships(node_dir=self.main_node_dir,relationship_dir=self.main_relationship_dir)

    def get_node_id_maps(self,node_dir=None,graph_dir=None):
        """
        Get the node id maps for the graph database.

        Args:
            node_dir (str,optional): The directory where the node csv files are stored. Defaults to None.
            graph_dir (str,optional): The directory where the graph database is stored. Defaults to None.

        Returns:
            tuple: A tuple containing the node id maps.
        """
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

    def initialize_nodes(self,node_dir):
        """
        Initialize the nodes for the graph database.

        Args:
            node_dir (str): The directory where the node csv files are stored.

        """
        # Material
        if not os.path.exists(os.path.join(node_dir,'material.csv')):
            LOGGER.info("Creating material nodes")
            materials,materials_properties,material_id_map=self.node_types.get_material_nodes()
            create_nodes(node_names=materials,
                        node_type='Material',
                        node_prefix='material',
                        node_properties=materials_properties,
                        filepath=os.path.join(node_dir, 'material.csv'))
        # Element
        if not os.path.exists(os.path.join(node_dir,'element.csv')):
            LOGGER.info("Creating element nodes")
            elements,elements_properties,element_id_map=self.node_types.get_element_nodes()
            create_nodes(node_names=elements,
                        node_type='Element',
                        node_prefix='element',
                        node_properties=elements_properties,
                        filepath=os.path.join(node_dir, 'element.csv'))
        # Crystal System
        if not os.path.exists(os.path.join(node_dir,'crystal_system.csv')):
            LOGGER.info("Creating crystal system nodes")
            crystal_systems,crystal_systems_properties,crystal_system_id_map=self.node_types.get_crystal_system_nodes()
            create_nodes(node_names=crystal_systems, 
                        node_type='CrystalSystem', 
                        node_prefix='crystalSystem', 
                        # node_properties=crystal_systems_properties,
                        filepath=os.path.join(node_dir, 'crystal_system.csv'))
        # Chemenv
        if not os.path.exists(os.path.join(node_dir,'chemenv.csv')):
            LOGGER.info("Creating chemenv nodes")
            chemenv_names,chemenv_names_properties,chemenv_name_id_map=self.node_types.get_chemenv_nodes()
            create_nodes(node_names=chemenv_names, 
                        node_type='Chemenv', 
                        node_prefix='chemenv', 
                        # node_properties=chemenv_names_properties,
                        filepath=os.path.join(node_dir, 'chemenv.csv'))
        # Chemenv Element
        if not os.path.exists(os.path.join(node_dir,'chemenv_element.csv')):
            LOGGER.info("Creating chemenv element nodes")
            chemenv_element_names,chemenv_element_names_properties,chemenv_element_name_id_map=self.node_types.get_chemenv_element_nodes()
            create_nodes(node_names=chemenv_element_names, 
                        node_type='ChemenvElement', 
                        node_prefix='chemenvElement', 
                        filepath=os.path.join(node_dir, 'chemenv_element.csv'))
        # Magnetic State
        if not os.path.exists(os.path.join(node_dir,'magnetic_state.csv')):
            LOGGER.info("Creating magnetic state nodes")
            magnetic_states,magnetic_states_properties,magnetic_state_id_map=self.node_types.get_magnetic_states_nodes()
            create_nodes(node_names=magnetic_states, 
                        node_type='MagneticState', 
                        node_prefix='magState', 
                        filepath=os.path.join(node_dir, 'magnetic_state.csv'))
        # Space Groups
        if not os.path.exists(os.path.join(node_dir,'spg.csv')):
            LOGGER.info("Creating space group nodes")
            space_groups,space_groups_properties,space_groups_id_map=self.node_types.get_space_group_nodes()
            create_nodes(node_names=space_groups, 
                        node_type='SpaceGroup', 
                        node_prefix='spg', 
                        filepath=os.path.join(node_dir, 'spg.csv'))
        # Oxidation States
        if not os.path.exists(os.path.join(node_dir,'oxidation_state.csv')):
            LOGGER.info("Creating oxidation state nodes")
            oxidation_states,oxidation_states_names,oxidation_state_id_map=self.node_types.get_oxidation_states_nodes()
            create_nodes(node_names=oxidation_states, 
                        node_type='OxidationState', 
                        node_prefix='oxiState', 
                        filepath=os.path.join(node_dir, 'oxidation_state.csv'))
        
        # SPG_WYCKOFFS
        if not os.path.exists(os.path.join(node_dir,'spg_wyckoff.csv')):
            LOGGER.info("Creating space group wyckoff nodes")
            spg_wyckoffs,spg_wyckoff_properties,spg_wyckoff_id_map=self.node_types.get_wyckoff_positions_nodes()
            create_nodes(node_names=spg_wyckoffs,
                        node_type='SPGWyckoff',
                        node_prefix='spgWyckoff',
                        # node_properties=spg_wyckoff_properties,
                        filepath=os.path.join(node_dir, 'spg_wyckoff.csv'))
        return None

    def initialize_relationships(self,node_dir,relationship_dir):
        """
        Initialize the relationships for the graph database.

        Args:
            node_dir (str): The directory where the node csv files are stored.
            relationship_dir (str): The directory where the relationship csv files are stored.

        """
        # Element - Element Connections
        LOGGER.info("Attemping to create element-element geometric-electric relationship")
        create_relationships(node_a_csv=os.path.join(node_dir,'element.csv'),
                            node_b_csv=os.path.join(node_dir,'element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric_electric'},
                            connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir,
                            )
        
        LOGGER.info("Attemping to create element-element geometric relationship")
        create_relationships(node_a_csv=os.path.join(node_dir,'element.csv'),
                            node_b_csv=os.path.join(node_dir,'element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric'},
                            connection_name='GEOMETRIC_CONNECTS',
                            relationship_dir=relationship_dir)

        LOGGER.info("Attempting to create element-element electric relationship")
        create_relationships(node_a_csv=os.path.join(node_dir,'element.csv'),
                            node_b_csv=os.path.join(node_dir,'element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'), 
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'electric'}, 
                            connection_name='ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir)
        
        # Chemenv - Chemenv Connections
        LOGGER.info("Attempting to create Chemenv - Chemenv Geometric-Electric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv.csv'), 
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric_electric'},
                            connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir)

        LOGGER.info("Attempting to create Chemenv - Chemenv Geometric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric'}, 
                            connection_name='GEOMETRIC_CONNECTS',
                            relationship_dir=relationship_dir)

        LOGGER.info("Attempting to create Chemenv - Chemenv Electric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'electric'},
                            connection_name='ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir)
            
        # ChemenvElement - ChemenvElement Connections
        LOGGER.info("Attempting to create ChemenvElement - ChemenvElement Geometric-Electric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric_electric'},
                            connection_name='GEOMETRIC_ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir)
        
        LOGGER.info("Attempting to create ChemenvElement - ChemenvElement Geometric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'geometric'},
                            connection_name='GEOMETRIC_CONNECTS',
                            relationship_dir=relationship_dir)
    
        LOGGER.info("Attempting to create ChemenvElement - ChemenvElement Electric Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_bonding_task,
                            mp_task_params={'bonding_method':'electric'},
                            connection_name='ELECTRIC_CONNECTS',
                            relationship_dir=relationship_dir)

        # Chemenv - Element Connections
        LOGGER.info("Attempting to create Chemenv - Element Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'chemenv.csv'),
                            node_b_csv=os.path.join(node_dir,'element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'), 
                            mp_task=create_chemenv_element_task,
                            connection_name='CAN_OCCUR',
                            relationship_dir=relationship_dir)
        
        # Material - Element Connections
        LOGGER.info("Attempting to create Material - Element Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'material.csv'),
                            node_b_csv=os.path.join(node_dir,'element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_material_element_task,
                            connection_name='HAS',
                            relationship_dir=relationship_dir)
        
        # Material - Chemenv Connections
        LOGGER.info("Attempting to create Material - Chemenv Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'material.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_material_chemenv_task,
                            connection_name='HAS',
                            relationship_dir=relationship_dir)
    
        # Material - ChemenvElement Connections
        LOGGER.info("Attempting to create Material - ChemenvElement Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'material.csv'),
                            node_b_csv=os.path.join(node_dir,'chemenv_element.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_material_chemenvElement_task,
                            connection_name='HAS',
                            relationship_dir=relationship_dir)
            
        # Material - spg Connections
        LOGGER.info("Attempting to create Material - spg Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'material.csv'),
                            node_b_csv=os.path.join(node_dir,'spg.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_material_spg_task,
                            connection_name='HAS',
                            relationship_dir=relationship_dir)

        # Material - crystal_system Connections
        LOGGER.info("Attempting to create Material - crystal_system Connections")
        create_relationships(node_a_csv=os.path.join(node_dir,'material.csv'),
                            node_b_csv=os.path.join(node_dir,'crystal_system.csv'),
                            material_csv=os.path.join(node_dir,'material.csv'),
                            mp_task=create_material_crystal_system_task,
                            connection_name='HAS',
                            relationship_dir=relationship_dir)

    def list_sub_graphs(self,graph_dir=None):
        """
        List the subgraphs in the graph directory.

        Args:
            graph_dir (str): The directory of the graph.

        Returns:
            list: A list of the subgraphs in the graph directory.
        """
        if graph_dir is None:
            print("No graph directory provided. Using main graph directory.")
            graph_dir=self.main_graph_dir

        sub_graph_dir=os.path.join(graph_dir,'sub_graphs')
        if not os.path.exists(sub_graph_dir):
            raise Exception("No subgraphs found in graph directory.")

        graph_dirs = glob(os.path.join(sub_graph_dir, '*'))
        return [os.path.basename(d) for d in graph_dirs]
    
    def list_graph_nodes(self,graph_dir):
        """
        List the graph nodes in the graph directory.

        Args:
            graph_dirname (str): The name of the graph database directory.

        Returns:
            list: A list of the graph nodes in the graph directory.
        """
        main_graph_dir=os.path.join(graph_dir,'neo4j_csv')
        main_node_dir=os.path.join(main_graph_dir,'nodes')

        if not os.path.exists(main_node_dir):
            raise Exception("No nodes found in graph directory.")
        
        node_files = glob(os.path.join(main_node_dir, '*.csv'))
        return node_files
    
    def list_graph_relationships(self,graph_dir):
        """
        List the graph relationships in the graph directory.

        Args:
            graph_dirname (str): The name of the graph database directory.

        Returns:
            list: A list of the graph relationships in the graph directory.
        """
        main_graph_dir=os.path.join(graph_dir,'neo4j_csv')
        main_relationship_dir=os.path.join(main_graph_dir,'relationships')

        if not os.path.exists(main_relationship_dir):
            raise Exception("No relationships found in graph directory.")

        relationship_files = glob(os.path.join(main_relationship_dir, '*.csv'))
        return relationship_files
    
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
        LOGGER.info(f"Number of materials after filtering: {len(filtered_df)}")
        return filtered_df

    def screen_graph(self,graph_dir, sub_graph_name, from_scratch=False,**kwargs):
        """
        Screen the graph database for materials.

        Args:
            graph_dir (str): The directory of the graph database.
            sub_graph_name (str): The name of the sub graph name.
            from_scratch (bool, optional): If True, deletes the graph database and recreates it from scratch. Defaults to False.

        Returns:
            None
        """
        root_graph_dir=os.path.dirname(graph_dir)

        # Define main graph directory paths
        main_graph_dir=graph_dir
        main_node_dir=os.path.join(main_graph_dir,'nodes')
        main_relationship_dir=os.path.join(main_graph_dir,'relationships')

        # Define subgraph directory paths
        sub_graphs_dir=os.path.join(main_graph_dir,'sub_graphs')
        sub_graph_dir=os.path.join(sub_graphs_dir,sub_graph_name)
        if from_scratch and os.path.exists(sub_graph_dir):
            LOGGER.info(f'Starting from scratch. Deleting graph directory {sub_graph_dir}')
            shutil.rmtree(sub_graph_dir)

        node_dir=os.path.join(sub_graph_dir,'nodes')
        relationship_dir=os.path.join(sub_graph_dir,'relationships')
        os.makedirs(node_dir,exist_ok=True)
        os.makedirs(relationship_dir,exist_ok=True)

        LOGGER.info('Screening the graph database')
        # Copy all the nodes besides the material node to the graph directory
        node_files=glob(main_node_dir+ os.sep +'*.csv')
        for file_paths in node_files:
            filename=os.path.basename(file_paths)
            if filename == "material.csv":
                continue
            LOGGER.info(f"Copying {filename} to {node_dir}")
            shutil.copy(os.path.join(main_node_dir,filename),os.path.join(node_dir,filename))

        LOGGER.info(f"Creating new material.csv")
        original_materials_file=os.path.join(main_node_dir,'material.csv')
        new_materials_file=os.path.join(node_dir,'material.csv')
        materials_df=self.screen_material_nodes(material_csv=original_materials_file,**kwargs)
        materials_df.to_csv(new_materials_file,index=True)

        self.initialize_relationships(node_dir=node_dir,relationship_dir=relationship_dir)

    def create_sub_graph(self,graph_dir, sub_graph_name, node_files, relationship_files, from_scratch=False):
        """
        Create subgraphs from the graph.

        Args:
            graph_dir (str): The directory of the graph.
            sub_graph_name (str): The name of the subgraph directory.
            node_files (list): A list of node csv files to be included in the subgraph. Use list_graph_nodes(graph_dirname) to get the list of node files.
            relationship_files (list): A list of relationship csv files to be included in the subgraph. Use list_graph_relationships(graph_dirname) to get the list of relationship files.
            from_scratch (bool, optional): If True, deletes the graph database and recreates it from scratch. Defaults to False.

        Returns:
            None
        """

        # Define main graph directory
        main_graph_dir=os.path.join(graph_dir,'neo4j_csv')
        main_node_dir=os.path.join(main_graph_dir,'nodes')
        main_relationship_dir=os.path.join(main_graph_dir,'relationships')

        # Define subgraph directory
        sub_graphs_dir=os.path.join(graph_dir,'sub_graphs')
        sub_graph_dir=os.path.join(sub_graphs_dir,sub_graph_name)
        if from_scratch and os.path.exists(sub_graph_dir):
            LOGGER.info(f'Starting from scratch. Deleting graph directory {sub_graph_dir}')
            shutil.rmtree(sub_graph_dir)
        node_dir=os.path.join(sub_graph_dir,'neo4j_csv','nodes')
        relationship_dir=os.path.join(sub_graph_dir,'neo4j_csv','relationships')
        os.makedirs(node_dir,exist_ok=True)
        os.makedirs(relationship_dir,exist_ok=True)

        # Copy node files from the main graph to the subgraph
        for file_paths in node_files:
            filename=os.path.basename(file_paths)
            LOGGER.info(f"Copying {filename} to {node_dir}")
            shutil.copy(os.path.join(main_node_dir,filename),os.path.join(node_dir,filename))

        # Copy relationship files from the main graph to the subgraph
        for file_paths in relationship_files:
            filename=os.path.basename(file_paths)
            LOGGER.info(f"Copying {filename} to {relationship_dir}")
            shutil.copy(os.path.join(main_relationship_dir,filename),os.path.join(relationship_dir,filename))

        return None

    def write_graphml(self,graph_dir,from_scratch=False):
        """
        Write a graphml file from the graph.

        Args:
            graph_dir (str): The directory of the graph.
            filepath (str,optional): The path to the file where the graphml file will be saved. Defaults to None.

        Returns:
            None
        """
        graph_dirname=os.path.basename(graph_dir)
        neo4j_graph_dir=os.path.join(graph_dir,'neo4j_csv')
        filepath=os.path.join(graph_dir,f'{graph_dirname}.graphml')
        if not os.path.exists(graph_dir):
            raise Exception("Graph directory does not exist.")
        if os.listdir(os.path.join(neo4j_graph_dir,'nodes'))==0:
            raise Exception("No nodes found in graph directory.")

        if from_scratch and os.path.exists(filepath):
            LOGGER.info(f'Starting from scratch. Deleting graph directory {filepath}')
            os.remove(filepath)
        node_files=self.list_graph_nodes(graph_dirname=graph_dirname)
        relationship_files=self.list_graph_relationships(graph_dirname=graph_dirname)


        generator=NetworkXGraphGenerator(graph_dir=graph_dir)
        generator.parse_node_files(node_files=node_files)
        generator.parse_relationship_files(relationship_files=relationship_files)
        generator.export_graph(filepath=filepath,method='graphml')
        return None

class NetworkXGraphGenerator:
    def __init__(self,graph_dir=None,from_scratch=False):
        """
        Initializes the NetworkXGraphGenerator object.

        Args:
            graph_dir (str,optional): The directory where the graph database is stored. Defaults to None.
            from_scratch (bool,optional): If True, deletes the graph database and recreates it from scratch. Defaults to False.

        """
        self.graph_dir=graph_dir
        self.graph_name=self.graph_dir.split(os.sep)[-1]
        self.graphml_file=os.path.join(self.graph_dir,f'{self.graph_name}.graphml')
        self.node_files=glob(os.path.join(self.graph_dir,'neo4j_csv','nodes','*.csv'))
        self.relationship_files=glob(os.path.join(self.graph_dir,'neo4j_csv','relationships','*.csv'))
        self.from_scratch=from_scratch
        self.graph=nx.Graph()
        self.neo4j_node_id_maps={}
        self.parse_node_files()

    def parse_node_file(self,file):
        """
        Parses a node csv file and returns a dictionary containing the node data.

        Args:
            file (str): The path to the node csv file.

        Returns:
            dict: A dictionary containing the node data.
        """
        df=pd.read_csv(file)
        column_names=list(df.columns)
        node_id_name=column_names[0].strip(')').split('(')[-1]
        self.neo4j_node_id_maps[node_id_name]={}
        node_list = df.to_dict(orient='records')
        return node_list
    
    def parse_node_files(self, node_files=None):
        """
        Parses all node csv files in the graph directory and returns a dictionary containing the node data.

        Returns:
            dict: A dictionary containing the node data.
        """
        files=None
        if node_files:
            files=node_files
        else:
            files=self.node_files
            
        node_dict={}
        tmp_node_list=[]
        node_list=[]
        for file in files:
            tmp_node_list.extend(self.parse_node_file(file))
        for i,node_dict in enumerate(tmp_node_list):
            neo4j_node_id_name, neo4j_node_id=list(node_dict.items())[0]
            neo4j_node_id_name=neo4j_node_id_name.strip(')').split('(')[-1]
            self.neo4j_node_id_maps[neo4j_node_id_name][neo4j_node_id]=i

            node_list.append((i,node_dict))
        self.graph.add_nodes_from(node_list)
        return node_list
    
    def parse_relationship_file(self,file,return_list=False):
        """
        Parses a relationship csv file and returns a dictionary containing the relationship data.

        Args:
            file (str): The path to the relationship csv file.
            return_list (bool,optional): If True, returns a list of tuples containing the relationship data. Defaults to False.

        Returns:
            dict: A dictionary containing the relationship data.
        """

        df=pd.read_csv(file)
        column_headers=list(df.columns)
        start_id_name=column_headers[0].strip(')').split('(')[-1]
        end_id_name=column_headers[1].strip(')').split('(')[-1]
        relationship_list=[]


        # Extract node identifiers
        start_id_name = column_headers[0].strip(')').split('(')[-1]
        end_id_name = column_headers[1].strip(')').split('(')[-1]

        # Prepare the node ID columns using precomputed maps
        df['start_id'] = df[column_headers[0]].map(self.neo4j_node_id_maps[start_id_name])
        df['end_id'] = df[column_headers[1]].map(self.neo4j_node_id_maps[end_id_name])

        # Collect other properties
        property_columns = column_headers[2:]  # assuming the first two are node ID columns
        
        if return_list:
            relationship_list = []
            for index, row in df.iterrows():
                relationship_list.append((row['start_id'], row['end_id'], row[property_columns].to_dict()))
            return relationship_list
        else:
            for index, row in df.iterrows():
                self.graph.add_edge(row['start_id'], row['end_id'], **row[property_columns].to_dict())
            return None

    def parse_relationship_files(self,relationship_files=None):
        """
        Parses all relationship csv files in the graph directory and returns a dictionary containing the relationship data.

        Returns:
            dict: A dictionary containing the relationship data.
        """
        if self.neo4j_node_id_maps == {}:
            raise Exception("Node ID maps do not exist. Please call parse_node_files() first.")

        files=None
        if relationship_files:
            files=relationship_files
        else:
            files=self.relationship_files
        for file in files:
            self.parse_relationship_file(file)
        return None

    def export_graph(self,filepath,method='graphml'):
        """
        Exports the graph to a file in the specified format.

        Args:
            filepath (str,optional): The path to the file where the graph will be exported. Defaults to None.
            method (str,optional): The method to use for exporting the graph. Defaults to 'graphml'.

        Returns:
            None
        """
        if method=='graphml':
            nx.write_graphml(self.graph,filepath)
        return None

# if __name__=='__main__':
    # graph=GraphGenerator()
    # graph.write_graphml(graph_dirname='nelements-2-2')

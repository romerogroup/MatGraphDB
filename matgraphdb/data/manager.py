from enum import Enum
import os
import json
from glob import glob
from typing import Dict, List, Tuple, Union
from multiprocessing import Pool
from functools import partial


import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pyarrow as pa
import pyarrow.parquet as pq

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES, GLOBAL_PROP_FILE,ENCODING_DIR, EXTERNAL_DATA_DIR, MP_DIR

from matgraphdb.calculations.mat_calcs.bonding_calc import calculate_cutoff_bonds,calculate_electric_consistent_bonds,calculate_geometric_consistent_bonds,calculate_geometric_electric_consistent_bonds
from matgraphdb.calculations.mat_calcs.bond_stats_calc import calculate_bond_orders_sum,calculate_bond_orders_sum_squared_differences
from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections
from matgraphdb.calculations.mat_calcs.embeddings import generate_composition_embeddings, generate_matminer_embeddings
from matgraphdb.calculations.mat_calcs.wyckoff_calc import calculate_wyckoff_positions
from matgraphdb.calculations.parsers import parse_chargemol_bond_orders,parse_chargemol_net_atomic_charges, parse_chargemol_atomic_moments, parse_chargemol_overlap_populations
from matgraphdb.data.utils import MATERIAL_PARQUET_SCHEMA
from matgraphdb.utils.periodic_table import atomic_symbols
from matgraphdb.utils import get_logger

logger=get_logger(__name__, console_out=False, log_level='info')

class DBManager:
    def __init__(self, directory_path=DB_DIR, calc_path=DB_CALC_DIR, n_cores=N_CORES):
        """
        Initializes the Manager object.

        Args:
            directory_path (str): The path to the directory where the database is stored.
            calc_path (str): The path to the directory where calculations are stored.
            n_cores (int): The number of CPU cores to be used for parallel processing.

        """
        self.directory_path = directory_path
        self.calculation_path = calc_path

        self.n_cores = N_CORES

    @property
    def properties(self):
        """
        Returns a list of properties available in the database.
        Returns:
            list: A list of property names.

        """
        file=self.database_files[-1].split(os.sep)[-1]
        data=self.load_json(file)
        return list(data.keys())
    
    def database_files(self):
        """
        Returns a list of JSON file paths in the specified directory.

        Returns:
            list: A list of JSON file paths.
        """
        return glob(self.directory_path + os.sep + '*.json')
    
    def calculation_dirs(self):
        return glob(self.calculation_path + os.sep + 'mp-*')

    def index_files(self, files):
        """
        Indexes a list of files.
        Args:
            files (list): A list of file paths.
        Returns:
            dict: A dictionary where the keys are the filenames and the values are the indexed filenames.

        """
        return {file.split(os.sep)[-1]:f"m-{i}" for i,file in enumerate(files)}

    def process_task(self, func, list, **kwargs):
        logger.info(f"Process full database using {self.n_cores} cores")
        logger.info(f"Using {self.n_cores} cores")
        with Pool(self.n_cores) as p:
            results=p.map(partial(func,**kwargs), list)
        return results

    def load_json(self, filename):
        """Load a JSON file given its filename."""
        try:
            with open(os.path.join(self.directory_path, filename), 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def load_data(self):
        files=self.database_files()
        results=self.process_task(MPTasks.load_data_tasks,files)
        return results
        
    def create_material(self,
                        # Do type hinting composition can either be a string or a dictionary
                        composition:Union[str,dict,Composition] =None,
                        structure:Structure=None,
                        coords:Union[List[Tuple[float,float,float]],np.ndarray]=None,
                        coords_are_cartesian :bool=False,
                        species:List[str]=None,
                        lattice:Union[List[Tuple[float,float,float]],np.ndarray]=None,
                        properties:dict=None):

        """
        Create a material entry in the database.

        Args:
            composition (Union[str, dict, Composition], optional): The composition of the material. It can be provided as a string, a dictionary, or an instance of the Composition class. Defaults to None.
            structure (Structure, optional): The structure of the material. Defaults to None.
            coords (Union[List[Tuple[float,float,float]], np.ndarray], optional): The atomic coordinates of the material. Defaults to None.
            coords_are_cartesian (bool, optional): Whether the atomic coordinates are in Cartesian coordinates. Defaults to False.
            species (List[str], optional): The atomic species of the material. Required if coords is provided. Defaults to None.
            lattice (Union[List[Tuple[float,float,float]], np.ndarray], optional): The lattice parameters of the material. Required if coords is provided. Defaults to None.
            properties (dict, optional): Additional properties of the material. Defaults to None.

        Returns:
            str: The path to the JSON file containing the material data.
        """

        if isinstance(composition, Composition):
            composition = composition
        if isinstance(composition, str):
            composition = Composition(composition)
        elif isinstance(composition, dict):
            composition = Composition.from_dict(composition)
        else:
            composition=None

        if coords:
            if not species:
                raise ValueError("If coords is used, species must be provided")
            if not lattice:
                raise ValueError("If coords is used, lattice must be provided")

        if species:
            if not coords:
                raise ValueError("If species is provided, coords must be provided")
            if not lattice:
                raise ValueError("If species is provided, lattice must be provided")

        if lattice:
            if not species:
                raise ValueError("If lattice is provided, species must be provided")
            if not coords:
                raise ValueError("If lattice is provided, coords must be provided")

        if isinstance(structure, Structure):
            structure = structure
        elif coords:
            if coords_are_cartesian:
                structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            else:
                structure = Structure(lattice, species, coords)
        else:
            structure=None

        if structure is None and composition is None:
            raise ValueError("Either a structure or a composition must be provided")

        if structure:
            composition=structure.composition

        n_current_files=len(self.database_files)
        filename=f"m-{n_current_files}"
        json_file=os.path.join(self.directory_path,f"{filename}.json")

        data={property_name:None for property_name in self.properties}
        data["elements"]=list(composition.as_dict().keys())
        data["nelements"]=len(composition.as_dict())
        data["composition"]=composition.as_dict()
        data["composition_reduced"]=dict(composition.to_reduced_dict)
        data["formula_pretty"]=composition.to_pretty_string()

  

        if structure:
            data["volume"]=structure.volume
            data["density"]=structure.density
            data["nsites"]=len(structure.sites)
            data["density_atomic"]=data["nsites"]/data["volume"]
            data["structure"]=structure.as_dict()

            symprec=0.01
            sym_analyzer=SpacegroupAnalyzer(structure,symprec=symprec)
            data["symmetry"]={}
            data["symmetry"]["crystal_system"]=sym_analyzer.get_crystal_system()
            data["symmetry"]["number"]=sym_analyzer.get_space_group_number()
            data["symmetry"]["point_group"]=sym_analyzer.get_point_group_symbol()
            data["symmetry"]["symbol"]=sym_analyzer.get_hall()
            data["symmetry"]["symprec"]=symprec

            sym_dataset=sym_analyzer.get_symmetry_dataset()

            data["wyckoffs"]=sym_dataset['wyckoffs']
            data["symmetry"]["version"]="1.16.2"
  

            try:
                data["bonding_cutoff_connections"]=calculate_cutoff_bonds(structure)
            except:
                pass
  
            try:
                coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(structure)
                data["coordination_environments_multi_weight"]=coordination_environments
                data["coordination_multi_connections"]=nearest_neighbors
                data["coordination_multi_numbers"]=coordination_numbers
            except:
                pass

        with open(json_file, 'w') as f:
            json.dump(data,f, indent=4)

        return json_file
    
    def check_property(self, property_name):
        """Check if a given property exists in all JSON files and categorize them."""
        
        database_files = self.database_files()
        logger.info("Processing files from : ",self.directory_path + os.sep + '*.json')
        results=self.process_task(MPTasks.check_property_task, database_files, property_name=property_name)

        success = []
        failed = []
        for file, result in zip(database_files, results):
            if result == True:
                success.append(file)
            else:
                failed.append(file)

        return success, failed
 
    def check_chargemol(self):
        """Check if a given property exists in all JSON files and categorize them."""
        
        
        calc_dirs = self.calculation_dirs()
        print("Processing files from : ",self.calculation_path + os.sep + 'mp-*')
        results=self.process_task(MPTasks.check_chargemol_task, calc_dirs)

        success = []
        failed = []
        for path, result in zip(calc_dirs,results):
            chargemol_dir=os.path.join(path,'chargemol')
            if result==True:
                success.append(chargemol_dir)
            else:
                failed.append(chargemol_dir)

        return success, failed
   
    def collect_chargemol_info(self):
        
        calc_dirs = self.calculation_dirs()
        self.process_task(MPTasks.collect_chargemol_info_task, calc_dirs, directory_path=self.calculation_path)
        logger.info(f"Finished collection Chargemol information")
        return None
    
    def chemenv_calc(self,from_scratch=False):
        """
        Perform Chemenv Calculation using Multi Weight Strategy.

        This function runs the Chemenv Calculation using the Multi Weight Strategy.
        It prints a header for the process and then processes the database with the defined function.

        Parameters:
        None

        Returns:
        None
        """
        logger.info(f"Starting collection ChemEnv Calculations")
        # Process the database with the defined function
        files=self.database_files()
        self.process_task(MPTasks.chemenv_task,files,from_scratch=from_scratch)
        logger.info(f"Finished collection ChemEnv Calculations")

        return None

    def bonding_calc(self):
        
        logger.info(f"Starting collection Bonding Calculations")
        # Process the database with the defined function
        files=self.database_files()
        self.process_task(MPTasks.bonding_task,files)
        logger.info(f"Finished collection Bonding Calculations")

        return None
    
    def bond_orders_stats_calculation(self):
        """
        Perform Bond Orders Statistics Calculation.

        This function runs the Bond Orders Statistics Calculation.
        It prints a header for the process and then processes the database with the defined function.

        Returns:
        None
        """
       

        ELEMENTS = atomic_symbols[1:]
        n_elements = len(ELEMENTS)
        n_bond_orders = np.zeros(shape=(n_elements, n_elements))
        bond_orders_avg = np.zeros(shape=(n_elements, n_elements))
        bond_orders_std = np.zeros(shape=(n_elements, n_elements))


        logger.info(f"Starting collection Bond Orders Sum Calculations")
        # Process the database with the defined function
        files=self.database_files()
        results=self.process_task(MPTasks.bond_orders_sum_task,files)
        logger.info(f"Finished collection Bond Orders Sum Calculations")
        # Initialize arrays for bond order calculations
        
        logger.info("Starting calculation of bond order average")
        # Calculate the average of the bond orders in the database
        for result in results:
            bond_orders_avg += result[0] 
            n_bond_orders += result[1]
        bond_orders_avg = np.divide(bond_orders_avg, n_bond_orders, out=np.zeros_like(bond_orders_avg), where=n_bond_orders != 0)
        with open(GLOBAL_PROP_FILE) as f:
            global_data = json.load(f)
        global_data['bond_orders_avg'] = bond_orders_avg.tolist()
        global_data['n_bond_orders'] = n_bond_orders.tolist()
        with open(GLOBAL_PROP_FILE,'w') as f:
            json.dump(global_data, f)
            
        logger.info("Finished calculation of bond order average")

        logger.info("Starting calculation of bond order standard deviation")
        # Calculate the standard deviation of the bond orders in the database
        results=self.process_task(MPTasks.bond_orders_sum_squared_differences_task,files)

        for result in results:
            #This results in the material database sum of the sum of squared differences for materials
            bond_orders_std += result[0] 
        bond_orders_std = np.divide(bond_orders_std, n_bond_orders, out=np.zeros_like(bond_orders_std), where=n_bond_orders != 0)
        bond_orders_std = bond_orders_std ** 0.5

        with open(GLOBAL_PROP_FILE) as f:
            global_data = json.load(f)
            global_data['bond_orders_std'] = bond_orders_std.tolist()
        logger.info("Finished calculation of bond order standard deviation")

        logger.info("Saving bond order average and standard deviation to global property file")
        with open(GLOBAL_PROP_FILE, 'w') as f:
            json.dump(global_data, f)

        logger.info("Finished calculation of bond order statistics")
        return None

    def generate_composition_embeddings(self):

        material_ids=[]
        files=self.database_files()

        results=self.process_task(MPTasks.get_structure_task,files)
        material_ids=[]
        structures=[]
        compositions=[]
        for result in results:
            structure, material_id, nsites = result
            if structure is not None:
                structures.append(structure)
                compositions.append(structure.composition)
                material_ids.append(material_id)


        features=generate_composition_embeddings(compositions,material_ids)
        for index, row in features.iterrows():
            encoding_file=os.path.join(ENCODING_DIR,index+'.json')

            embedding_dict={'element_fraction':row.values.tolist()}
            if os.path.exists(encoding_file):
                
                with open(encoding_file) as f:
                    data = json.load(f)
                data.update(embedding_dict)
                with open(encoding_file,'w') as f:
                    json.dump(data, f, indent=None)

            else:
                with open(encoding_file,'w') as f:
                    json.dump(embedding_dict, f, indent=None)
        return None

    def generate_matminer_embeddings(self,feature_set=['element_property']):
        
        
        files=self.database_files()

        print("Featchering structures")
        results=self.process_task(MPTasks.get_structure_task,files)
        material_ids=[]
        structures=[]
        for result in results:
            structure, material_id, nsites = result
            if structure is not None:
                structures.append(structure)
                material_ids.append(material_id)
        print("Generating embeddings")
        feature_df=generate_matminer_embeddings(structures,material_ids,features=feature_set)
        feature_names=feature_df.columns.tolist()


        json_embedding_tuples=[]
        for index, row in feature_df.iterrows():
            json_file=os.path.join(self.directory_path,index+'.json')
            feature_set_key='-'.join(feature_set)

            embedding_dict={'feature_vectors':{
                                    feature_set_key:{
                                        'values':row.values.tolist(),
                                        'feature_names':feature_names
                                    }
                                }
                            }
            json_embedding_tuples.append((json_file,embedding_dict))
        print("Storing embeddings")
        self.process_task(MPTasks.generate_matminer_embeddings_task,json_embedding_tuples)
        
        

        return None

    def generate_wyckoff_positions(self):
        """
        Generates the Wyckoff positions for all materials in the database.

        Returns:
            None
        """
        
                
        logger.info("Starting collection Wyckoff Positions")

        files=self.database_files()
        results=self.process_task(MPTasks.wyckoff_calc_task,files)
        logger.info("Finished processing database")
        return None

    def merge_oxidation_states_doc(self):
        """
        Merges oxidation_states doc from materials project into the database.

        This function merges external tasks into the database. It iterates over the database files and checks if the
        corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
        been completed, the task is merged into the database.

        Returns:
            None
        """
        
        
        logger.info("Starting merging external database")
        json_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','oxidation_states')
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"Directory {json_dir} does not exist")
        
        json_files=glob(os.path.join(json_dir,'*.json'))

        results=self.process_task(MPTasks.merge_oxidation_states_doc_task, json_files, directory_path=self.directory_path)

        logger.info("Finished merging external database")
        return None
    
    def merge_elasticity_doc(self):
        """
        Merges elasticity doc from materials project into the database.

        This function merges external tasks into the database. It iterates over the database files and checks if the
        corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
        been completed, the task is merged into the database.

        Returns:
            None
        """
        
    
        logger.info("Starting merging external database")
        json_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','elasticity')
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"Directory {json_dir} does not exist")
        
        json_files=glob(os.path.join(json_dir,'*.json'))

        results=self.process_task(MPTasks.merge_elasticity_doc_task, json_files, directory_path=self.directory_path)

        logger.info("Finished merging external database")
        return None
    
    def merge_summary_doc(self):
        """
        Merges json_database doc from materials project into the database.

        This function merges external tasks into the database. It iterates over the database files and checks if the
        corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
        been completed, the task is merged into the database.

        Returns:
            None
        """
        
        logger.info("Starting merging external database")
        json_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"Directory {json_dir} does not exist")
        
        json_files=glob(os.path.join(json_dir,'*.json'))

        results=self.process_task(MPTasks.merge_summary_doc_task, json_files, directory_path=self.directory_path)

        logger.info("Finished merging external database")
        return None

    def create_parquet_file(self):
        """
        Generates the a parquet file for all materials in the database.
        Returns:
            None
        """

        # Process the database with the defined function
        files=self.database_files()
        results=self.process_task(MPTasks.create_parquet_file_task,files)
        parquet_table=None
        main_data={}
        # Get all possible keys from the files
        for i,result in enumerate(results):
            data=result
            if data is not None:
                for key,value in data.items():
                    if key not in main_data.keys():
                        main_data[key]=[]
        # Putting all the data in the main data
        for i,result in enumerate(results):
            data=result
            if data is not None:
                # populate the main data if it has a value
                for key,value in data.items():
                    main_data[key].extend(value)

                # If the key does not exist in the main data, add it with None value
                for key,value in main_data.items():
                    if key not in data.keys():
                        main_data[key].extend([None])

        parquet_table=pa.Table.from_pydict(main_data,schema=MATERIAL_PARQUET_SCHEMA)

        pq.write_table(parquet_table, os.path.join(MP_DIR, 'materials_database.parquet'))
        logger.info("Database saved to parquet file")

        metadata = pq.read_metadata( os.path.join(MP_DIR, 'materials_database.parquet'))
        all_columns = []

        logger.info("Column names")
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue
  
            logger.info(filed_schema.name)
            all_columns.append(filed_schema.name)
        return None

class MPTasks:
    
    def bonding_task(json_file):
            # Load data from JSON file
            mpid=json_file.split('/')[-1].split('.')[0]
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    structure=Structure.from_dict(data['structure'])
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")
                return None
            
    def merge_elasticity_doc_task(json_file, directory_path=''):
            """
            Merges external elacticity doc from materials project into the database.

            This function merges external tasks into the database. It iterates over the database files and checks if the
            corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
            been completed, the task is merged into the database.

            Returns:
                None
            """
            with open(json_file) as f:
                new_data = json.load(f)
            try:
                mpid = new_data['material_id']

                main_json_file = os.path.join(directory_path,f'{mpid}.json')
                # Check if main json file exists
                if os.path.exists(main_json_file):
                    with open(main_json_file) as f:
                        main_data = json.load(f)
                else:
                    main_data={}
                main_data['elasticity']={}
                main_data['elasticity']['warnings']=new_data['warnings']
                main_data['elasticity']['order']=new_data['order']
                main_data['elasticity']['k_vrh']=new_data['bulk_modulus']['vrh']
                main_data['elasticity']['k_reuss']=new_data['bulk_modulus']['reuss']
                main_data['elasticity']['k_voigt']=new_data['bulk_modulus']['voigt']
                main_data['elasticity']['g_vrh']=new_data['shear_modulus']['vrh']
                main_data['elasticity']['g_reuss']=new_data['shear_modulus']['reuss']
                main_data['elasticity']['g_voigt']=new_data['shear_modulus']['voigt']
                main_data['elasticity']['sound_velocity_transverse']=new_data['sound_velocity']['transverse']
                main_data['elasticity']['sound_velocity_longitudinal']=new_data['sound_velocity']['longitudinal']
                main_data['elasticity']['sound_velocity_total']=new_data['sound_velocity']['snyder_total']
                main_data['elasticity']['sound_velocity_acoustic']=new_data['sound_velocity']['snyder_acoustic']
                main_data['elasticity']['sound_velocity_optical']=new_data['sound_velocity']['snyder_optical']
                main_data['elasticity']['thermal_conductivity_clarke']=new_data['thermal_conductivity']['clarke']
                main_data['elasticity']['thermal_conductivity_cahill']=new_data['thermal_conductivity']['cahill']
                main_data['elasticity']['young_modulus']=new_data['young_modulus']
                main_data['elasticity']['universal_anisotropy']=new_data['universal_anisotropy']
                main_data['elasticity']['homogeneous_poisson']=new_data['homogeneous_poisson']
                main_data['elasticity']['debye_temperature']=new_data['debye_temperature']
                main_data['elasticity']['state']=new_data['state']
                
                with open(main_json_file,'w') as f:
                    json.dump(main_data, f, indent=4)

                logger.info("Finished processing database")
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")

            return None
    
    def merge_oxidation_states_doc_task(json_file, directory_path=''):
            """
            Merges external oxidation_states doc from materials project into the database.

            This function merges external tasks into the database. It iterates over the database files and checks if the
            corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
            been completed, the task is merged into the database.

            Returns:
                None
            """
            with open(json_file) as f:
                new_data = json.load(f)
            try:
                mpid = new_data['material_id']

                main_json_file = os.path.join(directory_path,f'{mpid}.json')
                # Check if main json file exists
                if os.path.exists(main_json_file):
                    with open(main_json_file) as f:
                        main_data = json.load(f)
                else:
                    main_data={}
                main_data['oxidation_states']={}
                main_data['oxidation_states']["possible_species"]=new_data["possible_species"]
                main_data['oxidation_states']['possible_valences']=new_data['possible_valences']
                main_data['oxidation_states']['average_oxidation_states']=new_data['average_oxidation_states']
                main_data['oxidation_states']['method']=new_data['method']

                with open(main_json_file,'w') as f:
                    json.dump(main_data, f, indent=4)

                logger.info("Finished processing database")
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")

            return None
            
    def merge_summary_doc_task(json_file, directory_path=''):
            """
            Merges external summary doc from materials project into the database.

            This function merges external tasks into the database. It iterates over the database files and checks if the
            corresponding task has been completed. If the task has not been completed, it is skipped. If the task has
            been completed, the task is merged into the database.

            Returns:
                None
            """
            with open(json_file) as f:
                new_data = json.load(f)
            try:
                mpid = new_data['material_id']

                main_json_file = os.path.join(directory_path,f'{mpid}.json')
                # Check if main json file exists
                if os.path.exists(main_json_file):
                    with open(main_json_file) as f:
                        main_data = json.load(f)
                else:
                    main_data={}
                
                main_data['nsites']=new_data['nsites']
                main_data['nelements']=new_data['nelements']
                main_data['elements']=new_data['elements']
                main_data['composition']=new_data['composition']
                main_data['composition_reduced']=new_data['composition_reduced']
                main_data['formula_pretty']=new_data['formula_pretty']
                main_data['volume']=new_data['volume']
                main_data['density']=new_data['density']
                main_data['density_atomic']=new_data['density_atomic']
                main_data['symmetry']=new_data['symmetry']
                main_data['structure']=new_data['structure']
                main_data['material_id']=new_data['material_id']
                main_data['last_updated']=new_data['last_updated']
                main_data['uncorrected_energy_per_atom']=new_data['uncorrected_energy_per_atom']
                main_data['energy_per_atom']=new_data['energy_per_atom']
                main_data['formation_energy_per_atom']=new_data['formation_energy_per_atom']
                main_data['energy_above_hull']=new_data['energy_above_hull']
                main_data['is_stable']=new_data['is_stable']
                main_data['equilibrium_reaction_energy_per_atom']=new_data['equilibrium_reaction_energy_per_atom']
                main_data['decomposes_to']=new_data['decomposes_to']
                main_data['grain_boundaries']=new_data['grain_boundaries']
                main_data['band_gap']=new_data['band_gap']
                main_data['cbm']=new_data['cbm']
                main_data['vbm']=new_data['vbm']
                main_data['efermi']=new_data['efermi']
                main_data['is_gap_direct']=new_data['is_gap_direct']
                main_data['is_metal']=new_data['is_metal']
                main_data['ordering']=new_data['ordering']
                main_data['total_magnetization']=new_data['total_magnetization']
                main_data['total_magnetization_normalized_vol']=new_data['total_magnetization_normalized_vol']
                main_data['num_magnetic_sites']=new_data['num_magnetic_sites']
                main_data['num_unique_magnetic_sites']=new_data['num_unique_magnetic_sites']
                main_data['dos_energy_up']=new_data['dos_energy_up']
                main_data['is_magnetic']=new_data['is_magnetic']
                main_data['total_magnetization']=new_data['total_magnetization']
                main_data['total_magnetization_normalized_vol']=new_data['total_magnetization_normalized_vol']
                main_data['num_magnetic_sites']=new_data['num_magnetic_sites']
                main_data['num_unique_magnetic_sites']=new_data['num_unique_magnetic_sites']
                main_data['types_of_magnetic_species']=new_data['types_of_magnetic_species']
                main_data['k_voigt']=new_data['bulk_modulus']['voigt']
                main_data['k_reuss']=new_data['bulk_modulus']['reuss']
                main_data['k_vrh']=new_data['bulk_modulus']['vrh']
                main_data['g_voigt']=new_data['shear_modulus']['voigt']
                main_data['g_reuss']=new_data['shear_modulus']['reuss']
                main_data['g_vrh']=new_data['shear_modulus']['vrh']
                main_data['universal_anisotropy']=new_data['universal_anisotropy']
                main_data['homogeneous_poisson']=new_data['homogeneous_poisson']
                main_data['e_total']=new_data['e_total']
                main_data['e_ionic']=new_data['e_ionic']
                main_data['n']=new_data['n']
                main_data['e_ij_max']=new_data['e_ij_max']
                main_data['weighted_surface_energy_EV_PER_ANG2']=new_data['weighted_surface_energy_EV_PER_ANG2']
                main_data['weighted_surface_energy']=new_data['weighted_surface_energy']
                main_data['weighted_work_function']=new_data['weighted_work_function']
                main_data['surface_anisotropy']=new_data['surface_anisotropy']
                main_data['shape_factor']=new_data['shape_factor']
                main_data['has_reconstructed']=new_data['has_reconstructed']
                main_data['possible_species']=new_data['possible_species']
                main_data['has_props']=new_data['has_props']
                main_data['theoretical']=new_data['theoretical']

                with open(main_json_file,'w') as f:
                    json.dump(main_data, f, indent=4)

                logger.info("Finished processing database")
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")
            return None

    def create_parquet_file_task(json_file):
        """
        Gets all material as a python dictionary in the database.
        Returns:
            None
        """
        data=None
        try:
            with open(json_file) as f:
                data = json.load(f)

            mp_id=json_file.split(os.sep)[-1].split('.')[0]
            data['material_id']=mp_id
            data['name']=mp_id
            data['type']='MATERIAL'

            if 'composition' in data:
                compositions=data.pop('composition')
                data['composition-values']=list(compositions.values())
                data['composition-elements']=list(compositions.keys())
            if 'composition_reduced' in data:
                data.pop('composition_reduced')
                data['composition_reduced-values']=list(compositions.values())
                data['composition_reduced-elements']=list(compositions.keys())

            if 'symmetry' in data:
                symmetry=data.pop('symmetry')
                data['crystal_system']=symmetry.get('crystal_system')
                data['space_group']=symmetry.get('number')
                data['space_group_symbol']=symmetry.get('symbol')
                data['point_group']=symmetry.get('point_group')


            if 'structure' in data:
                structure=data.pop('structure')
                data['lattice']=structure['lattice']['matrix']
                data['pbc']=structure['lattice']['pbc']
                data['a']=structure['lattice']['a']
                data['b']=structure['lattice']['b']
                data['c']=structure['lattice']['c']
                data['alpha']=structure['lattice']['alpha']
                data['beta']=structure['lattice']['beta']
                data['gamma']=structure['lattice']['gamma']
                data['unit_cell_volume']=structure['lattice']['volume']

                frac_coords=[]
                cart_coords=[]
                species=[]
                for site in structure['sites']:
                    frac_coords.append(site['abc'])
                    cart_coords.append(site['xyz'])
                    species.append(site['label'])
                data['frac_coords']=frac_coords
                data['cart_coords']=cart_coords
                data['species']=species
            
            if 'oxidation_states' in data:
                oxidation_states=data.pop('oxidation_states')
                data['oxidation_states-possible_species']=oxidation_states.get('possible_species')
                data['oxidation_states-possible_valences']=oxidation_states.get('possible_valences')
                data['oxidation_states-method']=oxidation_states.get('method')

            if 'feature_vectors' in data:
                feature_vectors=data.pop('feature_vectors')

                tmp_dict=feature_vectors.get('sine_coulomb_matrix')
                if tmp_dict is not None:
                    data['sine_coulomb_matrix']=tmp_dict.get('values')

                tmp_dict=feature_vectors.get('element_property')
                if tmp_dict is not None:
                    data['element_property']=tmp_dict.get('values')

                tmp_dict=feature_vectors.get('element_fraction')
                if tmp_dict is not None:
                    data['element_fraction']=tmp_dict.get('values')

                tmp_dict=feature_vectors.get('xrd_pattern')
                if tmp_dict is not None:
                    data['xrd_pattern']=tmp_dict.get('values')

            if 'has_props' in data:
                has_props=data.pop('has_props')
                data['has_props-materials']=has_props.get('materials')
                data['has_props-thermo']=has_props.get('thermo')
                data['has_props-xas']=has_props.get('xas')
                data['has_props-grain_boundaries']=has_props.get('grain_boundaries')
                data['has_props-chemenv']=has_props.get('chemenv')
                data['has_props-electronic_structure']=has_props.get('electronic_structure')
                data['has_props-absorption']=has_props.get('absorption')
                data['has_props-bandstructure']=has_props.get('bandstructure')
                data['has_props-dos']=has_props.get('dos')
                data['has_props-magnetism']=has_props.get('magnetism')
                data['has_props-elasticity']=has_props.get('elasticity')
                data['has_props-dielectric']=has_props.get('dielectric')
                data['has_props-piezoelectric']=has_props.get('piezoelectric')
                data['has_props-surface_properties']=has_props.get('surface_properties')    
                data['has_props-oxi_states']=has_props.get('oxi_states')
                data['has_props-provenance']=has_props.get('provenance')
                data['has_props-charge_density']=has_props.get('charge_density')
                data['has_props-eos']=has_props.get('eos')
                data['has_props-phonon']=has_props.get('phonon')
                data['has_props-insertion_electrodes']=has_props.get('insertion_electrodes')
                data['has_props-substrates']=has_props.get('substrates')

            if 'elasticity' in data:
                elasticity=data.pop('elasticity')
                data['elasticity-warnings']=elasticity.get('warnings')
                data['elasticity-order']=elasticity.get('order')
                data['elasticity-k_vrh']=elasticity.get('k_vrh')
                data['elasticity-k_reuss']=elasticity.get('k_reuss')
                data['elasticity-k_voigt']=elasticity.get('k_voigt')
                data['elasticity-g_vrh']=elasticity.get('g_vrh')
                data['elasticity-g_reuss']=elasticity.get('g_reuss')
                data['elasticity-g_voigt']=elasticity.get('g_voigt')
                data['elasticity-sound_velocity_transverse']=elasticity.get('sound_velocity_transverse')
                data['elasticity-sound_velocity_longitudinal']=elasticity.get('sound_velocity_longitudinal')
                data['elasticity-sound_velocity_total']=elasticity.get('sound_velocity_total')
                data['elasticity-sound_velocity_acoustic']=elasticity.get('sound_velocity_acoustic')
                data['elasticity-sound_velocity_optical']=elasticity.get('sound_velocity_optical')
                data['elasticity-thermal_conductivity_clarke']=elasticity.get('thermal_conductivity_clarke')
                data['elasticity-thermal_conductivity_cahill']=elasticity.get('thermal_conductivity_cahill')
                data['elasticity-young_modulus']=elasticity.get('young_modulus')
                data['elasticity-universal_anisotropy']=elasticity.get('universal_anisotropy')
                data['elasticity-homogeneous_poisson']=elasticity.get('homogeneous_poisson')
                data['elasticity-debye_temperature']=elasticity.get('debye_temperature')
                data['elasticity-state']=elasticity.get('state')
            
            
            for key in data.keys():
                data[key] = [data.get(key, None)]
            return data
        except Exception as e:
            logger.debug(f"Error processing file {json_file}: {e}")
            return None
    
    def generate_matminer_embeddings_task(file_embedding_tuple):
            """
            Generates the Wyckoff positions for all materials in the database.
            Returns:
                None
            """
            json_file,embedding_dict=file_embedding_tuple
            mpid=json_file.split(os.sep)[-1].split('.')[0]
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if 'feature_vectors' not in data:
                    data['feature_vectors']={}

                data['feature_vectors'].update(embedding_dict['feature_vectors'])

                with open(json_file,'w') as f:
                    json.dump(data, f, indent=None)
            except Exception as e:
                print(f"Error processing file {mpid}: {e}")
                logger.error(f"Error processing file {mpid}: {e}")
                return None
            
    def get_property_task(json_file, property_name='structure'):
        """
        Get the structure for a given json file.

        Args:
            json_file (str): The path to the json file.

        Returns:
            list: A list of structures.
        """

        try:
            with open(json_file) as f:
                data = json.load(f)
                property = data.get(property_name)

                
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")

        return property
    
    def get_structure_task(json_file):
        """
        Get the structure for a given json file.

        Args:
            json_file (str): The path to the json file.

        Returns:
            list: A list of structures.
        """
        structure=None
        material_id=None
        nsites=None
        try:
            with open(json_file) as f:
                data = json.load(f)
                structure = Structure.from_dict(data['structure'])
                material_id=json_file.split(os.sep)[-1].split('.')[0]
                nsites=data['nsites']
                
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")

        return structure, material_id, nsites
    
    def bond_orders_sum_task(json_file):
        """
        Calculates the sum and count of bond orders for a given file.

        Args:
            file (str): The path to the JSON file containing the database.

        Returns:
            tuple: A tuple containing two numpy arrays. The first array represents the sum of bond orders
                between different elements, and the second array represents the count of bond orders.

        Raises:
            Exception: If there is an error processing the file.

        """
        mpid = json_file.split(os.sep)[-1].split('.')[0]
        try:
            # Load database from JSON file
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error processing file {mpid}: {e}")
            data={}
        
        bond_orders=None
        bond_connections=None
        site_element_names=None
        if 'chargemol_bonding_orders' in data:
            bond_orders = data["chargemol_bonding_orders"]
        if 'chargemol_bonding_connections' in data:
            bond_connections = data["chargemol_bonding_connections"]
        if 'structure' in data:
            site_element_names = [x['label'] for x in data['structure']['sites']]


        bond_orders_sum, n_bond_orders = calculate_bond_orders_sum(bond_orders, bond_connections, site_element_names)

        if bond_orders is None or bond_connections is None or site_element_names is None:
            logger.error(f"Error processing file {mpid}: Bond Orders Stats calculation failed")

        return bond_orders_sum, n_bond_orders

    def bond_orders_sum_squared_differences_task(json_file):
        """
        Calculate the sum_squared_differences of bond orders for a given material.

        Parameters:
        file (str): The path to the JSON file containing the material information.

        Returns:
        bond_orders_sum_squared_differences (numpy.ndarray): The sum_squared_differences of bond orders between different elements.
        1 (int): A placeholder value indicating the function has completed successfully.
        """
        mpid = json_file.split(os.sep)[-1].split('.')[0]
        try:
            # Load database from JSON file
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")
            data={}

        with open(GLOBAL_PROP_FILE) as f:
            global_data = json.load(f)
            bond_orders_avg=np.array(global_data['bond_orders_avg'])
            n_bond_orders=np.array(global_data['n_bond_orders'])


        bond_orders=None
        bond_connections=None
        site_element_names=None
        if 'chargemol_bonding_orders' in data:
            bond_orders = data["chargemol_bonding_orders"]
        if 'chargemol_bonding_connections' in data:
            bond_connections = data["chargemol_bonding_connections"]
        if 'structure' in data:
            site_element_names = [x['label'] for x in data['structure']['sites']]

        bond_orders_sum_squared_differences = calculate_bond_orders_sum_squared_differences(bond_orders, bond_connections, site_element_names, bond_orders_avg, n_bond_orders)

        return bond_orders_sum_squared_differences
    
    def bonding_task(json_file):
            # Load data from JSON file
            mpid=json_file.split('/')[-1].split('.')[0]
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    structure=Structure.from_dict(data['structure'])
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")
                return None
            

            geo_coord_connections=None
            elec_coord_connections=None
            chargemol_bond_orders=None
            if 'coordination_multi_connections' in data:
                geo_coord_connections = data['coordination_multi_connections']
            if 'chargemol_bonding_connections' in data:
                elec_coord_connections = data['chargemol_bonding_connections']
                chargemol_bond_orders = data['chargemol_bonding_orders']

            final_geo_connections, final_bond_orders = calculate_geometric_consistent_bonds(geo_coord_connections, elec_coord_connections, chargemol_bond_orders)
            data['geometric_consistent_bond_connections']=final_geo_connections
            data['geometric_consistent_bond_orders']=final_bond_orders

            final_elec_connections, final_bond_orders = calculate_electric_consistent_bonds(elec_coord_connections, chargemol_bond_orders)
            data['electric_consistent_bond_connections']=final_elec_connections
            data['electric_consistent_bond_orders']=final_bond_orders

            final_geo_elec_connections, final_bond_orders = calculate_geometric_electric_consistent_bonds(geo_coord_connections, elec_coord_connections, chargemol_bond_orders)
            data['geometric_electric_consistent_bond_connections']=final_geo_elec_connections
            data['geometric_electric_consistent_bond_orders']=final_bond_orders

            final_cutoff_connections = calculate_cutoff_bonds(structure)
            data['bond_cutoff_connections']=final_cutoff_connections

            with open(json_file,'w') as f:
                json.dump(data, f, indent=4)

            error_messge=""
            if final_geo_connections is None:
                error_messge+="| Geometric Consistent Bonding |"
            if final_elec_connections is None:
                error_messge+="| Electric Consistent Bonding |"
            if final_geo_elec_connections is None:
                error_messge+="| Geometric Electric Consistent Bonding |"
            if final_cutoff_connections is None:
                error_messge+="| Bond Cutoff |"
            if len(error_messge)>0:
                logger.error(f"Error processing file {mpid}: {error_messge} calculation failed")

            return None
    
    def chemenv_task(json_file, from_scratch=False):

            # Load data from JSON file
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    struct = Structure.from_dict(data['structure'])
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {e}")
                return None



            # Extract material project ID from file name
            mpid = json_file.split(os.sep)[-1].split('.')[0]

            coordination_environments=data.get('coordination_environments_multi_weight')
            nearest_neighbors=data.get('coordination_multi_connections')
            coordination_numbers=data.get('coordination_multi_numbers')
            # Check if calculation is needed
            if coordination_environments is None or from_scratch:
                coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(struct)
                # Update the database with computed values
                data['coordination_environments_multi_weight'] = coordination_environments
                data['coordination_multi_connections'] = nearest_neighbors
                data['coordination_multi_numbers'] = coordination_numbers
            

                with open(json_file,'w') as f:
                    json.dump(data, f, indent=4)
        

            if coordination_environments is None:
                logger.error(f"Error processing file {mpid}: Coordination Environments calculation failed")

            return None
    
    def wyckoff_calc_task(json_file):
            """
            Perform Wyckoff calculations on all materials in the database.

            Returns:
                None
            """
            mpid=json_file.split(os.sep)[-1].split('.')[0]
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    struct = Structure.from_dict(data['structure'])
            except Exception as e:
                logger.error(f"Error processing file {mpid}: {e}")
                return None
            
            wyckoffs=None
            if 'wyckoffs' not in data:
                wyckoffs=calculate_wyckoff_positions(struct)
                data['wyckoffs']=wyckoffs

            with open(json_file,'w') as f:
                json.dump(data, f, indent=4)
            
            if wyckoffs is None:
                logger.error(f"Error processing file {mpid}: Wyckoff Positions calculation failed")

    def collect_chargemol_info_task(dir, directory_path=''):
        """Check if a given property exists in the data."""
        material_id=dir.split(os.sep)[-1]
        json_file=os.path.join(directory_path, material_id+'.json')
        bond_orders_file = os.path.join(dir,'chargemol','DDEC6_even_tempered_bond_orders.xyz')
        squared_moments_file = os.path.join(dir,'chargemol','DDEC_atomic_Rsquared_moments.xyz')
        cubed_moments_file = os.path.join(dir,'chargemol','DDEC_atomic_Rcubed_moments.xyz')
        fourth_moments_file = os.path.join(dir,'chargemol','DDEC_atomic_Rfourth_moments.xyz')
        atomic_charges_file = os.path.join(dir,'chargemol','DDEC6_even_tempered_net_atomic_charges.xyz')
        overlap_population_file = os.path.join(dir,'chargemol','overlap_populations.xyz')
        

        bond_order_info=parse_chargemol_bond_orders(file=bond_orders_file)
        net_atomic_charges_info=parse_chargemol_net_atomic_charges(file=atomic_charges_file)
        overlap_population_info=parse_chargemol_overlap_populations(file=overlap_population_file)
        squared_moments_info=parse_chargemol_atomic_moments(file=squared_moments_file)
        cubed_moments_info=parse_chargemol_atomic_moments(file=cubed_moments_file)
        fourth_moments_info=parse_chargemol_atomic_moments(file=fourth_moments_file)

        with open(json_file, 'r') as file:
            data = json.load(file)

        data['chargemol_bonding_connections'] = bond_order_info[0]
        data['chargemol_bonding_orders'] = bond_order_info[1]
        # data['chargemol_net_atomic_charges'] = net_atomic_charges_info
        # data['chargemol_overlap_populations'] = overlap_population_info
        data['chargemol_squared_moments'] = squared_moments_info
        data['chargemol_cubed_moments'] = cubed_moments_info
        data['chargemol_fourth_moments'] = fourth_moments_info

        with open(json_file,'w') as f:
            json.dump(data, f, indent=4)

        if bond_order_info is None:
            logger.error(f"Error processing file {material_id}: Chargemol Bonding Orders calculation failed")

        return None
    
    def check_chargemol_task(dir):
            """
            Check if the chargemol task has been completed.

            Args:
                dir (str): The directory path where the chargemol task output is expected.

            Returns:
                bool: True if the chargemol task has been completed and the output file exists, False otherwise.
            """
            check = True

            file_path = os.path.join(dir, 'chargemol', 'DDEC6_even_tempered_bond_orders.xyz')

            if os.path.exists(file_path):
                check = True
            else:
                check = False

            return check
    
    def check_property_task(json_file, property_name=''):
        """
        Check if a given property exists in the data loaded from a JSON file.

        Args:
            file (str): The path to the JSON file.
            property_name (str, optional): The name of the property to check. Defaults to ''.

        Returns:
            bool: True if the property exists and is not None, False otherwise.
        """

        mpid=json_file.split('/')[-1].split('.')[0]
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error processing file {mpid}: {e}")
            data={}
            
        check=True
        if property_name not in data:
            check=False
            return check

        if data[property_name] is None:
            check=False

        return check
    
    def load_data_tasks(json_file):
        """
        Loads the data tasks.

        Returns:
            dict: A dictionary of data tasks.
        """
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Error processing file: {e}")
            return None

        return data
    

def get_data(data, key, default=None):
    return [data.get(key, default)]



if __name__=='__main__':

    properties=['chargemol_bonding_orders','coordination_environments_multi_weight','coordination_multi_connections',
                'geometric_consistent_bond_connections','electric_consistent_bond_connections','geometric_electric_consistent_bond_connections']

    data=DBManager()
    print("geometric consistent bond connections")
    success,failed=data.check_property(property_name=properties[1])
    print("Number of failed files: ", len(failed))
    print("Number of success files: ", len(success))

    # print("electric consistent bond connections")
    # success,failed=data.check_property(property_name=properties[4])
    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))

    # print("geometric electric consistent bond connections")
    # success,failed=data.check_property(property_name=properties[5])
    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))




    # data.create_material(composition='Li2O')
    # Define the structure

    # file=data.database_files[0]
    # structure = Structure.from_dict(data.load_json(file)['structure'])
    # print(structure)
    # # structure = Structure(
    # #     Lattice.cubic(3.0),
    # #     ["C", "C"],  # Elements
    # #     [
    # #         [0, 0, 0],          # Coordinates for the first Si atom
    # #         [0.25, 0.25, 0.25],  # Coordinates for the second Si atom (basis of the diamond structure)
    # #     ]
    # # )
    # data.create_material(structure=structure)


    
    #Create a test structure

    # success,failed=data.check_property(property_name=properties[0])


    # data.chargemol_task(dir=data.calculation_dirs()[0])
    # print(N_CORES)
    # data.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1') )

    # data.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00'),exclude=[] )
    # success,failed=data.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])
    # data.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1'),exclude=[] )
    # success,failed=data.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])

    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))
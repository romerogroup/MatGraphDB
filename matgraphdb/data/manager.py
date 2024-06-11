import os
import json
from glob import glob
from typing import Dict, List, Tuple, Union
from multiprocessing import Pool
from functools import partial

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition,Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES,LOGGER, GLOBAL_PROP_FILE,ENCODING_DIR

from matgraphdb.calculations.mat_calcs.bonding_calc import calculate_cutoff_bonds,calculate_electric_consistent_bonds,calculate_geometric_consistent_bonds,calculate_geometric_electric_consistent_bonds
from matgraphdb.calculations.mat_calcs.bond_stats_calc import calculate_bond_orders_sum,calculate_bond_orders_sum_squared_differences
from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections
from matgraphdb.calculations.mat_calcs.embeddings import generate_composition_embeddings,generate_openai_embeddings,extract_text_from_json
from matgraphdb.calculations.mat_calcs.wyckoff_calc import calculate_wyckoff_positions
from matgraphdb.calculations.parsers import parse_chargemol_bond_orders,parse_chargemol_net_atomic_charges, parse_chargemol_atomic_moments, parse_chargemol_overlap_populations
from matgraphdb.utils.periodic_table import atomic_symbols

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

    def process_task(self, func, list,**kwargs):
        LOGGER.info(f"Process full database using {self.n_cores} cores")
        print(f"Using {self.n_cores} cores")
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
    
    def index_files(self, files):
        """
        Indexes a list of files.
        Args:
            files (list): A list of file paths.
        Returns:
            dict: A dictionary where the keys are the filenames and the values are the indexed filenames.

        """
        return {file.split(os.sep)[-1]:f"m-{i}" for i,file in enumerate(files)}

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
    
    def check_property_task(self, file, property_name=''):
        """
        Check if a given property exists in the data loaded from a JSON file.

        Args:
            file (str): The path to the JSON file.
            property_name (str, optional): The name of the property to check. Defaults to ''.

        Returns:
            bool: True if the property exists and is not None, False otherwise.
        """
        data = self.load_json(file)
        
        check=True
        if property_name not in data:
            check=False
            return check

        if data[property_name] is None:
            check=False
 
        return check
    
    def check_property(self, property_name):
        """Check if a given property exists in all JSON files and categorize them."""
        
        database_files = self.database_files()
        print("Processing files from : ",self.directory_path + os.sep + '*.json')
        results=self.process_task(self.check_property_task, database_files, property_name=property_name)

        success = []
        failed = []
        for file, result in zip(database_files, results):
            if result == True:
                success.append(file)
            else:
                failed.append(file)

        return success, failed

    def check_chargemol_task(self, dir):
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
    
    def check_chargemol(self):
        """Check if a given property exists in all JSON files and categorize them."""
        
        calc_dirs = self.calculation_dirs()
        print("Processing files from : ",self.calculation_path + os.sep + 'mp-*')
        results=self.process_task(self.check_chargemol_task, calc_dirs)

        success = []
        failed = []
        for path, result in zip(calc_dirs,results):
            chargemol_dir=os.path.join(path,'chargemol')
            if result==True:
                success.append(chargemol_dir)
            else:
                failed.append(chargemol_dir)

        return success, failed

    def add_chargemol_slurm_script(self, partition_info=('comm_small_day','24:00:00','16', '1'), exclude=[]):
        """
        Adds a SLURM script for running Chargemol calculations to each calculation directory.

        Args:
            partition_info (tuple): A tuple containing information about the SLURM partition.
                Default is ('comm_small_day','24:00:00','16', '1').
            exclude (list): A list of nodes to exclude from the SLURM job. Default is an empty list.

        Returns:
            None
        """
        calc_dirs = glob(self.calculation_path + os.sep + 'mp-*')
        LOGGER.info("Processing files from : ",self.calculation_path + os.sep + 'mp-*')
        results=self.process_task(self.check_chargemol_task, calc_dirs)

        for path, result in zip(calc_dirs[:],results[:]):

            

            if result==False:
                with open(os.path.join(path,'POSCAR')) as f:
                    lines=f.readlines()
                    raw_natoms=lines[6].split()
                    natoms=0
                    for raw_natom in raw_natoms:
                        natoms+=int(raw_natom)

                # Read INCAR and modify NCORE and KPAR
                incar_path = os.path.join(path, 'chargemol','INCAR')
                with open(incar_path, 'r') as file:
                    incar_lines = file.readlines()

                if natoms >= 60:  
                    nnode=4
                    ncore = 32  
                    kpar=4   
                    ntasks=160
                elif natoms >= 40:  
                    nnode=3
                    ncore = 20  
                    kpar=3  
                    ntasks=120
                elif natoms >= 20: 
                    nnode=2
                    ncore = 16  
                    kpar = 2   
                    ntasks=80
                else:
                    nnode=1
                    ntasks=40
                    ncore = 40 
                    kpar = 1

                with open(incar_path, 'w') as file:
                    for line in incar_lines:
                        if line.strip().startswith('NCORE'):
                            file.write(f'NCORE = {ncore}\n')
                        elif line.strip().startswith('KPAR'):
                            file.write(f'KPAR = {kpar}\n')
                        else:
                            file.write(line)



                chargemol_dir=os.path.join(path,'chargemol')
                sumbit_script=os.path.join(chargemol_dir,'run.slurm')
                with open(sumbit_script, 'w') as file:
                    file.write('#!/bin/bash\n')
                    file.write('#SBATCH -J mp_database_chargemol\n')
                    file.write(f'#SBATCH --nodes={nnode}\n')
                    file.write(f'#SBATCH -n {ntasks}\n')
                    file.write(f'#SBATCH -p {partition_info[0]}\n')
                    file.write(f'#SBATCH -t {partition_info[1]}\n')
                    if exclude:
                        node_list_string= ','.join(exclude)
                        file.write(f'#SBATCH --exclude={node_list_string}\n')
                    file.write(f'#SBATCH --output={chargemol_dir}/jobOutput.out\n')
                    file.write(f'#SBATCH --error={chargemol_dir}/jobError.err\n')
                    file.write('\n')
                    file.write('source ~/.bashrc\n')
                    file.write('module load atomistic/vasp/6.2.1_intel22_impi22\n')
                    file.write(f'cd {chargemol_dir}\n')
                    file.write(f'echo "CALC_DIR: {chargemol_dir}"\n')
                    file.write(f'echo "NCORES: $((SLURM_NTASKS))"\n')
                    file.write('\n')
                    file.write(f'mpirun -np $SLURM_NTASKS vasp_std\n')
                    file.write('\n')
                    file.write(f'export OMP_NUM_THREADS=$SLURM_NTASKS\n')
                    file.write('~/SCRATCH/Codes/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries'
                    '/linux/Chargemol_09_26_2017_linux_parallel> chargemol_debug.txt 2>&1\n')
                    file.write('\n')
                    file.write(f'echo "run complete on `hostname`: `date`" 1>&2\n')

    def chargemol_task(self, dir):
        """Check if a given property exists in the data."""
        material_id=dir.split(os.sep)[-1]
        json_file=os.path.join(self.directory_path,material_id+'.json')
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
            LOGGER.error(f"Error processing file {material_id}: Chargemol Bonding Orders calculation failed")

        return None
        
    def collect_chargemol_info(self):
        
        calc_dirs = self.calculation_dirs()
        self.process_task(self.chargemol_task, calc_dirs)
        LOGGER.info(f"Finished collection Chargemol information")

    def chemenv_task(self, json_file, from_scratch=False):

        # Load data from JSON file
        with open(json_file) as f:
            data = json.load(f)
            struct = Structure.from_dict(db['structure'])

        # Extract material project ID from file name
        mpid = json_file.split(os.sep)[-1].split('.')[0]

        # Check if calculation is needed
        if 'coordination_environments_multi_weight' not in data or from_scratch:
            coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(struct)
            # Update the database with computed values
            data['coordination_environments_multi_weight'] = coordination_environments
            data['coordination_multi_connections'] = nearest_neighbors
            data['coordination_multi_numbers'] = coordination_numbers
        

            with open(json_file,'w') as f:
                json.dump(data, f, indent=4)

        if coordination_environments is None:
            LOGGER.error(f"Error processing file {mpid}: Coordination Environments calculation failed")

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
  
        LOGGER.info(f"Starting collection ChemEnv Calculations")
        # Process the database with the defined function
        files=self.database_files()
        self.process_task(self.chemenv_task,files,from_scratch)
        LOGGER.info(f"Finished collection ChemEnv Calculations")

        return None

    def bonding_task(self, json_file):
        # Load data from JSON file
        with open(json_file) as f:
            data = json.load(f)
            structure=Structure.from_dict(db['structure'])

        mpid=json_file.split('/')[-1].split('.')[0]

        geo_coord_connections = data['coordination_multi_connections']
        elec_coord_connections = data['chargemol_bonding_connections']
        chargemol_bond_orders = data['chargemol_bonding_orders']

        final_geo_connections, final_bond_orders = calculate_geometric_consistent_bonds(geo_coord_connections, elec_coord_connections, chargemol_bond_orders)
        data['geometric_consistent_bond_connections']=final_geo_connections
        data['geometric_consistent_bond_orders']=final_bond_orders

        final_elec_connections, final_bond_orders = calculate_electric_consistent_bonds(elec_coord_connections, chargemol_bond_orders)
        data['electric_consistent_bond_connections']=final_elec_connections
        data['electric_consistent_bond_orders']=final_bond_orders

        final_geo_elec_connections, final_bond_orders = calculate_geometric_electric_consistent_bonds(final_geo_connections, final_elec_connections, final_bond_orders)
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
            LOGGER.error(f"Error processing file {mpid}: {error_messge} calculation failed")

    def bonding_calc(self):

        LOGGER.info(f"Starting collection Bonding Calculations")
        # Process the database with the defined function
        files=self.database_files()
        self.process_task(self.bonding_task,files)
        LOGGER.info(f"Finished collection Bonding Calculations")

        return None
    
    def bond_orders_sum_task(self, json_file):
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

        # Load database from JSON file
        with open(json_file) as f:
            data = json.load(f)
        
        # Extract material project ID from file name
        mpid = json_file.split(os.sep)[-1].split('.')[0]

   
        try:
            bond_orders = data["chargemol_bonding_orders"]
            bond_connections = data["chargemol_bonding_connections"]
            site_element_names = [x['label'] for x in data['structure']['sites']]

            bond_orders_sum, n_bond_orders = calculate_bond_orders_sum(bond_orders, bond_connections, site_element_names)

        except Exception as e:
            LOGGER.error(f"Error processing file {mpid}")

        return bond_orders_sum, n_bond_orders
    
    def bond_orders_sum_squared_differences_task(self, json_file):
        """
        Calculate the sum_squared_differences of bond orders for a given material.

        Parameters:
        file (str): The path to the JSON file containing the material information.

        Returns:
        bond_orders_sum_squared_differences (numpy.ndarray): The sum_squared_differences of bond orders between different elements.
        1 (int): A placeholder value indicating the function has completed successfully.
        """

        # Load database from JSON file
        with open(json_file) as f:
            db = json.load(f)

        with open(GLOBAL_PROP_FILE) as f:
            data = json.load(f)
            bond_orders_avg=np.array(data['bond_orders_avg'])
            n_bond_orders=np.array(data['n_bond_orders'])

        # Extract material project ID from file name
        mpid = json_file.split(os.sep)[-1].split('.')[0]

        # Initialize arrays for bond order calculations
        n_elements = len(n_bond_orders)
        bond_orders_sum_squared_differences = np.zeros(shape=(n_elements, n_elements))
        try:
            bond_orders = db["chargemol_bonding_orders"]
            bond_connections = db["chargemol_bonding_connections"]
            site_element_names = [x['label'] for x in db['structure']['sites']]

            bond_orders_sum_squared_differences = calculate_bond_orders_sum_squared_differences(bond_orders, bond_connections, site_element_names, bond_orders_avg, n_bond_orders)

        except Exception as e:
            LOGGER.error(f"Error processing file {mpid}")
        return bond_orders_sum_squared_differences
    
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


        LOGGER.info(f"Starting collection Bond Orders Sum Calculations")
        # Process the database with the defined function
        files=self.database_files()
        results=self.process_task(self.bond_orders_sum_task,files)
        LOGGER.info(f"Finished collection Bond Orders Sum Calculations")
        # Initialize arrays for bond order calculations
        
        LOGGER.info("Starting calculation of bond order average")
        # Calculate the average of the bond orders in the database
        for result in results:
            bond_orders_avg += result[0] 
            n_bond_orders += result[1]
        bond_orders_avg = np.divide(bond_orders_avg, n_bond_orders, out=np.zeros_like(bond_orders_avg), where=n_bond_orders != 0)
        with open(GLOBAL_PROP_FILE) as f:
            data = json.load(f)
            data['bond_orders_avg'] = bond_orders_avg.tolist()
            data['n_bond_orders'] = n_bond_orders.tolist()
        LOGGER.info("Finished calculation of bond order average")

        LOGGER.info("Starting calculation of bond order standard deviation")
        # Calculate the standard deviation of the bond orders in the database
        results=self.process_task(self.bond_orders_sum_squared_differences_task,files)

        for result in results:
            #This results in the material database sum of the sum of squared differences for materials
            bond_orders_std += result[0] 
        bond_orders_std = np.divide(bond_orders_std, n_bond_orders, out=np.zeros_like(bond_orders_std), where=n_bond_orders != 0)
        bond_orders_std = bond_orders_std ** 0.5

        with open(GLOBAL_PROP_FILE) as f:
            data = json.load(f)
            data['bond_orders_std'] = bond_orders_std.tolist()
        LOGGER.info("Finished calculation of bond order standard deviation")

        LOGGER.info("Saving bond order average and standard deviation to global property file")
        with open(GLOBAL_PROP_FILE, 'w') as f:
            json.dump(data, f, indent=4)

        LOGGER.info("Finished calculation of bond order statistics")


        return None

    def generate_composition_embeddings(self):
        compositions=[]
        material_ids=[]
        files=self.database_files()
        for material_file in files:
            material_id=material_file.split(os.sep)[-1].split('.')[0]
            with open(material_file) as f:
                data = json.load(f)
                struct = Structure.from_dict(data['structure'])
                compositions.append(struct.composition)

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

    def extract_text_from_json_task(self,json_file):
        """
        Extracts specific text data from a JSON file and returns it as a compact JSON string.

        Args:
            json_file (str): The path to the JSON file.

        Returns:
            str: A compact JSON string containing the extracted text data.
        """
        compact_json_text = extract_text_from_json(json_file)
        return compact_json_text
    
    def generate_openai_embeddings(self,
                                model="text-embedding-3-small",
                                embedding_encoding = "cl100k_base"
                               ):
        """
        Main function for processing database and generating embeddings using OpenAI models.

        This function performs the following steps:
        1. Sets up the parameters for the models and cost per token.
        2. Initializes the OpenAI client using the API key.
        3. Processes the database and extracts raw JSON text.
        4. Calculates the total number of tokens and the cost.
        5. Retrieves the mp_ids from the database directory.
        6. Creates a dataframe of the results and adds the ada_embedding column.
        7. Creates a dataframe of the embeddings.
        8. Saves the embeddings to a CSV file.

        Args:
            model (str): The name of the OpenAI model to use for embedding. Default is "text-embedding-3-small". 
                            Possible values are "text-embedding-3-small", "text-embedding-3-large", and "ada v2".
            embedding_encoding (str): The name of the encoding to use for embedding. Default is "cl100k_base".

        Returns:
            None
        """
        LOGGER.info("Starting collection OpenAI Embeddings")
        # Get the mp_ids

        LOGGER.info("Processing files from : ",self.database_files())
        files=self.database_files()
        material_ids=[ file.split(os.sep)[-1].split('.')[0] for file in files if file.endswith('.json')]
        # Extracting raw json text from the database
        results=self.process_task(self.extract_text_from_json_task,files)
        LOGGER.info("Finished processing database")

        LOGGER.info("Starting calculation of OpenAI Embeddings")
        # Create a dataframe of the embeddings
        df_embeddings=generate_openai_embeddings(model=model,embedding_encoding=embedding_encoding)
        LOGGER.info("Finished calculation of OpenAI Embeddings")

        # Save the embeddings to a csv file
        df_embeddings.to_csv(os.path.join(ENCODING_DIR,model + ".csv"), index=True)
        LOGGER.info("Finished collection OpenAI Embeddings")
        return None

    def wyckoff_calc_task(self,json_file):
        """
        Perform Wyckoff calculations on all materials in the database.

        Returns:
            None
        """
        with open(json_file) as f:
            data = json.load(f)
            struct = Structure.from_dict(data['structure'])
        mpid=json_file.split(os.sep)[-1].split('.')[0]

        if 'wyckoffs' not in data:
            wyckoffs=calculate_wyckoff_positions(struct)
            db['wyckoffs']=wyckoffs

        with open(json_file,'w') as f:
            json.dump(data, f, indent=4)
        
        if wyckoffs is None:
            LOGGER.error(f"Error processing file {mpid}: Wyckoff Positions calculation failed")
        
    def generate_wyckoff_positions(self):
        """
        Generates the Wyckoff positions for all materials in the database.

        Returns:
            None
        """
        LOGGER.info("Starting collection Wyckoff Positions")

        files=self.database_files()
        results=self.process_task(self.wyckoff_calc_task,files)
        LOGGER.info("Finished processing database")
        return None


if __name__=='__main__':

    # properties=['chargemol_bonding_orders','coordination_environments_multi_weight']

    db=DBManager()
    # success,failed=db.check_property(property_name=properties[0])
    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))


    # db.create_material(composition='Li2O')
    # Define the structure

    # file=db.database_files[0]
    # structure = Structure.from_dict(db.load_json(file)['structure'])
    # print(structure)
    # # structure = Structure(
    # #     Lattice.cubic(3.0),
    # #     ["C", "C"],  # Elements
    # #     [
    # #         [0, 0, 0],          # Coordinates for the first Si atom
    # #         [0.25, 0.25, 0.25],  # Coordinates for the second Si atom (basis of the diamond structure)
    # #     ]
    # # )
    # db.create_material(structure=structure)


    
    #Create a test structure

    # success,failed=db.check_property(property_name=properties[0])


    # db.chargemol_task(dir=db.calculation_dirs()[0])
    # print(N_CORES)
    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1') )

    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00'),exclude=[] )
    # success,failed=db.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])
    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1'),exclude=[] )
    # success,failed=db.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])

    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))
import os
import json
from glob import glob
from typing import Dict, List, Tuple, Union
from multiprocessing import Pool
from functools import partial

import numpy as np
from pymatgen.core import Structure, Composition,Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES
from matgraphdb.calculations.mat_calcs.wyckoff_calc import wyckoff_calc_task
from matgraphdb.calculations.mat_calcs.bonding_calc import calculate_cutoff_bonds
from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections
class DatabaseManager:
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
    def database_files(self):
        """
        Returns a list of JSON file paths in the specified directory.

        Returns:
            list: A list of JSON file paths.
        """
        return glob(self.directory_path + os.sep + '*.json')
    
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
    
    def index_files(self, files):
        """
        Indexes a list of files.

        Args:
            files (list): A list of file paths.

        Returns:
            dict: A dictionary where the keys are the filenames and the values are the indexed filenames.

        """
        return {file.split(os.sep)[-1]:f"m-{i}" for i,file in enumerate(files)}

    def load_json(self, file):
            """
            Load a JSON file given its filename.

            Args:
                file (str): The filename of the JSON file to load.

            Returns:
                dict: The contents of the JSON file as a dictionary.

            Raises:
                FileNotFoundError: If the specified file is not found.

            """
            print(os.path.join(self.directory_path, file))
            try:
                with open(os.path.join(self.directory_path, file), 'r') as file:
                    return json.load(file)
            except FileNotFoundError as e:
                print(f"File not found: {file}")
                return {}
        
    def process_task(self, func, list, **kwargs):
            """
            Process a task using multiple cores in parallel.

            Args:
                func (function): The function to be executed in parallel.
                list (list): The list of inputs to be processed.
                **kwargs: Additional keyword arguments to be passed to the function.

            Returns:
                list: The results of the function execution.

            """
            with Pool(self.n_cores) as p:
                results = p.map(partial(func, **kwargs), list)
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

        check = True
        if property_name in data:
            if data[property_name] is None:
                check = False
            else:
                check = True
        else:
            check = False

        return check
    
    def check_property(self, property_name):
        """
        Check the specified property for all database files.

        Args:
            property_name (str): The name of the property to check.

        Returns:
            tuple: A tuple containing two lists - the list of files where the property check succeeded
                   and the list of files where the property check failed.
        """
        database_files = glob(self.directory_path + os.sep + '*.json')
        print("Processing files from: ", self.directory_path + os.sep + '*.json')
        results = self.process_task(self.check_property_task, database_files, property_name=property_name)

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
        """Check if a given property exists in all JSON files and categorize them.

        This method checks if a given property exists in all JSON files within the specified calculation path.
        It categorizes the directories based on whether the property exists or not.

        Returns:
            A tuple containing two lists:
            - success: A list of directories where the property exists.
            - failed: A list of directories where the property does not exist.
        """
        calc_dirs = glob(self.calculation_path + os.sep + 'mp-*')
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
        print("Processing files from : ",self.calculation_path + os.sep + 'mp-*')
        results=self.process_task(self.check_chargemol_task, calc_dirs)

        for path, result in zip(calc_dirs,results):
            if result==False:
                chargemol_dir=os.path.join(path,'chargemol')
                sumbit_script=os.path.join(chargemol_dir,'run.slurm')
                with open(sumbit_script, 'w') as file:
                    file.write('#!/bin/bash\n')
                    file.write('#SBATCH -J mp_database_chargemol\n')
                    file.write(f'#SBATCH --nodes={partition_info[3]}\n')
                    file.write(f'#SBATCH -c {partition_info[2]}\n')
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
                    file.write('export NUM_CORES=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_ON_NODE))\n')
                    file.write(f'cd {chargemol_dir}\n')
                    file.write(f'echo "CALC_DIR: {chargemol_dir}"\n')
                    file.write(f'echo "NCORES: $((NUM_CORES))"\n')
                    file.write('\n')
                    file.write(f'mpirun -np $NUM_CORES vasp_std\n')
                    file.write('\n')
                    file.write(f'export OMP_NUM_THREADS=$NUM_CORES\n')
                    file.write('~/SCRATCH/Codes/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries'
                    '/linux/Chargemol_09_26_2017_linux_parallel> chargemol_debug.txt 2>&1\n')
                    file.write('\n')
                    file.write(f'echo "run complete on `hostname`: `date`" 1>&2\n')

        



            

if __name__=='__main__':

    properties=['chargemol_bonding_orders','coordination_environments_multi_weight']

    db=DatabaseManager()

    # db.create_material(composition='Li2O')
    # Define the structure

    file=db.database_files[0]
    structure = Structure.from_dict(db.load_json(file)['structure'])
    print(structure)
    # structure = Structure(
    #     Lattice.cubic(3.0),
    #     ["C", "C"],  # Elements
    #     [
    #         [0, 0, 0],          # Coordinates for the first Si atom
    #         [0.25, 0.25, 0.25],  # Coordinates for the second Si atom (basis of the diamond structure)
    #     ]
    # )
    db.create_material(structure=structure)


    
    #Create a test structure

    # success,failed=db.check_property(property_name=properties[0])


    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1') )

    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1'),exclude=[] )
    # success,failed=db.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])

    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))
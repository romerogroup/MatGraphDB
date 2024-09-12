import os
import json

import subprocess
from typing import Callable, Dict, List, Tuple, Union
from multiprocessing import Pool
from functools import partial

from matgraphdb.calculations.job_scheduler_generator import SlurmScriptGenerator
from matgraphdb.utils import N_CORES


class CalculationManager:
    def __init__(self, main_dir, db_manager, n_cores=N_CORES, job_submission_script_name='run.slurm'):
        """
        Initializes the CalculationManager with the specified main directory, database manager, number of cores,
        and the name of the job submission script.
        
        Parameters:
        main_dir (str): Main directory path for calculations.
        db_manager (object): Database manager object for handling database operations.
        n_cores (int): Number of cores to use for multiprocessing.
        job_submission_script_name (str): Name of the job submission script.
        """
        self.db_manager = db_manager
        self.main_dir = main_dir
        self.n_cores = n_cores
        self.job_submission_script_name = job_submission_script_name

        self.calculation_dir = os.path.join(self.main_dir, 'MaterialsData')
        self.metadata_file = os.path.join(self.main_dir, 'metadata.json')

        os.makedirs(self.calculation_dir, exist_ok=True)

        self.initialized=False
        print("Make sure to initialize the calculation manager before using it")
        
    def _process_task(self, func, list, **kwargs):
        """
        Processes a task in parallel using multiprocessing.
        
        Parameters:
        func (callable): The function to apply to the task list.
        list (list): List of tasks to process.
        kwargs: Additional arguments for the function.
        
        Returns:
        list: Results from processing the tasks.
        """
        with Pool(self.n_cores) as p:
            results=p.map(partial(func,**kwargs), list)
        return results

    def _setup_material_directory(self, directory):
        """
        Creates the directory structure for a specific material if it doesn't exist.
        
        Parameters:
        directory (str): Path to the material directory.
        
        Returns:
        None
        """
        os.makedirs(directory, exist_ok=True)
        return None

    def _setup_material_directories(self):
        """
        Creates directories for all materials in the database and returns their paths.
        
        Returns:
        list: A list of material directory paths.
        """
        rows=self.db_manager.read()

        material_dirs = []
        for i, row in enumerate(rows):
            material_id = row.id
            material_directory = os.path.join(self.calculation_dir, material_id)
            self._setup_material_directory(material_directory)
            material_dirs.append(material_directory)
        return material_dirs
    
    def initialize(self):
        """
        Initializes the CalculationManager by loading metadata and setting up material directories.
        """
        self.metadata = self.load_metadata()
        self.material_dirs = self._setup_material_directories()
        self.initialized=True
    
    def get_calculation_names(self):
        """
        This method returns a list of all calculation names in the database.
        """
        calculation_names = os.listdir(self.material_dirs[0])
        self.update_metadata({'calculation_names': calculation_names})
        return calculation_names

    def create_calc(self, calc_func: Callable, calc_name: str = None, **kwargs):
        """
        This method creates a new calculation by applying the provided function to each row in the database. 
        The function is expecting an input of type dictionary-like object (each row's data) and a directory path 
        where the calculation results should be stored. The function is responsible for saving the results in this 
        directory, which is specific to each row and named based on the row's unique ID and the name of the 
        calculation function.

        Args:
            calc_func (Callable): The function to apply to each material.
            calc_name (str, optional): The name of the calculation. Defaults to the name of the function.
            **kwargs: Additional arguments to pass to the calculation function.

        Returns:
            List: The results of the calculation for each material.
        """
        if calc_name is None:
            calc_name = calc_func.__name__
    
        # Read data from the database
        rows = self.db_manager.read()

        multi_task_list=[]
        i=0
        for row in rows:
            row_data=row.data
            calc_dir=os.path.join(self.material_dirs[i],calc_name)

            multi_task_list.append((row_data,calc_dir))
            i+=1

        # Process each row using multiprocessing, passing the directory structure
        results = self._process_task(calc_func, multi_task_list, **kwargs)

        return results
    
    def generate_job_scheduler_script_for_calc(self, calc_dir: str, slurm_config: Dict = None, slurm_script: str = None):
        """
        Generates a SLURM job scheduler submission script for a specific calculation.

        Args:
            calc_dir (str): The directory where the calculation is stored.
            slurm_config (Dict, optional): Configuration settings for the SLURM script. Defaults to None.
            slurm_script (str, optional): Predefined SLURM script content. Defaults to None.

        Returns:
            Tuple: The path to the generated SLURM script and its content.
        """
        calc_name=os.path.basename(calc_dir)
        material_id=os.path.basename(os.path.dirname(calc_dir))

        if slurm_config:
            slurm_generator = SlurmScriptGenerator(
                job_name=slurm_config.get('job_name', f"{calc_name}_calc_{material_id}"),
                partition=slurm_config.get('partition', 'comm_small_day'),
                time=slurm_config.get('time', '24:00:00')
            )
            
            slurm_generator.init_header()
            slurm_generator.add_slurm_header_comp_resources(
                n_nodes=slurm_config.get('n_nodes'),
                n_tasks=slurm_config.get('n_tasks'),
                cpus_per_task=slurm_config.get('cpus_per_task')
            )
            slurm_generator.add_slurm_script_body(f"cd {calc_dir}")
            slurm_generator.add_slurm_script_body(slurm_config.get('command', './run_calculation.sh'))

            slurm_script = slurm_generator.finalize()
            
            # Save the SLURM script in the job directory
            slurm_script_path = os.path.join(calc_dir, self.job_submission_script_name)
            
        
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script)

        return slurm_script_path, slurm_script
    
    def generate_job_scheduler_script_for_calcs(self, calc_name, slurm_config: Dict = None, slurm_script: str = None, **kwargs):
        """
        Generates job scheduler submission scripts for all materials using the specified calculation name.

        Args:
            calc_name (str): The name of the calculation.
            slurm_config (Dict, optional): Configuration settings for the SLURM script. Defaults to None.
            slurm_script (str, optional): Predefined SLURM script content. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the script generator.

        Returns:
            List: The results of script generation for each material.
        """
        multi_task_list=[]
        for material_dir in self.material_dirs:
            calc_dir=os.path.join(material_dir,calc_name)
            multi_task_list.append(calc_dir)

        results=self._process_task(self.generate_job_scheduler_script_for_calc, multi_task_list, slurm_config=slurm_config, slurm_script=slurm_script, **kwargs)
        return results
 
    def submit_job(self, slurm_script_path: str, capture_output=True, text=True):
        """
        Submits a SLURM job using a specified SLURM script path.

        Args:
            slurm_script_path (str): The path to the SLURM script to be submitted.
            capture_output (bool, optional): Whether to capture the output of the SLURM job submission. Defaults to True.
            text (bool, optional): Whether to capture output as text. Defaults to True.

        Returns:
            str: The SLURM job ID if the submission is successful.

        Raises:
            RuntimeError: If the SLURM job submission fails.
        """
        result = subprocess.run(['sbatch', slurm_script_path], capture_output=capture_output, text=text)
        if result.returncode == 0:
            # Extract the SLURM job ID from sbatch output
            slurm_job_id = result.stdout.strip().split()[-1]
            return slurm_job_id
        else:
            raise RuntimeError(f"Failed to submit SLURM job. Error: {result.stderr}")

    def submit_jobs(self, calc_name, ids=None, **kwargs):
        """
        Submits SLURM jobs for all materials or a subset of materials by calculation name.

        Args:
            calc_name (str): The name of the calculation.
            ids (List, optional): A list of material IDs to submit jobs for. Defaults to all materials.
            **kwargs: Additional arguments to pass to the job submission function.

        Returns:
            List: The results of job submission for each material.
        """
        multi_task_list=[]
        if ids is None:
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                job_submission_script_path=os.path.join(calc_dir, self.job_submission_script_name)
                multi_task_list.append(job_submission_script_path)
        else:
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                job_submission_script_path=os.path.join(calc_dir, self.job_submission_script_name)
                multi_task_list.append(job_submission_script_path)

        results=self._process_task(self.generate_job_scheduler_script_for_calc, multi_task_list, **kwargs)
        return results
    
    def run_func_on_calc(self, calc_func:Callable, calc_name: str, material_id: str, **kwargs):
        """
        Runs a specified function on a specific calculation directory. 
        The calc_func expects the calculation directory path as its only argument.

        Args:
            calc_func (Callable): The function to run on the calculation directory.
            calc_name (str): The name of the calculation.
            material_id (str): The ID of the material for which the function will be run.
            **kwargs: Additional arguments to pass to the function.

        Returns:
            None
        """
        calc_dir=os.path.join(self.calculation_dir,material_id,calc_name)
        calc_func(calc_dir,**kwargs)
        return None
    
    def run_func_on_calcs(self, calc_func:Callable, calc_name: str, ids=None, **kwargs):
        """
        Runs a specified function on all calculation directories or a subset of directories.

        Args:
            calc_func (Callable): The function to run on each calculation directory.
            calc_name (str): The name of the calculation.
            ids (List, optional): A list of material IDs. Defaults to running on all directories.
            **kwargs: Additional arguments to pass to the function.

        Returns:
            List: The results of running the function on each calculation directory.
        """
        multi_task_list=[]
        if ids is None:
            i=0
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                multi_task_list.append(calc_dir)
                i+=1
        else:
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                multi_task_list.append(calc_dir)

        results=self._process_task(calc_func, multi_task_list, **kwargs)
        return results
    
    def add_calc_data_to_database(self, func:Callable, calc_name: str, ids=None, **kwargs):
        multi_task_list=[]
        if ids is None:
            ids=[]
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                multi_task_list.append(calc_dir)
                ids.append(os.path.dirname(material_dir))
        else:
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                multi_task_list.append(calc_dir)

        results=self._process_task(func, multi_task_list, **kwargs)

        update_list=[(id,result) for id,result in zip(ids,results)]
        
        self.db_manager.update_many(update_list)


    def load_metadata(self):
        """
        Loads metadata from a JSON file in the main directory.

        Returns:
            Dict: The loaded metadata. If the file does not exist, returns an empty dictionary.
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata
        else:
            return {}
    
    def update_metadata(self, metadata):
        """
        Updates the metadata with new information and saves it to the metadata file.

        Args:
            metadata (Dict): The new metadata to be added or updated.
        """
        self.metadata.update(metadata)
        self.save_metadata(self.metadata)

    def save_metadata(self, metadata):
        """
        Saves the metadata to the metadata JSON file in the main directory.

        Args:
            metadata (Dict): The metadata to save.
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

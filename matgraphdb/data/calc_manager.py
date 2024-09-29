import logging
import os
import json
import inspect

import subprocess
from typing import Callable, Dict, List, Tuple, Union

from matgraphdb.calculations.job_scheduler_generator import SlurmScriptGenerator
from matgraphdb.utils import N_CORES, multiprocess_task

logger = logging.getLogger(__name__)

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

        logger.info(f"Initializing CalculationManager with main directory: {main_dir}")
        logger.debug(f"Calculation directory set to: {self.calculation_dir}")
        logger.debug(f"Metadata file path set to: {self.metadata_file}")
        logger.debug(f"Job submission script name: {self.job_submission_script_name}")
        logger.debug(f"Number of cores for multiprocessing: {self.n_cores}")
        logger.info("Make sure to initialize the calculation manager before using it")
        
    def _setup_material_directory(self, directory):
        """
        Creates the directory structure for a specific material if it doesn't exist.
        
        Parameters:
        directory (str): Path to the material directory.
        
        Returns:
        None
        """
        logger.debug(f"Setting up material directory at: {directory}")
        os.makedirs(directory, exist_ok=True)
        return None

    def _setup_material_directories(self):
        """
        Creates directories for all materials in the database and returns their paths.
        
        Returns:
        list: A list of material directory paths.
        """
        logger.info("Setting up material directories.")
        logger.debug("Reading materials from the database.")
        table=self.db_manager.read(columns=['id'])
        id_df=table.to_pandas()

        material_dirs = []
        for i, row in id_df.iterrows():
            material_id = row['id']
            material_directory = os.path.join(self.calculation_dir, material_id)
            logger.debug(f"Setting up directory for material ID {material_id} at {material_directory}")
            self._setup_material_directory(material_directory)
            material_dirs.append(material_directory)
        return material_dirs
    
    def initialize(self):
        """
        Initializes the CalculationManager by loading metadata and setting up material directories.
        """
        logger.info("Initializing CalculationManager.")
        self.metadata = self.load_metadata()
        logger.debug("Metadata loaded.")
        self.material_dirs = self._setup_material_directories()
        logger.debug("Material directories set up.")
        self.initialized=True
        logger.info("CalculationManager initialization complete.")

    def run_inmemory_calculation(self, calc_func:Callable, save_results=False, verbose=False, read_args=None, **kwargs):
        """
        Runs a calculation on the data retrieved from the MaterialDatabaseManager.

        This method is designed to process data from the material database using a user-defined 
        calculation function. The `calc_func` is expected to be a callable that accepts a 
        dictionary-like object (each row's data) and returns a dictionary representing the results 
        of the calculation. The results can optionally be saved back to the database.

        Args:
            calc_func (Callable): A function that processes each row of data. This function should 
                                accept a dictionary-like object representing the row's data and return 
                                a dictionary containing the calculated results for that row.
            save_results (bool): A flag indicating whether to save the results back to the database 
                                after processing. Defaults to True.
            verbose (bool): A flag indicating whether to print error messages. Defaults to False.
            **kwargs: Additional keyword arguments that are passed to the calculation function.

        Returns:
            list: A list of result dictionaries returned by the `calc_func` for each row in the database.

        Example:
            def my_calc_func(row_data):
                # Perform some calculations on row_data
                return {'result': sum(row_data.values())}
            
            handler.run_calculation(my_calc_func)

        Notes:
            - The data is read from the database using `self.db_manager.read()`.
            - If `save_results` is True, the method will update the database with the results.
            - The `update_many` method is used to update the database by mapping each row's ID 
            to its corresponding calculated result.
        """
        logger.info("Running in-memory calculation on material data.")
        rows=self.db_manager.read(**read_args)
        ids=[]
        data=[]
        for row in rows:
            ids.append(row.id)
            data.append(row.data)
        logger.debug(f"Retrieved {len(rows)} rows from the database.")

        calc_func = calculation_error_handler(calc_func,verbose=verbose)
        results=multiprocess_task(calc_func, data, n_cores=self.n_cores, **kwargs)
        logger.info("Calculation completed.")

        if save_results:
            update_list=[(id,result) for id,result in zip(ids,results)]
            self.db_manager.update_many(update_list)
            logger.info("Results saved back to the database.")

        return results



    def get_calculation_names(self):
        """
        This method returns a list of all calculation names in the database.
        """
        logger.info("Retrieving calculation names.")
        calculation_names = os.listdir(self.material_dirs[0])
        logger.debug(f"Calculation names found: {calculation_names}")
        self.update_metadata({'calculation_names': calculation_names})
        return calculation_names

    def create_calc(self, calc_func: Callable, calc_name: str = None, read_args: dict = None, **kwargs):
        """
        This method creates a new calculation by applying the provided function to each row in the database. 
        The function is expecting an input of type dictionary-like object (each row's data) and a directory path 
        where the calculation results should be stored. The function is responsible for saving the results in this 
        directory, which is specific to each row and named based on the row's unique ID and the name of the 
        calculation function.

        Args:
            calc_func (Callable): The function to apply to each material.
                This function should have arguments that match the names of the columns in the database.
                ```python
                def my_calc_func(material_id: int, formula: str, density: float, lattice: List[List[float]]):
                    # Perform some calculation
                    return {'density': sum(density)}

                manager.create_calc(my_calc_func,'my_calc')
                ```
            calc_name (str, optional): The name of the calculation. Defaults to the name of the function.
            read_args (dict, optional): Additional arguments to pass to the `MaterialDatabaseManager.read` method.
            **kwargs: Additional arguments to pass to the calculation function.

        Returns:
            List: The results of the calculation for each material.
        """
        if read_args is None:
            read_args={}
        if calc_name is None:
            calc_name = calc_func.__name__

        logger.info(f"Creating calculation '{calc_name}' for all materials.")
        # Read data from the database

        signature = inspect.signature(calc_func)
        param_names=list(signature.parameters.keys())
        read_args['columns']=param_names
        read_args['columns'].append('id')

        table = self.db_manager.read(**read_args)
        df=table.to_pandas()
        logger.debug(f"Retrieved {len(table.shape[0])} rows from the database.")

        multi_task_list=[]
        for row in df.iterrows():
            id=row['id']
            row_data = row.drop('id')
            calc_dir=os.path.join(self.material_dirs[id],calc_name)

            multi_task_list.append((row_data,calc_dir))


        logger.info(f"Prepared tasks for {len(multi_task_list)} materials.")
        # Process each row using multiprocessing, passing the directory structure
        logger.debug("Starting calculation tasks.")
        results=multiprocess_task(calc_func, multi_task_list, n_cores=self.n_cores, **kwargs)
        logger.info(f"Calculation '{calc_name}' completed for all materials.")
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

        logger.info(f"Generating SLURM script for calculation '{calc_name}' in material '{material_id}'.")

        if slurm_config:
            logger.debug(f"SLURM config: {slurm_config}")
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
        logger.info(f"SLURM script saved at {slurm_script_path}")

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
        logger.info(f"Generating SLURM scripts for calculation '{calc_name}' for all materials.")
        multi_task_list=[]
        for material_dir in self.material_dirs:
            calc_dir=os.path.join(material_dir,calc_name)
            multi_task_list.append(calc_dir)
        logger.debug(f"Prepared SLURM script generation tasks for {len(multi_task_list)} materials.")

        results=multiprocess_task(self.generate_job_scheduler_script_for_calc, multi_task_list, n_cores=self.n_cores, slurm_config=slurm_config, slurm_script=slurm_script, **kwargs)
        logger.info("SLURM script generation completed for all materials.")
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
        logger.info(f"Submitting SLURM job with script: {slurm_script_path}")
        result = subprocess.run(['sbatch', slurm_script_path], capture_output=capture_output, text=text)
        if result.returncode == 0:
            # Extract the SLURM job ID from sbatch output
            slurm_job_id = result.stdout.strip().split()[-1]
            logger.info(f"SLURM job submitted successfully. Job ID: {slurm_job_id}")
            return slurm_job_id
        else:
            logger.error(f"Failed to submit SLURM job with script {slurm_script_path}. Error: {result.stderr}")
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
        logger.info(f"Submitting SLURM jobs for calculation '{calc_name}'")
        multi_task_list=[]
        if ids is None:
            logger.debug("No specific IDs provided, submitting jobs for all materials.")
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                job_submission_script_path=os.path.join(calc_dir, self.job_submission_script_name)
                multi_task_list.append(job_submission_script_path)
        else:
            logger.debug(f"Submitting jobs for material IDs: {ids}")
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                job_submission_script_path=os.path.join(calc_dir, self.job_submission_script_name)
                multi_task_list.append(job_submission_script_path)
        logger.debug(f"Prepared job submission scripts for {len(multi_task_list)} materials.")
        results=multiprocess_task(self.generate_job_scheduler_script_for_calc, multi_task_list, n_cores=self.n_cores, **kwargs)
        logger.info("Job submissions completed.")
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
        logger.info(f"Running function '{calc_func.__name__}' on calculation '{calc_name}' for material ID '{material_id}'.")
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
        logger.info(f"Running function '{calc_func.__name__}' on calculation '{calc_name}' for multiple materials.")
        multi_task_list=[]
        if ids is None:
            logger.debug("No specific IDs provided, running function on all materials.")
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                multi_task_list.append(calc_dir)

        else:
            logger.debug(f"Running function on material IDs: {ids}")
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                multi_task_list.append(calc_dir)
        logger.debug(f"Prepared function tasks for {len(multi_task_list)} materials.")
        results=multiprocess_task(calc_func, multi_task_list, n_cores=self.n_cores, **kwargs)
        logger.info(f"Function '{calc_func.__name__}' completed for all specified materials.")
        return results
    
    def add_calc_data_to_database(self, func:Callable, calc_name: str, ids=None, update_args: dict = None, **kwargs):
        logger.info(f"Adding calculation data to database for calculation '{calc_name}'.")
        multi_task_list=[]
        if ids is None:
            ids=[]
            for material_dir in self.material_dirs:
                calc_dir=os.path.join(material_dir,calc_name)
                multi_task_list.append(calc_dir)
                ids.append(os.path.dirname(material_dir))
        else:
            logger.debug(f"Processing material IDs: {ids}")
            for id in ids:
                calc_dir=os.path.join(self.calculation_dir,id,calc_name)
                multi_task_list.append(calc_dir)
        logger.info("Calculation data processing completed.")
        results=multiprocess_task(func, multi_task_list, n_cores=self.n_cores, **kwargs)

        update_list=[(id,result) for id,result in zip(ids,results)]
        update_data=[]
        
        for id, result in zip(ids,results):
            update_dict={}
            update_dict.update(result)
            update_dict['id']=id
            update_data.append(update_dict)


        logger.debug(f"Updating database with results.")
        
        self.db_manager.update(update_list, **update_args)
        logger.info("Database updated with calculation data.")

    def load_metadata(self):
        """
        Loads metadata from a JSON file in the main directory.

        Returns:
            Dict: The loaded metadata. If the file does not exist, returns an empty dictionary.
        """
        logger.info("Loading metadata.")
        if os.path.exists(self.metadata_file):
            logger.debug(f"Metadata file found at {self.metadata_file}")
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata
        else:
            logger.warning(f"Metadata file not found at {self.metadata_file}, returning empty metadata.")
            return {}
    
    def update_metadata(self, metadata):
        """
        Updates the metadata with new information and saves it to the metadata file.

        Args:
            metadata (Dict): The new metadata to be added or updated.
        """
        logger.info("Updating metadata.")
        logger.debug(f"New metadata to update: {metadata}")
        self.metadata.update(metadata)
        self.save_metadata(self.metadata)

    def save_metadata(self, metadata):
        """
        Saves the metadata to the metadata JSON file in the main directory.

        Args:
            metadata (Dict): The metadata to save.
        """
        logger.info(f"Saving metadata to {self.metadata_file}.")
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.debug("Metadata saved successfully.")



def calculation_error_handler(calc_func: Callable):
    """
    A decorator that wraps the user-defined calculation function in a try-except block.
    This allows graceful handling of any errors that may occur during the calculation process.
    
    Args:
        calc_func (Callable): The user-defined calculation function to be wrapped.
    
    Returns:
        Callable: A wrapped function that executes the calculation and handles any exceptions.
    """
    def wrapper(data):
        try:
            # Call the user-defined calculation function
            return calc_func(data)
        except Exception as e:
            # Log the error (you could customize this as needed)
            logger.error(f"Error in calculation function: {e}\nData: {data}")
            # Optionally, you could return a default or partial result, or propagate the error
            return {}
    
    return wrapper

def disk_calculation(disk_calc_func: Callable, base_directory: str):
    """
    A decorator that wraps the disk-based calculation function in a try-except block.
    It also ensures that the specified directory for each row (ID) and calculation name 
    is created before the calculation function is executed.

    Args:
        disk_calc_func (Callable): The user-defined disk-based calculation function to be wrapped.
        base_directory (str): The base directory where files related to the calculation will be stored.
        verbose (bool): If True, prints error messages. Defaults to False.

    Returns:
        Callable: A wrapped function that handles errors and ensures the calculation directory exists.
    """

    def wrapper(row_data, row_id, calculation_name):
        # Define the directory for this specific row (ID) and calculation
        material_specific_directory = os.path.join('MaterialsData',row_id,calculation_name)
        calculation_directory = os.path.join(base_directory, material_specific_directory)
        os.makedirs(calculation_directory, exist_ok=True)

        try:
            # Call the user-defined calculation function, passing the directory as a parameter
            return disk_calc_func(row_data, directory=calculation_directory)
        except Exception as e:
            # Log the error
            logger.error(f"Error in disk calculation function: {e}\nData: {row_data}")
            # Optionally, return a default or partial result, or propagate the error
            return {}

    return wrapper

import os
import json
from glob import glob
from typing import Dict, List, Tuple
from multiprocessing import Pool

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES

from functools import partial


class DatabaseManager:
    def __init__(self, directory_path=DB_DIR, calc_path=DB_CALC_DIR, n_cores=N_CORES):
        self.directory_path = directory_path
        self.calculation_path = calc_path
        self.n_cores = N_CORES

    @property
    def database_files(self):
        return glob(self.directory_path + os.sep + '*.json')

    def process_task(self, func, list,**kwargs):
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
    
    def check_property_task(self, file, property_name=''):
        """Check if a given property exists in the data."""

        data=self.load_json(file)

        check=True
        if property_name in data:
            if data[property_name] is None:
                check=False
            else:
                check=True
        else:
            check=False

        return check
    
    def check_property(self, property_name):
        """Check if a given property exists in all JSON files and categorize them."""
        
        database_files = glob(self.directory_path + os.sep + '*.json')
        print("Processing files from : ",self.directory_path + os.sep + '*.json')
        results=self.process_task(self.check_property_task, database_files, property_name=property_name)

        success = []
        failed = []
        for file,result in zip(database_files,results):
            if result==True:
                success.append(file)
            else:
                failed.append(file)

        return success, failed

    def check_chargemol_task(self, dir):
        """Check if a given property exists in the data."""
        check=True

        file_path = os.path.join(dir,'chargemol','DDEC6_even_tempered_bond_orders.xyz')
        
        if os.path.exists(file_path):
            check=True
        else:
            check=False

        return check
    
    def check_chargemol(self):
        """Check if a given property exists in all JSON files and categorize them."""
        
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
    # success,failed=db.check_property(property_name=properties[0])


    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1') )

    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1'),exclude=[] )
    success,failed=db.check_chargemol()
    # print(success[:10])
    print(failed[:20])

    print("Number of failed files: ", len(failed))
    print("Number of success files: ", len(success))
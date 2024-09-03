import os
import json
from glob import glob
import subprocess
from typing import Dict, List, Tuple, Union
from multiprocessing import Pool
from functools import partial

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints

from matgraphdb import DBManager
from matgraphdb.utils import get_logger

logger=get_logger(__name__, console_out=False, log_level='info')


class CalculationManager:
    def __init__(self, db_manager: DBManager):
        """
        Initializes the Manager object.

        Args:
            directory_path (str): The path to the directory where the database is stored.
            calc_path (str): The path to the directory where calculations are stored.
            n_cores (int): The number of CPU cores to be used for parallel processing.
        """
        self.db_manager = db_manager
        self.calculation_path = self.db_manager.calculation_path    

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
        calc_dirs = self.db_manager.calculation_dirs()
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

    @staticmethod
    def launch_calcs(slurm_scripts=[]):
        """
        Launches SLURM calculations by submitting SLURM scripts.

        Args:
            slurm_scripts (list): A list of SLURM script file paths.

        Returns:
            None
        """
        if slurm_scripts != []:
            for slurm_script in slurm_scripts:
                result = subprocess.run(['sbatch', slurm_script], capture_output=False, text=True)





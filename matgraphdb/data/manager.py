import os
import json
from glob import glob
from typing import Dict, List, Tuple
from multiprocessing import Pool
from functools import partial

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES,LOGGER
from matgraphdb.calculations.parsers import parse_chargemol_bond_orders,parse_chargemol_net_atomic_charges, parse_chargemol_atomic_moments, parse_chargemol_overlap_populations



class DBManager:
    def __init__(self, directory_path=DB_DIR, calc_path=DB_CALC_DIR, n_cores=N_CORES):
        self.directory_path = directory_path
        self.calculation_path = calc_path
        self.n_cores = N_CORES

    def database_files(self):
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
    
    def check_property_task(self, file, property_name=''):
        """Check if a given property exists in the data."""

        data=self.load_json(file)

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
        calc_dirs = glob(self.calculation_path + os.sep + 'mp-*')
        print("Processing files from : ",self.calculation_path + os.sep + 'mp-*')
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

        return None
        
    def collect_chargemol_info(self):
        LOGGER.info(f"Starting collection Chargemol information")
        calc_dirs = self.calculation_dirs()
        self.process_task(self.chargemol_task, calc_dirs)
        LOGGER.info(f"Finished collection Chargemol information")


            

if __name__=='__main__':

    properties=['chargemol_bonding_orders','coordination_environments_multi_weight']

    db=DBManager()
    success,failed=db.check_property(property_name=properties[0])
    print("Number of failed files: ", len(failed))
    print("Number of success files: ", len(success))

    # db.chargemol_task(dir=db.calculation_dirs()[0])
    # print(N_CORES)
    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00','20', '1') )

    # db.add_chargemol_slurm_script(partition_info=('comm_small_day','24:00:00'),exclude=[] )
    # success,failed=db.check_chargemol()
    # # print(success[:10])
    # print(failed[:20])
    # print("Number of failed files: ", len(failed))
    # print("Number of success files: ", len(success))
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
from matgraphdb.calculations.job_scheduler_generator import SlurmScriptGenerator
from matgraphdb.utils import get_logger

logger=get_logger(__name__, console_out=False, log_level='info')

class CalculationGenerator:
    def __init__(self, calc_dir):
        self.calc_dir=calc_dir
        os.makedirs(self.calc_dir,exist_ok=True)

    def create_job_scheduler_script(self):
        raise NotImplementedError("get_estimated_computaional_resources must be implemented in the child class")
    
    def write_job(self):
        raise NotImplementedError("write_job must be implemented in the child class")
    

class VaspCalcGenerator(CalculationGenerator):
    def __init__(self, species, frac_coords, lattice, calc_dir, vasp_pseudos_dir):
        super().__init__(calc_dir)
        self.vasp_pseudos_dir=vasp_pseudos_dir

        self.structure=None
        self.incar_args={}

        self.incar_str=''
        self.poscar_str=''
        self.potcar_str=''
        self.kpoints_str=''
        self.job_scheduler_str=''

        self._add_structure(species=species, frac_coords=frac_coords, lattice=lattice)

    def _add_structure(self, species, frac_coords, lattice,):
        self.structure=Structure(lattice=lattice, species=species, coords=frac_coords)
        self.composition=self.structure.composition
        self.elements=list(self.composition.as_dict().keys())
        self.n_atoms=len(species)
        return None
    
    def create_incar(self, incar_args={}):
        if self.job_scheduler_str is None:
            raise ValueError("Call create_job_scheduler_script first")
        for key,value in incar_args.items():
            self.incar_str+=f'{key} = {value}\n'

    def create_poscar(self):
        self.poscar_str=self.structure.to(fmt='poscar')

    def create_potcar(self, pseudo_type='potpaw_PBE.52', potcar_str=None):
        if potcar_str is not None:
            self.potcar_str=potcar_str
            return None
        
        # Create default POTCAR files
        pseudos_dir=os.path.join(self.vasp_pseudos_dir,pseudo_type)
        tmp_potcar=''
        for symbol in self.elements:
            pseudo_file=os.path.join(pseudos_dir,symbol,'POTCAR')

            if not os.path.exists(pseudos_dir):
                pseudo_file=os.path.join(pseudos_dir,symbol+'_sv','POTCAR')
            # open pseudo file and add it to potcar string
            if symbol=='Zr_sv':
                with open(pseudo_file,'r') as f:
                    lines=f.readlines()
                    new_line=lines[3].replace('r','Zr')
                    lines[3]=new_line
                    tmp_text=''.join(lines)
                    tmp_potcar+=tmp_text
                    tmp_potcar+=''
            else:
                with open(pseudo_file,'r') as f:
                    tmp_potcar+=f.read()
                    tmp_potcar+=''
        self.potcar_str=tmp_potcar
        return None

    def create_kpoints(self,kpoints:Kpoints):
        """
        Create a KPOINTS file from a Kpoints object. Refer to their documentation for more information.
        https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.inputs.Kpoints

        Args:
            kpoints (Kpoints): A Kpoints object.
        """
        self.kpoints_str=str(kpoints)
        print(self.kpoints_str)
        return None

    def create_job_scheduler_script(self,slurm_script_body=None):
        if self.incar_str=='':
            raise ValueError("Incar must be created first. Call create_incar")
        if self.poscar_str=='':
            raise ValueError("Poscar must be created first. Call create_poscar")
        if self.potcar_str=='':
            raise ValueError("Potcar must be created first. Call create_potcar")
        if self.kpoints_str=='':
            raise ValueError("Kpoints must be created first. Call create_kpoints")
        if self.n_atoms >= 60:  
            nnode=4
            ncore = 32  
            kpar=4   
            ntasks=160
        elif self.n_atoms >= 40:  
            nnode=3
            ncore = 20  
            kpar=3 
            ntasks=120
        elif self.n_atoms >= 20: 
            nnode=2
            ncore = 16  
            kpar = 2   
            ntasks=80
        else:
            nnode=1
            ntasks=40
            ncore = 40 
            kpar = 1

        self.job_scheduler_script_generator=SlurmScriptGenerator()
        self.job_scheduler_script_generator.init_header()
        self.job_scheduler_script_generator.add_slurm_header_comp_resources(n_nodes=nnode, n_tasks=ntasks)
        if slurm_script_body is None:
            slurm_script_body="\n"
            'source ~/.bashrc\n'
            'module load atomistic/vasp/6.2.1_intel22_impi22\n'
            f'cd {self.calc_dir}\n'
            f'echo "CALC_DIR: {self.calc_dir}"\n'
            f'echo "NCORES: $((SLURM_NTASKS))"\n'
            '\n'
            f'mpirun -np $SLURM_NTASKS vasp_std\n'


            
        self.job_scheduler_script_generator.add_slurm_script_body(slurm_script_body)
        self.job_scheduler_str=self.job_scheduler_script_generator.finalize()

        self.incar_args['NCORE']=ncore
        self.incar_args['KPAR']=kpar
        
        return None

    def write_job(self):
        if self.structure is None:
            raise ValueError("Structure must be added first")
        
        if self.incar_str=='':
            raise ValueError("Incar must be created first. Call create_incar")
        if self.poscar_str=='':
            raise ValueError("Poscar must be created first. Call create_poscar")
        if self.potcar_str=='':
            raise ValueError("Potcar must be created first. Call create_potcar")
        if self.kpoints_str=='':
            raise ValueError("Kpoints must be created first. Call create_kpoints")
        if self.job_scheduler_str=='':
            raise ValueError("Job scheduler script must be created first. Call create_job_scheduler_script")
        

        with open(os.path.join(self.calc_dir,'INCAR'),'w') as incar:
            incar.write(self.incar_str)

        with open(os.path.join(self.calc_dir,'POSCAR'),'w') as poscar:
            poscar.write(self.poscar_str)

        with open(os.path.join(self.calc_dir,'POTCAR'),'w') as potcar:
            potcar.write(self.potcar_str)

        with open(os.path.join(self.calc_dir,'KPOINTS'),'w') as kpoints:
            kpoints.write(self.kpoints_str)

        with open(os.path.join(self.calc_dir,'job_submit.run'),'w') as job_control:
            job_control.write(self.job_scheduler_str)
        

class ChargemolCalcGenerator(CalculationGenerator):
    def __init__(self, species, frac_coords, lattice, calc_dir, vasp_pseudos_dir):
        super().__init__(calc_dir)
        self.vasp_pseudos_dir=vasp_pseudos_dir

        self.structure=None
        self.incar_args={}

        self.incar_str=''
        self.poscar_str=''
        self.potcar_str=''
        self.kpoints_str=''
        self.job_scheduler_str=''
        self.job_control_str=''

        self._add_structure(species=species, frac_coords=frac_coords, lattice=lattice)

    def _add_structure(self, species, frac_coords, lattice,):
        self.structure=Structure(lattice=lattice, species=species, coords=frac_coords)
        self.composition=self.structure.composition
        self.elements=list(self.composition.as_dict().keys())
        self.n_atoms=len(species)
        return None
    
    def create_incar(self, incar_args={}):
        if self.job_scheduler_str is None:
            raise ValueError("Call create_job_scheduler_script first")
        for key,value in incar_args.items():
            self.incar_str+=f'{key} = {value}\n'

    def create_poscar(self):
        self.poscar_str=self.structure.to(fmt='poscar')

    def create_potcar(self, pseudo_type='potpaw_PBE.52', potcar_str=None):
        if potcar_str is not None:
            self.potcar_str=potcar_str
            return None
        
        # Create default POTCAR files
        pseudos_dir=os.path.join(self.vasp_pseudos_dir,pseudo_type)
        tmp_potcar=''
        for symbol in self.elements:
            pseudo_file=os.path.join(pseudos_dir,symbol,'POTCAR')

            if not os.path.exists(pseudos_dir):
                pseudo_file=os.path.join(pseudos_dir,symbol+'_sv','POTCAR')
            # open pseudo file and add it to potcar string
            if symbol=='Zr_sv':
                with open(pseudo_file,'r') as f:
                    lines=f.readlines()
                    new_line=lines[3].replace('r','Zr')
                    lines[3]=new_line
                    tmp_text=''.join(lines)
                    tmp_potcar+=tmp_text
                    tmp_potcar+=''
            else:
                with open(pseudo_file,'r') as f:
                    tmp_potcar+=f.read()
                    tmp_potcar+=''
        self.potcar_str=tmp_potcar
        return None

    def create_kpoints(self,kpoints:Kpoints):
        """
        Create a KPOINTS file from a Kpoints object. Refer to their documentation for more information.
        https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.inputs.Kpoints

        Args:
            kpoints (Kpoints): A Kpoints object.
        """
        self.kpoints_str=str(kpoints)
        print(self.kpoints_str)
        return None

    def create_job_scheduler_script(self,slurm_script_body=None):
        if self.incar_str=='':
            raise ValueError("Incar must be created first. Call create_incar")
        if self.poscar_str=='':
            raise ValueError("Poscar must be created first. Call create_poscar")
        if self.potcar_str=='':
            raise ValueError("Potcar must be created first. Call create_potcar")
        if self.kpoints_str=='':
            raise ValueError("Kpoints must be created first. Call create_kpoints")
        if self.n_atoms >= 60:  
            nnode=4
            ncore = 32  
            kpar=4   
            ntasks=160
        elif self.n_atoms >= 40:  
            nnode=3
            ncore = 20  
            kpar=3 
            ntasks=120
        elif self.n_atoms >= 20: 
            nnode=2
            ncore = 16  
            kpar = 2   
            ntasks=80
        else:
            nnode=1
            ntasks=40
            ncore = 40 
            kpar = 1

        self.job_scheduler_script_generator=SlurmScriptGenerator()
        self.job_scheduler_script_generator.init_header()
        self.job_scheduler_script_generator.add_slurm_header_comp_resources(n_nodes=nnode, n_tasks=ntasks)
        

        if slurm_script_body is None:
            slurm_script_body=("\n"
            'source ~/.bashrc\n'
            'module load atomistic/vasp/6.2.1_intel22_impi22\n'
            f'cd {self.calc_dir}\n'
            f'echo "CALC_DIR: {self.calc_dir}"\n'
            f'echo "NCORES: $((SLURM_NTASKS))"\n'
            '\n'
            f'mpirun -np $SLURM_NTASKS vasp_std\n'
            '\n'
            f'export OMP_NUM_THREADS=$SLURM_NTASKS\n'
            '~/SCRATCH/Codes/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries'
            '/linux/Chargemol_09_26_2017_linux_parallel> chargemol_debug.txt 2>&1\n'
            '\n'
            f'echo "run complete on `hostname`: `date`" 1>&2\n')

        self.job_scheduler_script_generator.add_slurm_script_body(slurm_script_body)
        self.job_scheduler_str=self.job_scheduler_script_generator.finalize()

        self.incar_args['NCORE']=ncore
        self.incar_args['KPAR']=kpar
        
        return None

    def create_job_control_script(self,atomic_densities_dir):
        job_control_script=("<atomic densities directory complete path>\n"+
        atomic_densities_dir+ '\n'
        "</atomic densities directory complete path>\n\n"

        "<charge type>\n"
        "DDEC6\n"
        "</charge type>\n\n"

        "<compute BOs>\n"
        ".true.\n"
        "</compute BOs>\n")

        self.job_control_str=job_control_script
        return job_control_script


    def write_job(self):   
        if self.incar_str=='':
            raise ValueError("Incar must be created first. Call create_incar")
        if self.poscar_str=='':
            raise ValueError("Poscar must be created first. Call create_poscar")
        if self.potcar_str=='':
            raise ValueError("Potcar must be created first. Call create_potcar")
        if self.kpoints_str=='':
            raise ValueError("Kpoints must be created first. Call create_kpoints")
        if self.job_scheduler_str=='':
            raise ValueError("Job scheduler script must be created first. Call create_job_scheduler_script")
        

        with open(os.path.join(self.calc_dir,'INCAR'),'w') as incar:
            incar.write(self.incar_str)

        with open(os.path.join(self.calc_dir,'POSCAR'),'w') as poscar:
            poscar.write(self.poscar_str)

        with open(os.path.join(self.calc_dir,'POTCAR'),'w') as potcar:
            potcar.write(self.potcar_str)

        with open(os.path.join(self.calc_dir,'KPOINTS'),'w') as kpoints:
            kpoints.write(self.kpoints_str)

        with open(os.path.join(self.calc_dir,'job_submit.run'),'w') as job_control:
            job_control.write(self.job_scheduler_str)

        with open(os.path.join(self.calc_dir,'job_control.txt'),'w') as job_control:
            job_control.write(self.job_control_str)




if __name__=='__main__':
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    # calc_manager=CalculationManager()
    pseudos_dir=os.path.join('data','PP_Vasp')

    materials_parquet=os.path.join('data','production','materials_project','materials_database.parquet')
    data_dir=os.path.join('data','raw','test_dir')

    
    

    table = pq.read_table(materials_parquet, columns=['material_id','lattice','frac_coords','species'])
    df = table.to_pandas()
    index=1
    lattice=np.array(list(df.iloc[index]['lattice']))
    frac_coords=np.array(list(df.iloc[index]['frac_coords']))
    species=list(df.iloc[index]['species'])

    print(df.head())
    # generator=VaspCalcGenerator(species=species, frac_coords=frac_coords, lattice=lattice,
    #                             calc_dir=os.path.join(data_dir,'mp-1000'),
    #                             vasp_pseudos_dir=pseudos_dir)


    # kpoints=Kpoints.gamma_automatic(kpts=(9,9,9), shift=(0,0,0))
    # generator.create_kpoints(kpoints=kpoints)

    # vasp_parameters = {
    #     "EDIFF": 1e-08,
    #     "ENCUT": 600,
    #     "IBRION": -1,
    #     "ISMEAR": -5,
    #     "ISPIN": 1,
    #     "ISTART": 0,
    #     "LASPH": True,
    #     "LMAXMIN": 4,
    #     "LORBIT": 11,
    #     "LREAL": False,
    #     "NELM": 100,
    #     "NELMIN": 8,
    #     "NSIM": 2,
    #     "NSW": 0,
    #     "LCHARG": True,
    #     "LAECHG": True,
    #     "LWAVE": False,
    #     "PREC": "Accurate",
    #     "SIGMA": 0.01,
    #     "NWRITE": 3
    # }

    # generator.create_incar(incar_args=vasp_parameters)

    # generator.create_poscar()
    # generator.create_potcar()
    # generator.create_job_scheduler_script()

    # generator.write_job()



    generator=ChargemolCalcGenerator(species=species, frac_coords=frac_coords, lattice=lattice,
                                calc_dir=os.path.join(data_dir,'mp-1000','chargemol'),
                                vasp_pseudos_dir=pseudos_dir)


    kpoints=Kpoints.gamma_automatic(kpts=(9,9,9), shift=(0,0,0))
    generator.create_kpoints(kpoints=kpoints)

    vasp_parameters = {
        "EDIFF": 1e-08,
        "ENCUT": 600,
        "IBRION": -1,
        "ISMEAR": -5,
        "ISPIN": 1,
        "ISTART": 0,
        "LASPH": True,
        "LMAXMIN": 4,
        "LORBIT": 11,
        "LREAL": False,
        "NELM": 100,
        "NELMIN": 8,
        "NSIM": 2,
        "NSW": 0,
        "LCHARG": True,
        "LAECHG": True,
        "LWAVE": False,
        "PREC": "Accurate",
        "SIGMA": 0.01,
        "NWRITE": 3
    }

    generator.create_incar(incar_args=vasp_parameters)

    generator.create_poscar()
    generator.create_potcar()

    atomic_densities_dir="/users/lllang/SCRATCH/Codes/chargemol_09_26_2017/atomic_densities/"
    generator.create_job_control_script(atomic_densities_dir=atomic_densities_dir)
    generator.create_job_scheduler_script()

    generator.write_job()

    
    




import os 
import json
import shutil
from glob import glob

from matgraphdb.utils import MP_DIR,DB_CALC_DIR


def generate_batch_script(calc_dir,chargemol_file_dir):
    if os.path.exists(os.path.join(calc_dir,'run.slurm')):
        os.remove(os.path.join(calc_dir,'run.slurm'))

    shutil.copy(os.path.join(chargemol_file_dir,'run.slurm'), os.path.join(calc_dir,'run.slurm'))
 
def generate_incar(calc_dir,chargemol_file_dir):
    template_file=os.path.join(chargemol_file_dir,'INCAR')
    file=os.path.join(calc_dir,'INCAR')

    shutil.copy(template_file, file)

def generate_kpoints(calc_dir,chargemol_file_dir):
    template_file=os.path.join(chargemol_file_dir,'KPOINTS')
    file=os.path.join(calc_dir,'KPOINTS')

    shutil.copy(template_file, file)

def generate_job_control(calc_dir,chargemol_file_dir):
    template_file=os.path.join(chargemol_file_dir,'job_control.txt')
    file=os.path.join(calc_dir,'job_control.txt')
    
    shutil.copy(template_file, file)

def generate_potcar(calc_dir,potcar_dir):
    template_file=os.path.join(potcar_dir,'POTCAR_PBE')
    file=os.path.join(calc_dir,'POTCAR')
    
    shutil.copy(template_file, file)

def generate_poscar(calc_dir,poscar):
    template_file=poscar
    file=os.path.join(calc_dir,'POSCAR')
    
    shutil.copy(template_file, file)


def chargemol_calc_setup():
    chargemol_file_dir=os.path.join(MP_DIR,'calculations','calculation_files','chargemol')
    calc_dirs=glob(DB_CALC_DIR + '/*') 

    # print(calc_dirs[:10])
    # chargemol
    for calc_dir in calc_dirs:
        # Create the calculation directory
        potcar_dir=os.path.join(calc_dir,'potcar')
        chargemol_dir=os.path.join(calc_dir,'chargemol')
        poscar=os.path.join(calc_dir,'POSCAR')
        os.makedirs(chargemol_dir,exist_ok=True)

        generate_incar(chargemol_dir,chargemol_file_dir)
        generate_kpoints(chargemol_dir,chargemol_file_dir)
        generate_potcar(chargemol_dir,potcar_dir)
        generate_poscar(chargemol_dir,poscar)
        generate_job_control(chargemol_dir,chargemol_file_dir)
        generate_batch_script(chargemol_dir,chargemol_file_dir)

        


    

if __name__=='__main__':
    chargemol_calc_setup()
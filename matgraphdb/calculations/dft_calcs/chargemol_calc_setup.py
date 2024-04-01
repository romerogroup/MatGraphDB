import os 
import json
import shutil
from glob import glob

from matgraphdb.utils import MP_DIR,DB_CALC_DIR

def generate_batch_scripts(calc_dirs,calc_file_dir):
    for calc_dir in calc_dirs[:]:
        scf_dir=os.path.join(calc_dir,'static')

        if os.path.exists(os.path.join(scf_dir,'run.slurm')):
            os.remove(os.path.join(scf_dir,'run.slurm'))

        try:
            shutil.copy(os.path.join(calc_file_dir,'run.slurm'), os.path.join(scf_dir,'run.slurm'))
        except:
            pass

def generate_potcar(calc_dirs,pseudos_dir):

    for calc_dir in calc_dirs[:]:
        scf_dir=os.path.join(calc_dir,'static')
        potcar_file=os.path.join(scf_dir,'POTCAR')
        incomplete_dir=os.path.dirname(os.path.dirname(calc_dir))
        incomplete_dir=os.path.join(incomplete_dir,'incomplete_database')

        # # Remove pre-existing potcar file
        if os.path.exists(potcar_file):
            os.remove(potcar_file)

        try:
            with open(os.path.join(scf_dir,'POTCAR_files.json'),'r') as f:
                data = json.load(f)
            functional=data['functional']
            symbols=data['symbols']
            tmp_potcar=''

            # Loop through element symbols
            for symbol in symbols:
                pseudo_file=os.path.join(pseudos_dir,symbol,'POTCAR')

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

            # Save file in new POTCAR file
            with open(potcar_file,'w') as potcar:
                potcar.write(tmp_potcar)

        except Exception as e:
            shutil.move(calc_dir, incomplete_dir)
            print(e)
            pass

def generate_incar(calc_dirs,calc_file_dir):
    template_incar_file=os.path.join(calc_file_dir,'INCAR')

    for calc_dir in calc_dirs[:]:
        scf_dir=os.path.join(calc_dir,'static')
        incar_file=os.path.join(scf_dir,'INCAR')

        
        # Rename existing INCAR file to INCAR_old
        incar_old_file=os.path.join(scf_dir,'INCAR_old')
        if os.path.exists(incar_file):
            os.rename(incar_file, incar_old_file)

        # Copy template INCAR file to calc dir
        try:
            shutil.copy(template_incar_file, incar_file)
        except:
            pass
    

def generate_kpoints(calc_dirs,calc_file_dir):
    template_incar_file=os.path.join(calc_file_dir,'KPOINTS')

    for calc_dir in calc_dirs[:]:
        scf_dir=os.path.join(calc_dir,'static')
        incar_file=os.path.join(scf_dir,'KPOINTS')

        # Rename existing INCAR file to INCAR_old
        incar_old_file=os.path.join(scf_dir,'KPOINTS_old')
        if os.path.exists(incar_file):
            os.rename(incar_file, incar_old_file)

        # Copy template INCAR file to calc dir
        try:
            shutil.copy(template_incar_file, incar_file)
        except:
            pass

def generate_job_control(calc_dirs,calc_file_dir):
    template_file=os.path.join(calc_file_dir,'job_control.txt')

    for calc_dir in calc_dirs[:]:
        scf_dir=os.path.join(calc_dir,'static')
        copy_file=os.path.join(scf_dir,'job_control.txt')

        # # Remove pre-existing potcar file
        if os.path.exists(copy_file):
            os.remove(copy_file)

        try:
            shutil.copy(template_file, copy_file)
        except:
            pass

        

def chargemol_calc_setup():
    calc_file_dir=os.path.join(MP_DIR,'calculations','calculation_files','chargemol')
    calc_dirs=glob(DB_CALC_DIR + '/mp-*') 

    pseudos_dir=os.path.join("/users/lllang/SCRATCH",'PP_Vasp','potpaw_PBE.52')

    calc_dir=os.path.join(MP_DIR,'calculations','database')
    print(len(os.listdir(calc_dir)))
    # generate_potcar(calc_dirs,pseudos_dir)

    # generate_incar(calc_dirs,calc_file_dir)

    # generate_batch_scripts(calc_dirs,calc_file_dir)

    # generate_kpoints(calc_dirs,calc_file_dir)

    # generate_job_control(calc_dirs,calc_file_dir)
    

if __name__=='__main__':
    chargemol_calc_setup()
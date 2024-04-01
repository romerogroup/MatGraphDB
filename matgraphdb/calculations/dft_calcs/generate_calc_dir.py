

import os
from glob import glob
import json

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from matgraphdb.utils import DB_DIR, DB_CALC_DIR
from matgraphdb.data.utils import process_database

def generate_calc_dir_task(file):
    with open(file, 'r') as f:
        data = json.load(f)

        # Get mpid name
        mpid = data['material_id']

        # Create the Structure object
        structure = Structure.from_dict(data['structure'])

    # Create calc directory
    calc_dir=os.path.join(DB_CALC_DIR,mpid)
    os.makedirs(calc_dir, exist_ok=True)

    
    # Write the structure to a POSCAR file
    poscar_file=os.path.join(calc_dir,'POSCAR')
    poscar = Poscar(structure)
    poscar.write_file(poscar_file)

    return None

def generate_calc_dir():
    results=process_database(generate_calc_dir_task)



def generate_potcars_task(file):
    mpid=file.split(os.sep)[-1].split('.')[0]

    # Create calc directory
    calc_dir=os.path.join(DB_CALC_DIR,mpid)
    poscar_file=os.path.join(calc_dir,'POSCAR')

    # Create local potcar directory to store POTCAR. 
    #This is if we have to switch pseudopotentials in future
    potcar_dir=os.path.join(calc_dir,'potcar')
    os.makedirs(potcar_dir, exist_ok=True)
    
    # Get element symbols from POSCAR
    with open(poscar_file) as f:
        lines=f.readlines()
        elements=lines[5].split()

    # Create POTCAR files
    pseudos_dir=os.path.join("/users/lllang/SCRATCH",'PP_Vasp','potpaw_PBE.52')
    tmp_potcar=''
    for symbol in elements:
        pseudo_file=os.path.join(pseudos_dir,symbol,'POTCAR')

        if not os.path.exists(pseudo_file):
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

    # Save file in new POTCAR file
    potcar_file=os.path.join(potcar_dir,'POTCAR_PBE')
    with open(potcar_file,'w') as potcar:
        potcar.write(tmp_potcar)

    return None

def generate_potcars():
    results=process_database(generate_potcars_task)


def main():

    database_files=glob(DB_DIR + '/*.json')

    print('#'*100)
    print('Generating Calculation dir')
    print('#'*100)

    generate_calc_dir()


    print('#'*100)
    print('Generating potcar dir')
    print('#'*100)


    calc_dirs=glob(DB_CALC_DIR + '/*')
    print(calc_dirs[:10])
    generate_potcars()


if __name__=='__main__':
    main()

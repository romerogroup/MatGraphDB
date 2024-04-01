
import os 
import json
import re
import shutil
from glob import glob

scratch_dir='/users/lllang/SCRATCH'
root_dir=os.path.join(scratch_dir,'projects','crystal_graph') 
database_dir=os.path.join(root_dir,'data','raw','mp_database_calcs_no_restriction')
calc_dirs=glob(database_dir + '/mp-*')


for calc_dir in calc_dirs[:]:
    print(calc_dir)
    scf_dir=os.path.join(calc_dir,'static')
    wavecar_file=os.path.join(scf_dir,'WAVECAR')

    try:
    
        os.remove(wavecar_file)
    except:
        pass
import os 
import json
import shutil

from glob import glob

from poly_graphs_lib.database.json import MP_DIR,DB_CALC_DIR

from poly_graphs_lib.utils import LOG_DIR


def check_chargmol_calcs():
    calc_file_dir=os.path.join(MP_DIR,'calculations','calculation_files','chargemol')
    calc_dirs=glob(DB_CALC_DIR + '/mp-*') 

    failed_calcs=[]
    for calc_dir in calc_dirs:

        mpid=calc_dir.split(os.sep)[-1]
        scf_dir=os.path.join(calc_dir,'static')

        bond_order_file=os.path.join(scf_dir,'DDEC6_even_tempered_bond_orders.xyz')

        if not os.path.exists(bond_order_file):
            failed_calcs.append(mpid)

    failed_calc_file=os.path.join(LOG_DIR,'calculations','chargemol','failed_calc.txt')
    with open(failed_calc_file,'w') as f:
        for failed_calc in failed_calcs:
            f.write(failed_calc+'\n')


if __name__=='__main__':
    check_chargmol_calcs()
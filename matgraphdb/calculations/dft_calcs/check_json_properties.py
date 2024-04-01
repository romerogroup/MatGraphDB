import os 
import shutil
from glob import glob

from matgraphdb.utils import LOG_DIR,MP_DIR,DB_CALC_DIR, DB_DIR,TMP_DIR


def check_json_props(prop_name='material_id'):
    os.makedirs(TMP_DIR,exists_ok=True)


    files=glob(DB_DIR + '/*.json') 

    print(f'Checking json poperties located : {DB_CALC_DIR}')

    failed_calcs=[]
    successful_calcs=[]
    for file in files:

        
        

    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # os.makedirs(log_dir)
    # failed_calc_file=os.path.join(log_dir,'failed_calc.txt')
    # with open(failed_calc_file,'w') as f:
    #     for failed_calc in failed_calcs:
    #         f.write(failed_calc+'\n')

    # successful_calc_file=os.path.join(log_dir,'successful_calc.txt')
    # with open(successful_calc_file,'w') as f:
    #     for successful_calc in successful_calcs:
    #         f.write(successful_calc+'\n')


if __name__=='__main__':

    properties=['chargemol_bonding_orders','coordination_environments_multi_weight']
    check_json_props(prop_name='chargemol_bonding_orders')
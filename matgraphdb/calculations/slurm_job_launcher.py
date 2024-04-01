import os
import subprocess

from matgraphdb.data.manager import DatabaseManager

def launch_calcs(slurm_scripts=[]):
    if slurm_scripts != []:
        for slurm_script in slurm_scripts:
            result = subprocess.run(['sbatch', slurm_script], capture_output=False, text=True)

def launch_failed_chargemol_calcs():

    db=DatabaseManager()
    success,failed=db.check_chargemol()

    slurm_scripts=[]

    print(f"About to launch {len(failed)} calculations")
    for path in failed[:]:
        # print(path)
        slurm_script=os.path.join(path,'run.slurm')
        slurm_scripts.append(slurm_script)
        
    launch_calcs(slurm_scripts)



if __name__=='__main__':
    launch_failed_chargemol_calcs()
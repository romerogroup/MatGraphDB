import os
import subprocess

from matgraphdb.data.manager import DBManager

import subprocess

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

def launch_failed_chargemol_calcs():
    """
    Launches failed Chargemol calculations.

    This function retrieves the list of failed Chargemol calculations from the database
    and launches the corresponding SLURM scripts for each failed calculation.

    Returns:
        None
    """
    db = DBManager()
    success, failed = db.check_chargemol()

    slurm_scripts = []

    print(f"About to launch {len(failed)} calculations")
    for path in failed[:]:
        slurm_script = os.path.join(path, 'run.slurm')
        slurm_scripts.append(slurm_script)
        
    launch_calcs(slurm_scripts)



if __name__=='__main__':
    launch_failed_chargemol_calcs()
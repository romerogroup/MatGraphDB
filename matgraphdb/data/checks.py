import os
from glob import glob
import json
from multiprocessing import Pool

from matgraphdb.utils import DB_DIR, N_CORES
from matgraphdb.database.utils import process_database

def check_chemenv_task(file):
    """
    Check if the given file contains the necessary data for the chemenv task.

    Args:
        file (str): The path to the file to be checked.

    Returns:
        tuple: A tuple containing a boolean value indicating the success of the check and the file path.
    """
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        
        if "coordination_environments_multi_weight" not in data:
            success = False
        elif data["coordination_environments_multi_weight"] is not None:
            success = True
        else:
            success = False
    except:
        success = False

    return success, file


def check_chemenv(files):
    """
    Check the chemical environment for a list of files.

    Args:
        files (list): A list of file paths to be processed.

    Returns:
        None

    Prints the number of files that passed and failed the check.
    """
    results = process_database(check_chemenv_task)

    success_files = []
    failed_files = []
    for result in results:
        file = results[1]
        if result[0] == True:
            success_files.append(file)
        elif result[0] == False:
            failed_files.append(file)

    print("Success: ", len(success_files))
    print("Failed: ", len(failed_files))


def check_chargemol_task(file):
    """
    Check if the given file contains the 'chargemol_bonding_orders' key in its JSON data.

    Args:
        file (str): The path to the file to be checked.

    Returns:
        tuple: A tuple containing a boolean indicating the success of the check and the file path.
    """
    with open(file, 'r') as f:
        data = json.load(f)
    
    if "chargemol_bonding_orders" not in data:
        success = False
    elif data["chargemol_bonding_orders"] is not None:
        success = True
    else:
        success = False

    return success, file

def check_chargemol(files):
    """
    Check the chargemol for a list of files.

    Args:
        files (list): A list of file paths to be checked.

    Returns:
        None
    """
    results = process_database(check_chargemol_task)

    success_files = []
    failed_files = []
    for result in results:
        file = results[1]
        if result[0] == True:
            success_files.append(file)
        elif result[0] == False:
            failed_files.append(file)

    print("Success: ", len(success_files))
    print("Failed: ", len(failed_files))


    

def main():


    database_files=glob(DB_DIR + '/*.json')

    print('#'*100)
    print('Checking chemenv analysis success')
    print('#'*100)

    check_chemenv(database_files)

    # print('#'*100)
    # print('Checking chergemol analysis success')
    # print('#'*100)

    # check_chargemol(database_files)

if __name__=='__main__':
    main()

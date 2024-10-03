
import os

from matgraphdb import DBManager
from matgraphdb.utils import EXTERNAL_DATA_DIR

def main():
    manager=DBManager()
    json_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')
    manager.merge_external_database(json_dir=json_dir)


    json_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','bonds_database')
    manager.merge_external_database(json_dir=json_dir, save_key='bonding_info')
    
    
if __name__ == '__main__':
    main()
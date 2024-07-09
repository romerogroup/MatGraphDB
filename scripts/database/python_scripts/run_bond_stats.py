import os

from matgraphdb import DBManager
from matgraphdb.utils import EXTERNAL_DIR

def main():
    manager=DBManager()
    manager.merge_external_database(database_dir=os.path.join(EXTERNAL_DIR,'json_database'))
    
if __name__ == '__main__':
    main()
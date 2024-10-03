import os

from matgraphdb import DBManager
from matgraphdb.utils import EXTERNAL_DATA_DIR

def main():
    manager=DBManager()
    manager.merge_summary_doc()
    manager.merge_elasticity_doc()
    manager.merge_oxidation_states_doc()

    
    
if __name__ == '__main__':
    main()
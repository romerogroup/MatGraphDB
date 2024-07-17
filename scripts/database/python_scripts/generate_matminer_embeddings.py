import os

from matgraphdb import DBManager
from matgraphdb.utils import EXTERNAL_DATA_DIR

def main():
    manager=DBManager()
    feature_sets=[
        ['sine_coulomb_matrix'],
        ['xrd_pattern'],
        ['element_fraction'],
        ['element_property'],
        ['sine_coulomb_matrix','element_property'],
        ['sine_coulomb_matrix','element_fraction'],
        ['element_property','element_fraction'],
        ['sine_coulomb_matrix','element_property','element_fraction'],
    ]
    for feature_set in feature_sets:
        manager.generate_matminer_embeddings(feature_set)

if __name__ == '__main__':
    main()


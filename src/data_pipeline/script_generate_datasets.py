import os

from poly_graphs_lib.data.generate_datasets import generate_datasets
from poly_graphs_lib.utils import ROOT, PROJECT_DIR


if __name__ == '__main__':

    config_file=os.path.join(ROOT,'cfg', 'datasets.yml')
    config_file=os.path.join(PROJECT_DIR,'src','data_pipeline','datasets_config.yml')
    config_file=os.path.join(PROJECT_DIR,'src','data_pipeline','tmp_config.yml')
    generate_datasets(config_file)
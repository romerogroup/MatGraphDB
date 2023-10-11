import os
import yaml

from poly_graphs_lib.utils import PROJECT_DIR

with open(os.path.join(PROJECT_DIR,'poly_graphs_lib','cfg','mp_api.yml'), 'r') as file:
    api_dict = yaml.safe_load(file)


apikey=api_dict['apikey']
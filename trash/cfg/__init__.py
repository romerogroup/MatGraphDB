import os
import yaml

from matgraphdb.utils import PROJECT_DIR

with open(os.path.join(PROJECT_DIR,'poly_graphs_lib','cfg','mp_api.yml'), 'r') as file:
    api_dict = yaml.safe_load(file)


apikey=api_dict['apikey']
import os

from poly_graphs_lib.data.create_datasets import create_material_polyhedra_dataset_2,create_material_polyhedra_dataset_3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class DataGeneratorConfig:

    dataset_dir = f"{PROJECT_DIR}{os.sep}datasets"
    mpcif_data_dir : str = f"{dataset_dir}{os.sep}raw{os.sep}nelement_max_2_nsites_max_6_3d"
    node_type: str = "face"
    val_size : int = 0.10
    def __init__(self):
        if self.node_type not in ['face', 'vert']:
            raise ValueError("node_type must be either 'face' or 'vert'")

class DataGenerator:

    def __init__(self):
        self.config = DataGeneratorConfig() 

    def initialize_ingestion(self):
        try:
            create_material_polyhedra_dataset_3(data_dir=self.config.dataset_dir,
                                            mpcif_data_dir=self.config.mpcif_data_dir,
                                            node_type=self.config.node_type, 
                                            val_size=self.config.val_size)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    obj = DataGenerator()
    obj.initialize_ingestion()
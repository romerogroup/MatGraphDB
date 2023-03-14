import os

from poly_graphs_lib.data.create_datasets import create_material_polyhedra_dataset

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class DataGeneratorConfig:

    dataset_dir = f"{PROJECT_DIR}{os.sep}datasets"
    mpcif_data_dir : str = f"{dataset_dir}{os.sep}raw{os.sep}nelement_max_2_nsites_max_6_3d"
    feature_set_index : int = 3
    val_size : int = 0.10
    def __init__(self):
        pass

class DataGenerator:

    def __init__(self):
        self.config = DataGeneratorConfig()

    def initialize_ingestion(self):
        try:
            create_material_polyhedra_dataset(data_dir=self.config.dataset_dir,
                                            mpcif_data_dir=self.config.mpcif_data_dir,
                                            feature_set_index=self.config.feature_set_index, 
                                            val_size=self.config.val_size)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    obj = DataGenerator()
    obj.initialize_ingestion()